# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pprint
import random
import sys
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.utils import create_env, register_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


LOCAL_DEPS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".deps")
if os.path.isdir(LOCAL_DEPS_DIR) and LOCAL_DEPS_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DEPS_DIR)
    os.environ["PYTHONPATH"] = (
        LOCAL_DEPS_DIR
        if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]
        else f"{LOCAL_DEPS_DIR}:{os.environ['PYTHONPATH']}"
    )

try:
    import chess
except ImportError as e:  # pragma: no cover - import verified at runtime
    raise ImportError(
        "run_grpo_chess.py requires the optional 'python-chess' dependency.\n"
        "Install it in the active environment before launching training.\n"
        "If you are using the NeMo RL repo venv, one working option is:\n"
        "  uv pip install python-chess\n"
        "If you are using a system or container Python instead, use:\n"
        "  python -m pip install python-chess"
    ) from e


CHESS_ENV_FQN = "nemo_rl.environments.chess_environment.ChessEnvironment"


def ensure_chess_env_registered() -> None:
    try:
        register_env("chess", CHESS_ENV_FQN)
    except ValueError as e:
        if "already registered" not in str(e):
            raise


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one-step GRPO training with the chess environment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def load_positions(path: str) -> list[dict[str, Any]]:
    if not os.path.isabs(path):
        repo_relative_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), path
        )
        if os.path.exists(repo_relative_path):
            path = repo_relative_path

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError(f"No positions found in {path}")

    for i, row in enumerate(rows):
        if "fen" not in row:
            raise ValueError(f"Row {i} in {path} is missing required key 'fen'")
        chess.Board(row["fen"])

    return rows


SYSTEM_PROMPT = """You are a competitive chess agent.
Reason about the current position and then choose exactly one legal move.
Always respond in this exact format:
ANALYSIS:
<short reasoning>
MOVE: <uci_move>

Only output a legal UCI move from the provided legal move list."""

SYSTEM_PROMPT_GUIDED = """You are a grandmaster-level chess engine. Before choosing a move, you MUST reason through the position step by step.

Always respond in exactly this format:
ANALYSIS:
<your step-by-step reasoning>
MOVE: <uci_move>

Your reasoning MUST cover these steps in order (max 2 sentences each):
1. THREATS: What is my opponent threatening? Any checks, captures, or mate threats I must address?
2. MATERIAL: What is the material balance? Any pieces hanging or en prise?
3. KING SAFETY: Is my king safe? Is the opponent's king exposed?
4. CANDIDATES: List 3-5 candidate moves from the legal move list and briefly evaluate each.
5. DECISION: Which candidate is best and why?

Your final MOVE must be one of the provided legal moves, in UCI format (e.g. e2e4, e7e8q). Do not invent moves.
"""


def _board_with_coords(board_str: str, flip: bool = False) -> str:
    """Add rank numbers and file letters to a bare python-chess str(board) string.

    flip=True renders from Black's perspective: rank 8 at bottom, rank 1 at top,
    h-file on the left.
    """
    rows = board_str.splitlines()  # 8 rows, rank 8 first (index 0 = rank 8)
    if flip:
        # Black POV: reverse row order so rank 1 is at top, rank 8 at bottom
        rows = list(reversed(rows))
        # Mirror pieces left-right within each row (h-file on left)
        rows = [" ".join(reversed(r.split())) for r in rows]
        ranks = list(range(1, 9))   # 1 at top, 8 at bottom
        files = list("hgfedcba")
    else:
        ranks = list(range(8, 0, -1))  # 8 at top, 1 at bottom
        files = list("abcdefgh")
    labeled = [f"{ranks[i]} | {rows[i]}" for i in range(8)]
    file_row = "  +-----------------"
    file_labels = "    " + " ".join(files)
    return "\n".join(labeled) + "\n" + file_row + "\n" + file_labels


def _render_prompt_json(turn: str, fen: str, board_ascii: str, legal_moves: list[str]) -> str:
    import json as _json
    payload = {
        "side_to_move": turn.lower(),
        "fen": fen,
        "board": board_ascii,
        "legal_moves": legal_moves,
        "instructions": {
            "required_output": "Use the exact template with ANALYSIS and MOVE.",
            "move_format": "Move must be UCI, e.g. e2e4 or e7e8q.",
            "legality_requirement": "Choose only from the provided legal_moves list.",
        },
    }
    return _json.dumps(payload, indent=2)


def _render_prompt_plain(turn: str, fen: str, board_ascii: str, legal_moves: list[str]) -> str:
    return "\n".join([
        f"You are playing {turn}. It is {turn} to move.",
        "",
        f"FEN: {fen}",
        "",
        f"Legal moves: {', '.join(legal_moves)}",
    ])


def _render_prompt_compact(turn: str, fen: str, board_ascii: str, legal_moves: list[str]) -> str:
    return (
        f"side={turn.lower()} "
        f"fen={fen} "
        f"legal={','.join(legal_moves)}"
    )


def _render_prompt_guided(turn: str, fen: str, board_ascii: str, legal_moves: list[str]) -> str:
    is_black = turn.lower() == "black"
    board_str = _board_with_coords(board_ascii, flip=is_black)
    perspective = "shown from Black's perspective (rank 1 at top)" if is_black else "shown from White's perspective (rank 8 at top)"
    return "\n".join([
        f"You are playing {turn}.",
        "",
        f"BOARD ({perspective}):",
        board_str,
        "",
        f"FEN: {fen}",
        "",
        f"Legal moves ({len(legal_moves)} available): {', '.join(legal_moves)}",
        "",
        "Work through the 5-step reasoning process in your ANALYSIS, then output your MOVE.",
    ])


PROMPT_RENDERERS = {
    "json": _render_prompt_json,
    "plain": _render_prompt_plain,
    "compact": _render_prompt_compact,
    "guided": _render_prompt_guided,
}

DEFAULT_PROMPT_STYLE = "guided"


def format_chess_prompt(
    fen: str,
    board: chess.Board,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    prompt_template: Optional[str] = None,
) -> str:
    legal_moves = [move.uci() for move in board.legal_moves]
    board_ascii = str(board)
    turn = "White" if board.turn == chess.WHITE else "Black"

    if prompt_template is not None:
        return prompt_template.format(
            fen=fen,
            board=board_ascii,
            turn=turn,
            legal_moves=" ".join(legal_moves),
        )

    renderer = PROMPT_RENDERERS.get(prompt_style)
    if renderer is None:
        raise ValueError(
            f"Unknown prompt_style '{prompt_style}'. Available: {sorted(PROMPT_RENDERERS)}"
        )
    return renderer(turn, fen, board_ascii, legal_moves)


def build_chess_datum(
    tokenizer,
    sample: dict[str, Any],
    idx: int,
    task_name: str,
    add_system_prompt: bool,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    prompt_template: Optional[str] = None,
) -> DatumSpec:
    fen = sample["fen"]
    board = chess.Board(fen)
    prompt = sample.get("prompt") or format_chess_prompt(
        fen=fen,
        board=board,
        prompt_style=prompt_style,
        prompt_template=prompt_template,
    )

    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    tokenized_prompt = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": rendered_prompt,
            "token_ids": tokenized_prompt,
        }
    ]

    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": {"fen": fen},
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
    }
    return datum


class IterableChessDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        positions: Sequence[dict[str, Any]],
        task_name: str,
        add_system_prompt: bool,
        length: int,
        seed: int,
        shuffle_positions: bool,
        prompt_style: str = DEFAULT_PROMPT_STYLE,
        prompt_template: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.positions = list(positions)
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length
        self.seed = seed
        self.shuffle_positions = shuffle_positions
        self.prompt_style = prompt_style
        self.prompt_template = prompt_template

    def __iter__(self) -> Iterator[DatumSpec]:
        rng = random.Random(self.seed)
        positions = list(self.positions)
        datum_idx = 0

        while True:
            if self.shuffle_positions:
                rng.shuffle(positions)

            for sample in positions:
                yield build_chess_datum(
                    tokenizer=self.tokenizer,
                    sample=sample,
                    idx=datum_idx,
                    task_name=self.task_name,
                    add_system_prompt=self.add_system_prompt,
                    prompt_style=self.prompt_style,
                    prompt_template=self.prompt_template,
                )
                datum_idx += 1

    def __len__(self) -> int:
        return self.length


def setup_chess_data(
    tokenizer,
    config: MasterConfig,
    task_name: str,
) -> tuple[IterableChessDataset, Optional[IterableChessDataset], dict, dict]:
    data_cfg = config["data"]
    env_cfg = config["env"]["chess"]
    train_cfg = data_cfg["train"]
    val_cfg = data_cfg.get("validation")

    train_positions = load_positions(train_cfg["data_path"])
    prompt_style = data_cfg.get("prompt_style", DEFAULT_PROMPT_STYLE)
    prompt_template = data_cfg.get("prompt_template")
    add_system_prompt = data_cfg.get("add_system_prompt", False)
    shuffle_positions = data_cfg.get("shuffle_positions", True)

    train_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    val_length = config["grpo"]["max_val_samples"]

    dataset = IterableChessDataset(
        tokenizer=tokenizer,
        positions=train_positions,
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=train_length,
        seed=config["grpo"]["seed"],
        shuffle_positions=shuffle_positions,
        prompt_style=prompt_style,
        prompt_template=prompt_template,
    )

    val_dataset = None
    if val_cfg is not None:
        val_positions = load_positions(val_cfg["data_path"])
        val_dataset = IterableChessDataset(
            tokenizer=tokenizer,
            positions=val_positions,
            task_name=task_name,
            add_system_prompt=add_system_prompt,
            length=val_length,
            seed=config["grpo"]["seed"] + 1,
            shuffle_positions=False,
            prompt_style=prompt_style,
            prompt_template=prompt_template,
        )

    env = create_env("chess", env_cfg)
    task_to_env = {task_name: env}
    val_task_to_env = task_to_env if val_dataset is not None else {}
    return dataset, val_dataset, task_to_env, val_task_to_env


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_chess_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")

    init_ray()
    set_seed(config["grpo"]["seed"])
    ensure_chess_env_registered()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    task_name = config["data"].get("task_name", "chess_one_step")
    if "default" not in config["data"] or config["data"]["default"] is None:
        config["data"]["default"] = {"env_name": "chess"}
    else:
        config["data"]["default"].setdefault("env_name", "chess")

    dataset, val_dataset, task_to_env, val_task_to_env = setup_chess_data(
        tokenizer=tokenizer,
        config=config,
        task_name=task_name,
    )

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
