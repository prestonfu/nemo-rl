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
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, TypedDict

import chess
import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import set_seed
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.utils import create_env, register_env
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

from run_grpo_chess import format_chess_prompt, load_positions


CHESS_ENV_FQN = "nemo_rl.environments.chess_environment.ChessEnvironment"


class MockEvalConfig(TypedDict, total=False):
    seed: int
    num_tests_per_prompt: int
    save_path: str | None


class ModelConfig(TypedDict, total=False):
    mode: str
    model: str


@dataclass
class GenerationResult:
    text: str
    reasoning_content: str
    finish_reason: str
    latency_s: float
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class MockChessClient:
    def __init__(self, model_name: str = "mock-chess"):
        self.model_name = model_name

    def generate_move(self, prompt_payload: Dict[str, object]) -> GenerationResult:
        legal_moves: list[str] = list(prompt_payload["legal_moves"])
        start = time.perf_counter()
        move = self._choose_move(legal_moves)
        analysis = (
            "I considered captures, checks, promotions, and centralizing moves. "
            "I selected a legal move that preserves material and activity."
        )
        text = f"ANALYSIS:\n{analysis}\nMOVE: {move}\n"
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": len(text.split()),
            "total_tokens": len(text.split()),
        }
        return GenerationResult(
            text=text,
            reasoning_content=analysis,
            finish_reason="stop",
            latency_s=time.perf_counter() - start,
            model=self.model_name,
            usage=usage,
        )

    def _choose_move(self, legal_moves: list[str]) -> str:
        promotions = [move for move in legal_moves if len(move) == 5]
        captures = [move for move in legal_moves if move[2:4] != move[:2]]
        center_targets = {"d4", "e4", "d5", "e5", "c4", "f4", "c5", "f5"}
        center = [move for move in legal_moves if move[2:4] in center_targets]
        for bucket in (promotions, captures, center, legal_moves):
            if bucket:
                return random.choice(bucket)
        return legal_moves[0]


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        raise NotImplementedError(
            "OpenAICompatibleClient is not implemented in this script. "
            "Use eval_client.mode=mock, or extend this script with your remote client."
        )


def build_client(config: ModelConfig):
    if config["mode"] == "mock":
        return MockChessClient(model_name=config["model"])
    return OpenAICompatibleClient(config)


def ensure_chess_env_registered() -> None:
    try:
        register_env("chess", CHESS_ENV_FQN)
    except ValueError as e:
        if "already registered" not in str(e):
            raise


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate chess positions with a mock chess client"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a chess config file",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _get_eval_config(config: dict[str, Any]) -> MockEvalConfig:
    eval_cfg = dict(config.get("eval", {}))
    eval_cfg.setdefault("seed", config.get("grpo", {}).get("seed", 42))
    eval_cfg.setdefault("num_tests_per_prompt", 1)
    eval_cfg.setdefault("save_path", None)
    return eval_cfg


def _get_client_config(config: dict[str, Any]) -> ModelConfig:
    client_cfg = dict(config.get("eval_client", {}))
    client_cfg.setdefault("mode", "mock")
    client_cfg.setdefault("model", "mock-chess")
    return client_cfg


def _build_prompt_payload(
    sample: dict[str, Any],
    prompt_template: Optional[str],
) -> dict[str, object]:
    fen = sample["fen"]
    board = chess.Board(fen)
    prompt = sample.get("prompt") or format_chess_prompt(
        fen=fen,
        board=board,
        prompt_template=prompt_template,
    )
    turn = "White" if board.turn == chess.WHITE else "Black"
    legal_moves = [move.uci() for move in board.legal_moves]
    return {
        "fen": fen,
        "board": str(board),
        "turn": turn,
        "prompt": prompt,
        "legal_moves": legal_moves,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _save_results(
    save_path: str,
    summary: dict[str, Any],
    sample_results: list[dict[str, Any]],
) -> None:
    os.makedirs(save_path, exist_ok=True)

    summary_path = os.path.join(save_path, "summary.json")
    results_path = os.path.join(save_path, "samples.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        for row in sample_results:
            f.write(json.dumps(row) + "\n")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved sample-level results to: {results_path}")


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

    eval_cfg = _get_eval_config(config)
    client_cfg = _get_client_config(config)
    print("Mock chess eval settings:")
    pprint.pprint(eval_cfg)
    print("Client settings:")
    pprint.pprint(client_cfg)

    init_ray()
    ensure_chess_env_registered()
    set_seed(eval_cfg["seed"])
    random.seed(eval_cfg["seed"])

    eval_source = config["data"].get("validation") or config["data"].get("train")
    if eval_source is None or "data_path" not in eval_source:
        raise ValueError(
            "Chess evaluation requires data.validation.data_path or data.train.data_path"
        )

    positions = load_positions(eval_source["data_path"])
    prompt_template = config["data"].get("prompt_template")
    env = create_env("chess", config["env"]["chess"])
    client = build_client(client_cfg)

    num_tests_per_prompt = eval_cfg["num_tests_per_prompt"]
    sample_results: list[dict[str, Any]] = []

    try:
        for sample_idx, sample in enumerate(positions):
            payload = _build_prompt_payload(sample, prompt_template)
            for test_idx in range(num_tests_per_prompt):
                result = client.generate_move(payload)
                message_log = [
                    {"role": "user", "content": str(payload["prompt"])},
                    {"role": "assistant", "content": result.text},
                ]
                metadata = [{"fen": str(payload["fen"])}]
                env_return = ray.get(env.step.remote([message_log], metadata))
                sample_metadata = env_return.metadata[0]
                reward = float(env_return.rewards.tolist()[0])
                answer = env_return.answers[0]

                sample_results.append(
                    {
                        "sample_index": len(sample_results),
                        "dataset_index": sample_idx,
                        "test_index": test_idx,
                        "fen": payload["fen"],
                        "turn": payload["turn"],
                        "legal_moves": payload["legal_moves"],
                        "prompt": payload["prompt"],
                        "response": result.text,
                        "generation": asdict(result),
                        "reward": reward,
                        "answer": answer,
                        "extracted_move": (
                            sample_metadata.get("extracted_move")
                            if sample_metadata
                            else None
                        ),
                        "eval_before": (
                            sample_metadata.get("eval_before")
                            if sample_metadata
                            else None
                        ),
                        "eval_after": (
                            sample_metadata.get("eval_after")
                            if sample_metadata
                            else None
                        ),
                        "is_legal_move": (
                            sample_metadata.get("is_legal_move")
                            if sample_metadata
                            else None
                        ),
                        "delivered_checkmate": (
                            sample_metadata.get("delivered_checkmate")
                            if sample_metadata
                            else None
                        ),
                    }
                )
    finally:
        ray.get(env.shutdown.remote())

    rewards = [float(row["reward"]) for row in sample_results]
    legal_flags = [bool(row["is_legal_move"]) for row in sample_results]
    mate_flags = [bool(row["delivered_checkmate"]) for row in sample_results]
    latencies = [float(row["generation"]["latency_s"]) for row in sample_results]
    eval_deltas = [
        float(row["eval_after"]) - float(row["eval_before"])
        for row in sample_results
        if row["eval_before"] is not None and row["eval_after"] is not None
    ]

    summary: dict[str, Any] = {
        "client_mode": client_cfg["mode"],
        "model_name": client_cfg["model"],
        "data_path": eval_source["data_path"],
        "num_positions": len(positions),
        "num_samples": len(sample_results),
        "num_tests_per_prompt": num_tests_per_prompt,
        "avg_reward": _mean(rewards),
        "legal_move_rate": _mean([1.0 if flag else 0.0 for flag in legal_flags]),
        "checkmate_rate": _mean([1.0 if flag else 0.0 for flag in mate_flags]),
        "avg_latency_s": _mean(latencies),
        "avg_eval_delta": _mean(eval_deltas),
    }

    if num_tests_per_prompt > 1:
        grouped_results = [
            sample_results[i : i + num_tests_per_prompt]
            for i in range(0, len(sample_results), num_tests_per_prompt)
        ]
        summary["best_of_n_avg_reward"] = _mean(
            [max(float(row["reward"]) for row in group) for group in grouped_results]
        )
        summary["any_legal_move_rate"] = _mean(
            [
                1.0 if any(bool(row["is_legal_move"]) for row in group) else 0.0
                for group in grouped_results
            ]
        )

    print("\n" + "=" * 60)
    print("Mock Chess Evaluation Results")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")

    save_path = eval_cfg.get("save_path")
    if save_path:
        _save_results(save_path, summary, sample_results)


if __name__ == "__main__":
    main()
