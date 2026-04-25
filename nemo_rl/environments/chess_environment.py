from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

LOCAL_DEPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".deps"
)
if os.path.isdir(LOCAL_DEPS_DIR) and LOCAL_DEPS_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DEPS_DIR)

try:
    import chess
    import chess.engine
except ImportError:  # pragma: no cover - handled during actor initialization
    chess = None


class ChessEnvConfig(TypedDict, total=False):
    stockfish_path: str
    analysis_depth: int
    illegal_move_penalty: float
    mate_bonus: float


class ChessEnvironmentMetadata(TypedDict, total=False):
    fen: str
    extracted_move: str | None
    eval_before: float | None
    eval_after: float | None
    is_legal_move: bool
    delivered_checkmate: bool


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class ChessEnvironment(EnvironmentInterface[ChessEnvironmentMetadata]):
    """Single-step chess environment scored with python-chess + Stockfish."""

    UCI_MOVE_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
    TAGGED_MOVE_PATTERN = re.compile(
        r"(?im)(?:^|\n)\s*move\s*:\s*([a-h][1-8][a-h][1-8][qrbn]?)\b"
    )
    XML_MOVE_PATTERN = re.compile(
        r"(?is)<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>"
    )
    DEFAULT_STOCKFISH_CANDIDATES = (
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    )

    def _resolve_stockfish_path(self, configured_path: str) -> str:
        if os.path.isabs(configured_path):
            if os.path.isfile(configured_path) and os.access(configured_path, os.X_OK):
                return configured_path
            raise FileNotFoundError(
                f"Configured Stockfish binary was not found or is not executable: {configured_path}"
            )

        resolved_path = shutil.which(configured_path)
        if resolved_path is not None:
            return resolved_path

        for candidate in self.DEFAULT_STOCKFISH_CANDIDATES:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        candidate_list = ", ".join(
            [configured_path, *self.DEFAULT_STOCKFISH_CANDIDATES]
        )
        raise FileNotFoundError(
            "Could not find a Stockfish binary. "
            f"Tried command/path '{configured_path}' and common locations: {candidate_list}. "
            "Set env.chess.stockfish_path to the full executable path if needed."
        )

    def __init__(self, cfg: Optional[ChessEnvConfig] = None):
        if chess is None:
            raise ImportError(
                "ChessEnvironment requires the optional 'python-chess' dependency."
            )

        self.cfg = cfg or {}
        self.stockfish_path = self.cfg.get("stockfish_path", "stockfish")
        self.analysis_depth = self.cfg.get("analysis_depth", 15)
        self.illegal_move_penalty = float(self.cfg.get("illegal_move_penalty", 1.0))
        self.mate_bonus = float(self.cfg.get("mate_bonus", 100.0))
        self.stockfish_path = self._resolve_stockfish_path(self.stockfish_path)

        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True
        )

    def _extract_move(self, response: str) -> str | None:
        for pattern in (
            self.XML_MOVE_PATTERN,
            self.TAGGED_MOVE_PATTERN,
            self.UCI_MOVE_PATTERN,
        ):
            match = pattern.search(response)
            if match:
                return match.group(1).lower()
        return None

    def _score_to_float(self, score: Any) -> float:
        centipawns = score.score()
        return float(centipawns) if centipawns is not None else 0.0

    def _evaluate(self, board: Any, perspective: Any) -> float:
        info = self.engine.analyse(
            board, chess.engine.Limit(depth=self.analysis_depth)
        )
        return self._score_to_float(info["score"].pov(perspective))

    def _parse_legal_move(
        self, board: Any, move_text: str | None
    ) -> Any:
        if move_text is None:
            return None

        try:
            move = chess.Move.from_uci(move_text)
        except ValueError:
            move = None

        if move is not None and move in board.legal_moves:
            return move

        try:
            move = board.parse_san(move_text)
        except ValueError:
            return None

        return move if move in board.legal_moves else None

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ChessEnvironmentMetadata],
    ) -> EnvironmentReturn[ChessEnvironmentMetadata]:
        observations: list[dict[str, str]] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        answers: list[str | None] = []
        next_metadata: list[ChessEnvironmentMetadata | None] = []

        for message_log, sample_metadata in zip(message_log_batch, metadata):
            board = chess.Board(sample_metadata["fen"])
            player_to_move = board.turn
            assistant_response = "".join(
                str(message["content"])
                for message in message_log
                if message["role"] == "assistant"
            )
            extracted_move = self._extract_move(assistant_response)
            eval_before = self._evaluate(board, player_to_move)
            legal_move = self._parse_legal_move(board, extracted_move)

            think_end = assistant_response.find("</think>")
            if think_end != -1:
                thinking_text = assistant_response[: think_end + len("</think>")]
                move_text = assistant_response[think_end + len("</think>") :]
            else:
                thinking_text = ""
                move_text = assistant_response
            thinking_tokens = len(self.tokenizer.encode(thinking_text, add_special_tokens=False))
            move_tokens = len(self.tokenizer.encode(move_text, add_special_tokens=False))

            if legal_move is None:
                reward = -self.illegal_move_penalty
                observations.append(
                    {
                        "role": "environment",
                        "content": (
                            f"Illegal move: {extracted_move or 'missing move'}. "
                            f"Reward={reward:.2f}"
                        ),
                    }
                )
                rewards.append(reward)
                terminateds.append(True)
                answers.append(extracted_move)
                next_metadata.append(
                    {
                        "fen": sample_metadata["fen"],
                        "extracted_move": extracted_move,
                        "eval_before": eval_before,
                        "eval_after": eval_before,
                        "is_legal_move": False,
                        "delivered_checkmate": False,
                        "thinking_tokens": thinking_tokens,
                        "move_tokens": move_tokens,
                    }
                )
                continue

            board.push(legal_move)
            eval_after = self._evaluate(board, player_to_move)
            delivered_checkmate = board.is_checkmate()
            reward = eval_after - eval_before
            if delivered_checkmate:
                reward += self.mate_bonus

            observations.append(
                {
                    "role": "environment",
                    "content": (
                        f"Legal move: {legal_move.uci()}. "
                        f"eval_before={eval_before:.2f}, "
                        f"eval_after={eval_after:.2f}, "
                        f"mate_bonus={self.mate_bonus if delivered_checkmate else 0.0:.2f}, "
                        f"reward={reward:.2f}, "
                        f"fen={board.fen()}"
                    ),
                }
            )
            rewards.append(reward)
            terminateds.append(True)
            answers.append(legal_move.uci())
            next_metadata.append(
                {
                    "fen": board.fen(),
                    "extracted_move": legal_move.uci(),
                    "eval_before": eval_before,
                    "eval_after": eval_after,
                    "is_legal_move": True,
                    "delivered_checkmate": delivered_checkmate,
                    "thinking_tokens": thinking_tokens,
                    "move_tokens": move_tokens,
                }
            )

        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def shutdown(self) -> None:
        self.engine.quit()

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        rewards = batch.get("total_reward", batch.get("rewards"))
        if rewards is None or len(rewards) == 0:
            return batch, {}
        rewards = rewards.float()
        n = len(rewards)

        env_infos: list[ChessEnvironmentMetadata] = batch.get("extra_env_info", [{}] * n)
        legal_flags = [bool(info.get("is_legal_move", False)) for info in env_infos]
        checkmate_flags = [bool(info.get("delivered_checkmate", False)) for info in env_infos]
        eval_deltas = [
            float(info["eval_after"]) - float(info["eval_before"])
            for info in env_infos
            if info.get("eval_after") is not None and info.get("eval_before") is not None
        ]

        truncated = batch.get("truncated")
        if truncated is not None:
            is_max_tokens = [bool(t) for t in truncated]
        else:
            is_max_tokens = [False] * n

        thinking_tokens = [int(info.get("thinking_tokens", 0)) for info in env_infos]
        move_tokens = [int(info.get("move_tokens", 0)) for info in env_infos]

        legal_rate = sum(legal_flags) / n
        avg_centipawn_delta = sum(eval_deltas) / len(eval_deltas) if eval_deltas else 0.0

        return batch, {
            "env/avg_reward": rewards.mean().item(),
            "env/legal_move_rate": legal_rate,
            "env/illegal_move_rate": 1.0 - legal_rate,
            "env/checkmate_rate": sum(checkmate_flags) / n,
            "env/avg_centipawn_delta": avg_centipawn_delta,
            "env/avg_thinking_tokens": sum(thinking_tokens) / n,
            "env/avg_move_tokens": sum(move_tokens) / n,
            "env/truncated_rate": sum(is_max_tokens) / n,
        }
