# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import re
from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class SudokuEnvConfig(TypedDict, total=False):
    correct_reward: float
    incorrect_reward: float
    format_error_reward: float


class SudokuEnvironmentMetadata(TypedDict, total=False):
    solution: str   # 81-char ground-truth solution string, digits 1-9
    puzzle: str     # 81-char puzzle string, 0 = empty


def _parse_grid(text: str) -> str | None:
    """Extract 81 digits from model output (9 rows of 9 digits each)."""
    rows = re.findall(r"[1-9]{9}", text)
    if len(rows) >= 9:
        return "".join(rows[:9])
    # fallback: strip all non-digits and take first 81
    digits = re.sub(r"[^1-9]", "", text)
    if len(digits) == 81:
        return digits
    return None


def _check_sudoku(grid: str) -> bool:
    """Return True if the 81-char grid is a valid solved Sudoku."""
    if len(grid) != 81 or not grid.isdigit() or "0" in grid:
        return False
    for i in range(9):
        row = set(grid[i * 9 : i * 9 + 9])
        col = set(grid[i::9])
        box_r, box_c = (i // 3) * 3, (i % 3) * 3
        box = set(grid[box_r * 9 + box_c + r * 9 + c] for r in range(3) for c in range(3))
        if row != set("123456789") or col != set("123456789") or box != set("123456789"):
            return False
    return True


def _cell_accuracy(predicted: str, solution: str) -> float:
    if len(predicted) != 81 or len(solution) != 81:
        return 0.0
    return sum(p == s for p, s in zip(predicted, solution)) / 81.0


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class SudokuEnvironment(EnvironmentInterface[SudokuEnvironmentMetadata]):
    """Single-turn Sudoku environment. Rewards exact solution correctness."""

    def __init__(self, cfg: Optional[SudokuEnvConfig] = None):
        self.cfg = cfg or {}
        self.correct_reward = float(self.cfg.get("correct_reward", 1.0))
        self.incorrect_reward = float(self.cfg.get("incorrect_reward", 0.0))
        self.format_error_reward = float(self.cfg.get("format_error_reward", -1.0))

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[SudokuEnvironmentMetadata],
    ) -> EnvironmentReturn[SudokuEnvironmentMetadata]:
        observations = []
        rewards = []
        terminateds = []
        answers = []
        next_metadata = []

        for message_log, sample_metadata in zip(message_log_batch, metadata):
            solution = sample_metadata["solution"]
            assistant_response = "".join(
                str(msg["content"])
                for msg in message_log
                if msg["role"] == "assistant"
            )

            predicted = _parse_grid(assistant_response)

            if predicted is None:
                reward = self.format_error_reward
                obs = f"Could not parse a valid 9×9 grid from your response. reward={reward:.2f}"
                acc = 0.0
                correct = False
            elif predicted == solution:
                reward = self.correct_reward
                obs = f"Correct! reward={reward:.2f}"
                acc = 1.0
                correct = True
            else:
                acc = _cell_accuracy(predicted, solution)
                valid = _check_sudoku(predicted)
                reward = self.incorrect_reward
                obs = (
                    f"Incorrect solution. cell_accuracy={acc:.3f}, "
                    f"valid_sudoku={valid}, reward={reward:.2f}"
                )
                correct = False

            observations.append({"role": "environment", "content": obs})
            rewards.append(reward)
            terminateds.append(True)
            answers.append(predicted)
            next_metadata.append({
                **sample_metadata,
                "predicted": predicted,
                "correct": correct,
                "cell_accuracy": acc,
            })

        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def shutdown(self) -> None:
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict[str, float]]:
        rewards = batch.get("total_reward", batch.get("rewards"))
        if rewards is None or len(rewards) == 0:
            return batch, {}
        rewards = rewards.float()
        n = len(rewards)

        env_infos: list[dict] = batch.get("extra_env_info", [{}] * n)
        correct_flags = [bool(info.get("correct", False)) for info in env_infos]
        cell_accs = [float(info.get("cell_accuracy", 0.0)) for info in env_infos]
        parse_failures = [info.get("predicted") is None for info in env_infos]

        truncated = batch.get("truncated")
        truncated_flags = [bool(t) for t in truncated] if truncated is not None else [False] * n

        return batch, {
            "env/avg_reward": rewards.mean().item(),
            "env/correct_rate": sum(correct_flags) / n,
            "env/avg_cell_accuracy": sum(cell_accs) / n,
            "env/format_error_rate": sum(parse_failures) / n,
            "env/truncated_rate": sum(truncated_flags) / n,
        }
