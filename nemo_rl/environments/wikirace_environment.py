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

import random
import re
from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

SYSTEM_PROMPT = "You are a helpful assistant helping play the Wikipedia link game."

USER_PROMPT_TEMPLATE = """\
You are playing a game where you start at Wikipedia page "{start_page}" and want to reach page "{target_page}" by clicking links.
So far, you have visited the following pages in order:
{history}
You see the following possible links from the current page:
{link_list}
Which link should you click to get closer to the target? Reply with just the number of your choice (0 to {max_choice_num}).\
"""


class WikiRaceEnvConfig(TypedDict, total=False):
    dataset: str           # HuggingFace dataset id with article/links columns
    num_links_shown: int   # number of neighbor links shown per turn (default 50)
    win_reward: float
    step_penalty: float    # per-step penalty to encourage shorter paths
    invalid_choice_penalty: float
    max_steps: int         # hard cap before forced termination


class WikiRaceEnvironmentMetadata(TypedDict, total=False):
    start: str
    goal: str
    current: str
    history: list[str]       # pages visited so far including current
    shortest_path_length: int
    won: bool
    num_steps: int
    # links shown on the last turn (needed to resolve model's choice)
    links_shown: list[str]


def _build_graph(dataset_id: str) -> dict[str, list[str]]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split="train")
    graph: dict[str, list[str]] = {}
    all_titles: set[str] = set()
    for row in ds:
        graph[row["article"]] = row["links"]
        all_titles.add(row["article"])
    for title in graph:
        graph[title] = [l for l in graph[title] if l in all_titles]
    return graph


def _extract_choice(response: str) -> int | None:
    matches = re.findall(r"\b(\d+)\b", response.strip())
    if matches:
        return int(matches[-1])
    return None


def _make_user_prompt(
    start: str,
    goal: str,
    history: list[str],
    links: list[str],
) -> str:
    history_str = " → ".join(history)
    link_list = "\n".join(f"{i}. {title}" for i, title in enumerate(links))
    return USER_PROMPT_TEMPLATE.format(
        start_page=start,
        target_page=goal,
        history=history_str,
        link_list=link_list,
        max_choice_num=len(links) - 1,
    )


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class WikiRaceEnvironment(EnvironmentInterface[WikiRaceEnvironmentMetadata]):
    """Multi-turn WikiRace environment.

    Each turn the model sees the current page's links and picks one by index.
    The episode ends when the model reaches the goal or exceeds max_steps.

    The dataset is loaded once at init and the graph lives in memory
    (~350k nodes for simplewiki-pruned-350k, fits easily in RAM).
    """

    def __init__(self, cfg: Optional[WikiRaceEnvConfig] = None):
        self.cfg = cfg or {}
        dataset_id = self.cfg.get("dataset", "HuggingFaceTB/simplewiki-pruned-350k")
        self.num_links_shown = int(self.cfg.get("num_links_shown", 50))
        self.win_reward = float(self.cfg.get("win_reward", 1.0))
        self.step_penalty = float(self.cfg.get("step_penalty", 0.0))
        self.invalid_choice_penalty = float(self.cfg.get("invalid_choice_penalty", -0.1))
        self.max_steps = int(self.cfg.get("max_steps", 20))

        print(f"WikiRaceEnvironment: loading graph from {dataset_id}...")
        self.graph = _build_graph(dataset_id)
        print(f"WikiRaceEnvironment: {len(self.graph)} nodes loaded.")

    def _sample_links(self, page: str, goal: str, rng: random.Random) -> list[str]:
        """Return up to num_links_shown neighbors, always including goal if reachable."""
        neighbors = list(self.graph.get(page, []))
        if goal in neighbors:
            # guarantee goal is visible if it's a direct neighbor
            others = [n for n in neighbors if n != goal]
            rng.shuffle(others)
            shown = [goal] + others[: self.num_links_shown - 1]
            rng.shuffle(shown)
        else:
            rng.shuffle(neighbors)
            shown = neighbors[: self.num_links_shown]
        return shown

    def make_initial_message_log(
        self, metadata: WikiRaceEnvironmentMetadata
    ) -> LLMMessageLogType:
        """Build the opening message log for a new episode."""
        links = self._sample_links(
            metadata["current"],
            metadata["goal"],
            random.Random(hash(metadata["start"] + metadata["goal"])),
        )
        metadata["links_shown"] = links
        user_content = _make_user_prompt(
            metadata["start"], metadata["goal"], metadata["history"], links
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[WikiRaceEnvironmentMetadata],
    ) -> EnvironmentReturn[WikiRaceEnvironmentMetadata]:
        observations = []
        rewards = []
        terminateds = []
        answers = []
        next_metadata = []

        for message_log, meta in zip(message_log_batch, metadata):
            assistant_response = "".join(
                str(msg["content"])
                for msg in message_log
                if msg["role"] == "assistant"
            )

            links_shown = meta["links_shown"]
            choice = _extract_choice(assistant_response)

            # Invalid choice
            if choice is None or choice < 0 or choice >= len(links_shown):
                reward = self.invalid_choice_penalty
                obs = (
                    f"Invalid choice '{assistant_response.strip()[:40]}'. "
                    f"Please reply with a number between 0 and {len(links_shown) - 1}. "
                    f"reward={reward:.2f}"
                )
                observations.append({"role": "environment", "content": obs})
                rewards.append(reward)
                terminateds.append(True)
                answers.append(None)
                next_metadata.append({**meta, "won": False})
                continue

            chosen_page = links_shown[choice]
            new_history = meta["history"] + [chosen_page]
            num_steps = meta["num_steps"] + 1
            won = chosen_page == meta["goal"]
            at_max = num_steps >= self.max_steps

            reward = self.win_reward if won else -self.step_penalty
            terminated = won or at_max

            if won:
                obs = (
                    f"You reached '{chosen_page}'! Goal reached in {num_steps} steps "
                    f"(shortest path was {meta['shortest_path_length']}). reward={reward:.2f}"
                )
            elif at_max:
                obs = (
                    f"Moved to '{chosen_page}'. Max steps ({self.max_steps}) reached. "
                    f"Goal was '{meta['goal']}'. reward={reward:.2f}"
                )
            else:
                # Continuing — show the next page's links
                new_links = self._sample_links(
                    chosen_page,
                    meta["goal"],
                    random.Random(hash(chosen_page + meta["goal"] + str(num_steps))),
                )
                next_prompt = _make_user_prompt(
                    meta["start"], meta["goal"], new_history, new_links
                )
                obs = (
                    f"Moved to '{chosen_page}'. reward={reward:.2f}\n\n{next_prompt}"
                )

            updated_meta: WikiRaceEnvironmentMetadata = {
                **meta,
                "current": chosen_page,
                "history": new_history,
                "num_steps": num_steps,
                "won": won,
                "links_shown": new_links if (not terminated) else [],
            }

            observations.append({"role": "environment", "content": obs})
            rewards.append(reward)
            terminateds.append(terminated)
            answers.append(chosen_page)
            next_metadata.append(updated_meta)

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

        env_infos: list[WikiRaceEnvironmentMetadata] = batch.get("extra_env_info", [{}] * n)
        won_flags = [bool(info.get("won", False)) for info in env_infos]
        num_steps = [int(info.get("num_steps", 0)) for info in env_infos]
        spl_list = [
            int(info["shortest_path_length"])
            for info in env_infos
            if info.get("shortest_path_length")
        ]
        steps_list = [
            info["num_steps"]
            for info in env_infos
            if info.get("won") and info.get("num_steps")
        ]

        # Success-weighted Path Length (SPL): quality of winning paths
        spl = (
            sum(
                opt / max(act, opt)
                for opt, act, won in zip(spl_list, num_steps, won_flags)
                if won
            ) / n
            if spl_list else 0.0
        )

        truncated = batch.get("truncated")
        truncated_flags = [bool(t) for t in truncated] if truncated is not None else [False] * n

        return batch, {
            "env/avg_reward": rewards.mean().item(),
            "env/win_rate": sum(won_flags) / n,
            "env/avg_steps": sum(num_steps) / n,
            "env/avg_steps_on_win": sum(steps_list) / len(steps_list) if steps_list else 0.0,
            "env/spl": spl,
            "env/truncated_rate": sum(truncated_flags) / n,
        }
