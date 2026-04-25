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
import asyncio
import json
import os
import pprint
import sys
from copy import deepcopy
from typing import Any, Optional, TypedDict

import ray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.collate_fn import eval_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.utils import create_env, register_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_grpo_chess import build_chess_datum, load_positions  # noqa: E402


CHESS_ENV_FQN = "nemo_rl.environments.chess_environment.ChessEnvironment"


class ChessEvalConfig(TypedDict, total=False):
    batch_size: int
    num_tests_per_prompt: int
    seed: int
    save_path: str | None


class ChessEvalDataset(Dataset[DatumSpec]):
    def __init__(
        self,
        tokenizer,
        positions: list[dict[str, Any]],
        task_name: str,
        add_system_prompt: bool,
        prompt_template: Optional[str] = None,
    ):
        self.samples = [
            build_chess_datum(
                tokenizer=tokenizer,
                sample=sample,
                idx=idx,
                task_name=task_name,
                add_system_prompt=add_system_prompt,
                prompt_template=prompt_template,
            )
            for idx, sample in enumerate(positions)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatumSpec:
        return self.samples[idx]


def ensure_chess_env_registered() -> None:
    try:
        register_env("chess", CHESS_ENV_FQN)
    except ValueError as e:
        if "already registered" not in str(e):
            raise


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on chess positions using ChessEnvironment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a chess config file",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _get_eval_config(config: dict[str, Any]) -> ChessEvalConfig:
    eval_cfg = deepcopy(config.get("eval", {}))
    eval_cfg.setdefault("batch_size", 8)
    eval_cfg.setdefault("num_tests_per_prompt", 1)
    eval_cfg.setdefault("seed", config.get("grpo", {}).get("seed", 42))
    eval_cfg.setdefault("save_path", None)
    return eval_cfg


def _setup_dataset(tokenizer, config: dict[str, Any]) -> ChessEvalDataset:
    data_cfg = config["data"]
    task_name = data_cfg.get("task_name", "chess_one_step")

    eval_source = data_cfg.get("validation") or data_cfg.get("train")
    if eval_source is None or "data_path" not in eval_source:
        raise ValueError(
            "Chess evaluation requires data.validation.data_path or data.train.data_path"
        )

    positions = load_positions(eval_source["data_path"])
    return ChessEvalDataset(
        tokenizer=tokenizer,
        positions=positions,
        task_name=task_name,
        add_system_prompt=data_cfg.get("add_system_prompt", False),
        prompt_template=data_cfg.get("prompt_template"),
    )


def _setup_generation(
    config: dict[str, Any],
    tokenizer,
) -> tuple[VllmGeneration, RayVirtualCluster, dict[str, Any]]:
    cluster_cfg = config["cluster"]
    generation_cfg = deepcopy(config["policy"]["generation"])
    generation_cfg["model_name"] = config["policy"]["model_name"]
    generation_cfg = configure_generation_config(
        generation_cfg,
        tokenizer,
        is_eval=True,
    )

    cluster = RayVirtualCluster(
        name="chess_eval_cluster",
        bundle_ct_per_node_list=[cluster_cfg["gpus_per_node"]] * cluster_cfg["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_cfg["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    generation = VllmGeneration(cluster=cluster, config=generation_cfg)
    return generation, cluster, generation_cfg


async def _generate_texts(
    generation: VllmGeneration,
    inputs: BatchedDataDict[Any],
    use_async: bool,
) -> list[str]:
    if use_async:
        results = []
        async for idx, result in generation.generate_text_async(inputs):
            results.append((idx, result["texts"][0]))
        results.sort(key=lambda x: x[0])
        return [text for _, text in results]

    return generation.generate_text(inputs)["texts"]


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


async def _run_eval_impl(
    generation: VllmGeneration,
    dataloader: DataLoader,
    env,
    eval_cfg: ChessEvalConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    num_tests_per_prompt = eval_cfg["num_tests_per_prompt"]
    use_async = generation.cfg["vllm_cfg"]["async_engine"]

    sample_results: list[dict[str, Any]] = []

    generation.prepare_for_generation()
    try:
        for batch in dataloader:
            if num_tests_per_prompt > 1:
                batch = batch.repeat_interleave(num_tests_per_prompt)

            prompts = []
            for message_log in batch["message_log"]:
                prompts.append(
                    "\n".join(str(message["content"]) for message in message_log)
                )

            outputs = await _generate_texts(
                generation,
                BatchedDataDict({"prompts": prompts}),
                use_async=use_async,
            )

            for idx, output in enumerate(outputs):
                batch["message_log"][idx].append(
                    {
                        "role": "assistant",
                        "content": output,
                    }
                )

            to_env = [
                get_keys_from_message_log(message_log, ["role", "content"])
                for message_log in batch["message_log"]
            ]
            env_return = ray.get(env.step.remote(to_env, batch["extra_env_info"]))

            rewards = env_return.rewards.tolist()
            metadata = env_return.metadata
            answers = env_return.answers

            for idx, (reward, sample_metadata, answer, prompt, response) in enumerate(
                zip(rewards, metadata, answers, prompts, outputs)
            ):
                sample_results.append(
                    {
                        "sample_index": len(sample_results),
                        "dataset_index": batch["idx"][idx],
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                        "answer": answer,
                        "fen": sample_metadata.get("fen") if sample_metadata else None,
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
        generation.shutdown()

    legal_flags = [bool(row["is_legal_move"]) for row in sample_results]
    mate_flags = [bool(row["delivered_checkmate"]) for row in sample_results]
    rewards = [float(row["reward"]) for row in sample_results]
    eval_deltas = [
        float(row["eval_after"]) - float(row["eval_before"])
        for row in sample_results
        if row["eval_before"] is not None and row["eval_after"] is not None
    ]
    legal_eval_deltas = [
        float(row["eval_after"]) - float(row["eval_before"])
        for row in sample_results
        if row["is_legal_move"]
        and row["eval_before"] is not None
        and row["eval_after"] is not None
    ]

    summary: dict[str, Any] = {
        "num_positions": len(sample_results) // num_tests_per_prompt,
        "num_samples": len(sample_results),
        "num_tests_per_prompt": num_tests_per_prompt,
        "avg_reward": _mean(rewards),
        "legal_move_rate": _mean([1.0 if flag else 0.0 for flag in legal_flags]),
        "checkmate_rate": _mean([1.0 if flag else 0.0 for flag in mate_flags]),
        "avg_eval_delta": _mean(eval_deltas),
        "avg_eval_delta_on_legal_moves": _mean(legal_eval_deltas),
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
        summary["any_checkmate_rate"] = _mean(
            [
                1.0 if any(bool(row["delivered_checkmate"]) for row in group) else 0.0
                for group in grouped_results
            ]
        )

    return summary, sample_results


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
    print("Chess eval settings:")
    pprint.pprint(eval_cfg)

    init_ray()
    ensure_chess_env_registered()
    set_seed(eval_cfg["seed"])

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    dataset = _setup_dataset(tokenizer, config)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    env = create_env("chess", config["env"]["chess"])

    generation, cluster, generation_cfg = _setup_generation(config, tokenizer)

    try:
        summary, sample_results = asyncio.run(
            _run_eval_impl(generation, dataloader, env, eval_cfg)
        )
    finally:
        cluster.shutdown()

    summary["model_name"] = generation_cfg["model_name"]
    summary["data_path"] = (
        config["data"].get("validation", {}) or config["data"].get("train", {})
    ).get("data_path")

    print("\n" + "=" * 60)
    print("Chess Evaluation Results")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")

    save_path = eval_cfg.get("save_path")
    if save_path:
        _save_results(save_path, summary, sample_results)


if __name__ == "__main__":
    main()
