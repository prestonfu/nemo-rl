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

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class ThinkBudgetLogitsProcessor(LogitsProcessor):
    """Forces </think> token after thinking_token_budget generated tokens.

    Per-request config via SamplingParams.extra_args:
        thinking_token_budget (int): max tokens before </think> is forced
        end_think_token_id (int): token id for </think> (default 13 for Nemotron)
    """

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra = params.extra_args or {}
        budget = extra.get("thinking_token_budget")
        token_id = extra.get("end_think_token_id")
        if budget is None and token_id is None:
            return
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("thinking_token_budget must be a non-negative int")
        if not isinstance(token_id, int):
            raise ValueError("end_think_token_id must be an int")

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        # Maps batch index -> (budget, end_think_token_id, output_token_ids_ref)
        self._state: dict[int, tuple[int, int, list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        if batch_update is None:
            return

        for idx in batch_update.removed:
            self._state.pop(idx, None)

        for idx, params, _prompt_ids, output_ids in batch_update.added:
            extra = (params.extra_args or {}) if params else {}
            budget = extra.get("thinking_token_budget")
            end_id = extra.get("end_think_token_id", 13)
            if budget is None:
                self._state.pop(idx, None)
            else:
                self._state[idx] = (budget, end_id, output_ids)

        for src, dst, direction in batch_update.moved:
            src_state = self._state.pop(src, None)
            dst_state = self._state.pop(dst, None)
            if src_state is not None:
                self._state[dst] = src_state
            if direction == MoveDirectionality.SWAP and dst_state is not None:
                self._state[src] = dst_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._state:
            return logits

        rows, cols = [], []
        for idx, (budget, end_id, output_ids) in self._state.items():
            if end_id in output_ids:
                continue
            if len(output_ids) < budget:
                continue
            rows.append(idx)
            cols.append(end_id)

        if not rows:
            return logits

        r = torch.tensor(rows, dtype=torch.long, device=logits.device)
        c = torch.tensor(cols, dtype=torch.long, device=logits.device)
        saved = logits[r, c].clone()
        logits[r] = float("-inf")
        logits[r, c] = saved
        return logits
