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

from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class OpenMementosDataset(RawDataset):
    """Simple wrapper around the OpenMementos dataset.

    Args:
        output_key: Key for the output text. `response` is the native OpenMementos
            field; `generated_solution` is also accepted for config compatibility.
        subset: Dataset subset/config name. Supports `default` and `full`.
        split: Split name for the dataset, default is `train`.
        split_validation_size: Size of the validation data, default is 0.05.
        seed: Seed for train/validation split when split_validation_size > 0.
    """

    _OUTPUT_KEY_ALIASES = {
        "response": "response",
        "generated_solution": "response",
    }

    def __init__(
        self,
        output_key: str = "response",
        subset: str | None = None,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        if subset is None:
            subset = "default"
        if subset not in ["default", "full"]:
            raise ValueError(
                f"Invalid subset: {subset}. Please use 'default' or 'full'."
            )
        if output_key not in self._OUTPUT_KEY_ALIASES:
            raise ValueError(
                f"Invalid output_key: {output_key}. Please use 'response' or 'generated_solution'."
            )

        self.input_key = "problem"
        self.output_key = self._OUTPUT_KEY_ALIASES[output_key]
        self.task_name = "OpenMementos"

        dataset_args = ["microsoft/OpenMementos"]
        if subset != "default":
            dataset_args.append(subset)
        self.dataset = load_dataset(*dataset_args, split=split)

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data[self.input_key]},
                {"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
        }
