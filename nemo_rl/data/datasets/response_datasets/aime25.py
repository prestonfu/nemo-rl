from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

class AIME2025Dataset(RawDataset):
    """Simple wrapper around the AIME2025 dataset with train split.

    Args:
        repeat: Number of times to repeat the dataset, default is 16
    """

    def __init__(self, repeat: int = 16, **kwargs) -> None:
        self.task_name = "AIME2025"

        # load from huggingface
        self.dataset = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        # repeat the dataset
        self.dataset = self.dataset.repeat(repeat)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
        }
