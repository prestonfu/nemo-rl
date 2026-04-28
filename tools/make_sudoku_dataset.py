"""
Generate a Sudoku dataset from Ritvik19/Sudoku-Dataset.

Each output record:
  {
    "puzzle": "530070000\n...",   # 9 rows, 0 = empty cell
    "solution": "534678912\n...", # 9 rows, digits 1-9
    "missing": 32,
    "difficulty": 0,              # 0 or 1
    "solving_time": 0.12,
  }

Usage:
  python tools/make_sudoku_dataset.py --train-output examples/data/sudoku_train.jsonl --val-output examples/data/sudoku_val.jsonl --num-train 10000 --num-val 1000 --seed 42
"""

import argparse
import json
import random
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm


def format_grid(flat: str) -> str:
    return "\n".join(flat[r * 9 : r * 9 + 9] for r in range(9))


def sample_records(ds, indices):
    records = []
    for idx in tqdm(indices, unit="puzzle"):
        row = ds[idx]
        records.append({
            "puzzle": format_grid(row["puzzle"]),
            "solution": format_grid(row["solution"]),
            "missing": row["missing"],
            "difficulty": row["difficulty"],
            "solving_time": row["solving_time"],
        })
    return records


def print_stats(label, records):
    diff_dist = Counter(r["difficulty"] for r in records)
    missing_vals = [r["missing"] for r in records]
    print(f"  {label}: {len(records)} records, difficulty={dict(sorted(diff_dist.items()))}, "
          f"missing min/mean/max={min(missing_vals)}/{sum(missing_vals)/len(missing_vals):.1f}/{max(missing_vals)}")


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", default="examples/data/sudoku_train.jsonl")
    parser.add_argument("--val-output", default="examples/data/sudoku_val.jsonl")
    parser.add_argument("--num-train", type=int, default=10_000)
    parser.add_argument("--num-val", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading dataset...")
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train")
    print(f"  {len(ds)} puzzles available")

    rng = random.Random(args.seed)
    all_indices = rng.sample(range(len(ds)), args.num_train + args.num_val)
    train_indices = all_indices[: args.num_train]
    val_indices = all_indices[args.num_train :]

    print("Sampling train...")
    train_records = sample_records(ds, train_indices)
    print("Sampling val...")
    val_records = sample_records(ds, val_indices)

    print_stats("train", train_records)
    print_stats("val", val_records)

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.val_output, val_records)


if __name__ == "__main__":
    main()
