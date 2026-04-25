"""Download chess positions from Kaggle and sample a subset as JSONL."""

import csv
import json
import os
import random
import sys

NUM_SAMPLES = 10_000
SEED = 42
OUTPUT = os.path.join(os.path.dirname(__file__), "data", "chess_positions_train.jsonl")


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    path = kagglehub.dataset_download("nikitricky/chess-positions")
    csv_path = os.path.join(path, "positions.csv")

    print(f"Reading {csv_path} ...")
    fens: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row.get("fen", "").strip()
            if fen:
                fens.append(fen)

    print(f"Total positions: {len(fens)}")

    rng = random.Random(SEED)
    sampled = rng.sample(fens, min(NUM_SAMPLES, len(fens)))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for fen in sampled:
            f.write(json.dumps({"fen": fen}) + "\n")

    print(f"Wrote {len(sampled)} positions to {OUTPUT}")


if __name__ == "__main__":
    main()
