"""
Sudoku eval against a vLLM-served model.

Usage:
  python tools/eval_sudoku.py --data examples/data/sudoku_val.jsonl --n 200 --workers 32 --base-url http://localhost:8771/v1
"""

import argparse
import json
import re
import sys

sys.path.insert(0, "tools")
from eval_utils import make_client, print_stats, run_parallel

SYSTEM_PROMPT = "You are a helpful assistant."

USER_TEMPLATE = """\
Solve this Sudoku. 0 denotes an empty cell.

{puzzle}

Return the completed 9 rows.\
"""


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_grid(text: str) -> str | None:
    text = strip_thinking(text)
    rows = re.findall(r"[1-9]{9}", text)
    if len(rows) >= 9:
        return "".join(rows[:9])
    digits = re.sub(r"[^1-9]", "", text)
    return digits if len(digits) == 81 else None


def cell_accuracy(predicted: str, solution: str) -> float:
    return sum(p == s for p, s in zip(predicted, solution)) / 81.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/data/sudoku_val.jsonl")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--base-url", default="http://localhost:8771/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=None, help="Save full results to this JSONL file")
    args = parser.parse_args()

    client = make_client(args.base_url)

    records = []
    with open(args.data) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[: args.n]
    print(f"Evaluating {len(records)} puzzles from {args.data} with {args.workers} workers")

    def eval_one(rec):
        solution = rec["solution"].replace("\n", "")
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(puzzle=rec["puzzle"])},
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        raw = response.choices[0].message.content or ""
        predicted = parse_grid(raw)
        if predicted is None:
            return {**rec, "predicted": None, "correct": False, "cell_acc": 0.0, "parse_error": True, "raw": raw}
        return {
            **rec,
            "predicted": "\n".join(predicted[r * 9 : r * 9 + 9] for r in range(9)),
            "correct": predicted == solution,
            "cell_acc": cell_accuracy(predicted, solution),
            "parse_error": False,
            "raw": raw,
        }

    results = [r for r in run_parallel(eval_one, records, args.workers, desc="puzzle") if r]

    n = len(results)
    print_stats(f"Sudoku Eval — {args.model} (n={n})", {
        "correct": f"{sum(r['correct'] for r in results)}/{n} ({100*sum(r['correct'] for r in results)/n:.1f}%)",
        "avg_cell_acc": sum(r["cell_acc"] for r in results) / n,
        "parse_errors": f"{sum(r['parse_error'] for r in results)}/{n}",
    })

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
