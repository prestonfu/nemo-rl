"""
WikiRace eval against a vLLM-served model (multi-turn).

Usage:
  python tools/eval_wikirace.py --data examples/data/wikirace_val.jsonl --n 100 --workers 16 --base-url http://localhost:8771/v1
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict

from datasets import load_dataset

sys.path.insert(0, "tools")
from eval_utils import make_client, print_stats, run_parallel

SYSTEM_PROMPT = "You are a helpful assistant helping play the Wikipedia link game."

USER_TEMPLATE = """\
You are playing a game where you start at Wikipedia page "{start}" and want to reach page "{goal}" by clicking links.
So far, you have visited the following pages in order:
{history}
You see the following possible links from the current page:
{link_list}
Which link should you click to get closer to the target? Reply with just the number of your choice (0 to {max_idx}).\
"""


def build_graph(dataset_id: str) -> dict[str, list[str]]:
    print(f"Loading graph from {dataset_id}...")
    ds = load_dataset(dataset_id, split="train")
    graph: dict[str, list[str]] = {}
    all_titles: set[str] = set()
    for row in ds:
        graph[row["article"]] = row["links"]
        all_titles.add(row["article"])
    for title in graph:
        graph[title] = [l for l in graph[title] if l in all_titles]
    print(f"  {len(graph)} nodes loaded.")
    return graph


def sample_links(graph, page, goal, num_shown, rng):
    neighbors = list(graph.get(page, []))
    if goal in neighbors:
        others = [n for n in neighbors if n != goal]
        rng.shuffle(others)
        shown = [goal] + others[: num_shown - 1]
        rng.shuffle(shown)
    else:
        rng.shuffle(neighbors)
        shown = neighbors[:num_shown]
    return shown


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_choice(text: str) -> int | None:
    text = strip_thinking(text)
    matches = re.findall(r"\b(\d+)\b", text.strip())
    return int(matches[-1]) if matches else None


def run_episode(client, model, graph, rec, num_links, max_steps, max_tokens, temperature, seed):
    start, goal, shortest = rec["start"], rec["goal"], rec["path_length"]
    history = [start]
    current = start
    messages = []
    rng = random.Random(seed)

    for step in range(max_steps):
        links = sample_links(graph, current, goal, num_links, rng)
        if not links:
            return {"won": False, "steps": step + 1, "shortest": shortest, "reason": "no_links"}

        link_list = "\n".join(f"{i}. {t}" for i, t in enumerate(links))
        user_content = USER_TEMPLATE.format(
            start=start, goal=goal,
            history=" → ".join(history),
            link_list=link_list,
            max_idx=len(links) - 1,
        )

        if step == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        else:
            messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        raw = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": raw})

        choice = extract_choice(raw)
        if choice is None or choice < 0 or choice >= len(links):
            return {"won": False, "steps": step + 1, "shortest": shortest, "reason": "invalid_choice"}

        current = links[choice]
        history.append(current)
        if current == goal:
            return {"won": True, "steps": step + 1, "shortest": shortest}

    return {"won": False, "steps": max_steps, "shortest": shortest, "reason": "max_steps"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/data/wikirace_val.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base-url", default="http://localhost:8771/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-links", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="HuggingFaceTB/simplewiki-pruned-350k")
    parser.add_argument("--output", default=None, help="Save full results to this JSONL file")
    args = parser.parse_args()

    graph = build_graph(args.dataset)
    client = make_client(args.base_url)

    records = []
    with open(args.data) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[: args.n]
    print(f"Evaluating {len(records)} episodes with {args.workers} workers")

    def eval_one(item):
        i, rec = item
        result = run_episode(
            client, args.model, graph, rec,
            args.num_links, args.max_steps, args.max_tokens, args.temperature,
            seed=args.seed + i,
        )
        return {**rec, **result}

    results = [r for r in run_parallel(eval_one, list(enumerate(records)), args.workers, desc="ep") if r and "won" in r]

    wins = [r for r in results if r["won"]]
    n = len(results)
    spl = sum(r["shortest"] / max(r["steps"], r["shortest"]) for r in wins) / n if wins else 0.0
    fail_reasons = defaultdict(int)
    for r in results:
        if not r["won"]:
            fail_reasons[r.get("reason", "unknown")] += 1

    print_stats(f"WikiRace Eval — {args.model} (n={n})", {
        "win_rate": f"{len(wins)}/{n} ({100*len(wins)/n:.1f}%)",
        "avg_steps_all": sum(r["steps"] for r in results) / n,
        "avg_steps_wins": sum(r["steps"] for r in wins) / len(wins) if wins else 0.0,
        "spl": spl,
        "fail_reasons": dict(fail_reasons),
    })

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
