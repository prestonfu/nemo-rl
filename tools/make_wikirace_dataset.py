"""
Generate a WikiRace dataset from HuggingFaceTB/simplewiki-pruned-350k.

Each output record:
  {
    "start": "Article A",
    "goal":  "Article B",
    "shortest_path": ["Article A", ..., "Article B"],
    "path_length": N
  }

Usage:
  python tools/make_wikirace_dataset.py --train-output examples/data/wikirace_train.jsonl --val-output examples/data/wikirace_val.jsonl --num-train 10000 --num-val 1000 --workers 32
"""

import argparse
import json
import multiprocessing as mp
import random
from collections import Counter, deque

from datasets import load_dataset
from tqdm import tqdm


def build_graph(dataset) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    all_titles: set[str] = set()
    for row in dataset:
        title = row["article"]
        graph[title] = row["links"]
        all_titles.add(title)
    for title in graph:
        graph[title] = [l for l in graph[title] if l in all_titles]
    return graph


def bfs_shortest_path(graph: dict[str, list[str]], start: str, goal: str) -> list[str] | None:
    if start == goal:
        return [start]
    visited = {start}
    queue: deque[list[str]] = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbor in graph.get(node, []):
            if neighbor == goal:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


# Module-level globals so forked workers share memory without pickling the graph.
_graph: dict[str, list[str]] = {}
_titles: list[str] = []


def _init_worker(graph, titles):
    global _graph, _titles
    _graph = graph
    _titles = titles


def _worker(args):
    """Try `batch_size` random pairs and return any that pass the filters."""
    seed, batch_size, min_length, max_length = args
    rng = random.Random(seed)
    results = []
    for _ in range(batch_size):
        start = rng.choice(_titles)
        goal = rng.choice(_titles)
        if start == goal:
            continue
        path = bfs_shortest_path(_graph, start, goal)
        if path is None:
            continue
        length = len(path) - 1
        if length < min_length or length > max_length:
            continue
        results.append({
            "start": start,
            "goal": goal,
            "shortest_path": path,
            "path_length": length,
        })
    return results


def collect_pairs(graph, titles, num_pairs, min_length, max_length, workers, batch_size, max_attempts_multiplier, seed):
    total_attempts = num_pairs * max_attempts_multiplier
    num_tasks = (total_attempts + batch_size - 1) // batch_size
    rng = random.Random(seed)

    def task_gen():
        for _ in range(num_tasks):
            yield (rng.randint(0, 2**31), batch_size, min_length, max_length)

    results = []
    with mp.Pool(workers, initializer=_init_worker, initargs=(graph, titles)) as pool:
        with tqdm(total=num_pairs, unit="pair") as pbar:
            for batch in pool.imap_unordered(_worker, task_gen(), chunksize=1):
                for record in batch:
                    if len(results) < num_pairs:
                        results.append(record)
                        pbar.update(1)
                if len(results) >= num_pairs:
                    pool.terminate()
                    break
    return results


def print_stats(label, results):
    dist = Counter(r["path_length"] for r in results)
    print(f"  {label}: {len(results)} pairs, length dist={dict(sorted(dist.items()))}")


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", default="examples/data/wikirace_train.jsonl")
    parser.add_argument("--val-output", default="examples/data/wikirace_val.jsonl")
    parser.add_argument("--num-train", type=int, default=10_000)
    parser.add_argument("--num-val", type=int, default=1_000)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-attempts-multiplier", type=int, default=20)
    args = parser.parse_args()

    print("Loading dataset...")
    ds = load_dataset("HuggingFaceTB/simplewiki-pruned-350k", split="train")
    print(f"  {len(ds)} articles")

    print("Building graph...")
    graph = build_graph(ds)
    titles = [t for t in graph if graph[t]]
    print(f"  {len(graph)} nodes, {len(titles)} with outgoing links")

    print(f"Sampling train ({args.num_train} pairs)...")
    train = collect_pairs(graph, titles, args.num_train, args.min_length, args.max_length,
                          args.workers, args.batch_size, args.max_attempts_multiplier, args.seed)

    print(f"Sampling val ({args.num_val} pairs)...")
    val = collect_pairs(graph, titles, args.num_val, args.min_length, args.max_length,
                        args.workers, args.batch_size, args.max_attempts_multiplier, args.seed + 1)

    print_stats("train", train)
    print_stats("val", val)

    write_jsonl(args.train_output, train)
    write_jsonl(args.val_output, val)


if __name__ == "__main__":
    main()
