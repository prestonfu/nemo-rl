"""Shared utilities for eval scripts: parallel inference with Ctrl-C handling."""
from __future__ import annotations

import signal
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from openai import OpenAI
from tqdm import tqdm

T = TypeVar("T")


class ShutdownPool:
    """ThreadPoolExecutor with SIGINT handling that cancels in-flight futures."""

    def __init__(self, max_workers: int) -> None:
        self._shutdown = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._prev_handler = None

    def __enter__(self) -> "ShutdownPool":
        self._prev_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        return self

    def __exit__(self, *_) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        signal.signal(signal.SIGINT, self._prev_handler or signal.SIG_DFL)

    @property
    def interrupted(self) -> bool:
        return self._shutdown.is_set()

    def submit(self, fn: Callable, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)

    def as_completed(self, futures) -> Iterable:
        return as_completed(futures)

    def _handle_sigint(self, sig, frame) -> None:
        print("\nInterrupted — printing partial results...")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._shutdown.set()


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="dummy")


def run_parallel(
    fn: Callable,
    items: list,
    workers: int,
    desc: str = "eval",
) -> list:
    """Run fn(item) in parallel over items, returning results in submission order.

    Results for items where fn raised are None. Stops accepting new results on
    Ctrl-C but prints whatever completed.
    """
    results = [None] * len(items)
    with ShutdownPool(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        with tqdm(total=len(items), unit=desc) as pbar:
            for future in pool.as_completed(futures):
                if pool.interrupted:
                    break
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"error": str(e)}
                pbar.update(1)
    return results


def print_stats(label: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {label} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
