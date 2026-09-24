"""Percentile bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean with a percentile bootstrap (1 - alpha) CI over items."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(arr.size),
    }


def bootstrap_statistic(
    n_items: int,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap an arbitrary statistic of a resampled index vector (e.g. micro-F1 over docs)."""
    rng = np.random.default_rng(seed)
    full = statistic(np.arange(n_items))
    samples = [statistic(rng.integers(0, n_items, size=n_items)) for _ in range(n_boot)]
    return {
        "value": float(full),
        "lo": float(np.quantile(samples, alpha / 2)),
        "hi": float(np.quantile(samples, 1 - alpha / 2)),
        "n": int(n_items),
    }
