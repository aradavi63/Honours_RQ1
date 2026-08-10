from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

TensorDict = Mapping[str, np.ndarray]


def weighted_fedavg(updates: Iterable[TensorDict], sample_counts: Iterable[int]) -> Dict[str, np.ndarray]:
    updates = list(updates)
    counts = np.asarray(list(sample_counts), dtype=np.float64)
    if not updates:
        raise ValueError("at least one client update is required")
    if len(updates) != len(counts):
        raise ValueError("updates and sample_counts must have equal lengths")
    if np.any(counts <= 0):
        raise ValueError("sample counts must be positive")
    keys = tuple(updates[0].keys())
    if any(tuple(update.keys()) != keys for update in updates):
        raise ValueError("all client updates must have identical ordered keys")

    result: Dict[str, np.ndarray] = {}
    total = float(counts.sum())
    for key in keys:
        shape = updates[0][key].shape
        if any(update[key].shape != shape for update in updates):
            raise ValueError(f"shape mismatch for parameter {key}")
        result[key] = sum(
            np.asarray(update[key], dtype=np.float64) * count
            for update, count in zip(updates, counts)
        ) / total
    return result


def add_to_model(model: TensorDict, update: TensorDict) -> Dict[str, np.ndarray]:
    if tuple(model.keys()) != tuple(update.keys()):
        raise ValueError("model and update keys do not match")
    return {key: np.asarray(model[key]) + np.asarray(update[key]) for key in model}

