from __future__ import annotations

from typing import Mapping

import numpy as np


def aggregation_error(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, float]:
    if tuple(reference.keys()) != tuple(candidate.keys()):
        raise ValueError("reference and candidate keys do not match")
    ref = np.concatenate([np.asarray(reference[k], dtype=np.float64).ravel() for k in reference])
    got = np.concatenate([np.asarray(candidate[k], dtype=np.float64).ravel() for k in candidate])
    difference = got - ref
    # Explicit reductions avoid platform-specific BLAS/OpenMP collisions when
    # the legacy FedSHE PyTorch stack and NumPy are loaded in one Windows process.
    ref_norm = float(np.sqrt(np.sum(ref * ref, dtype=np.float64)))
    got_norm = float(np.sqrt(np.sum(got * got, dtype=np.float64)))
    difference_norm = float(np.sqrt(np.sum(difference * difference, dtype=np.float64)))
    denominator = ref_norm * got_norm
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "relative_l2_error": difference_norm / ref_norm if ref_norm else difference_norm,
        "cosine_similarity": float(np.sum(ref * got, dtype=np.float64) / denominator) if denominator else float(ref_norm == got_norm),
    }


def targeted_attack_success_rate(labels: np.ndarray, predictions: np.ndarray, source: int, target: int) -> float:
    mask = np.asarray(labels) == source
    if not np.any(mask):
        raise ValueError("no source-class samples were supplied")
    return float(np.mean(np.asarray(predictions)[mask] == target))
