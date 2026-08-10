from __future__ import annotations

from typing import Mapping

import numpy as np


def aggregation_error(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, float]:
    if tuple(reference.keys()) != tuple(candidate.keys()):
        raise ValueError("reference and candidate keys do not match")
    ref = np.concatenate([np.asarray(reference[k], dtype=np.float64).ravel() for k in reference])
    got = np.concatenate([np.asarray(candidate[k], dtype=np.float64).ravel() for k in candidate])
    difference = got - ref
    ref_norm = np.linalg.norm(ref)
    denominator = np.linalg.norm(ref) * np.linalg.norm(got)
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "relative_l2_error": float(np.linalg.norm(difference) / ref_norm) if ref_norm else float(np.linalg.norm(difference)),
        "cosine_similarity": float(np.dot(ref, got) / denominator) if denominator else float(ref_norm == np.linalg.norm(got)),
    }


def targeted_attack_success_rate(labels: np.ndarray, predictions: np.ndarray, source: int, target: int) -> float:
    mask = np.asarray(labels) == source
    if not np.any(mask):
        raise ValueError("no source-class samples were supplied")
    return float(np.mean(np.asarray(predictions)[mask] == target))

