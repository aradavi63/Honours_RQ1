from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


ATTACK_SCHEDULES = ("all_rounds", "first_third", "final_third")


def select_malicious_clients(
    client_count: int, malicious_fraction: float, seed: int
) -> tuple[int, ...]:
    if client_count < 1:
        raise ValueError("client_count must be positive")
    if not 0.0 <= malicious_fraction <= 1.0:
        raise ValueError("malicious_fraction must be between zero and one")
    exact_count = client_count * malicious_fraction
    count = round(exact_count)
    if not math.isclose(exact_count, count, abs_tol=1e-9):
        raise ValueError(
            "malicious_fraction must map to a whole number of clients; "
            f"{malicious_fraction} * {client_count} = {exact_count}"
        )
    rng = np.random.default_rng(seed)
    return tuple(sorted(int(value) for value in rng.choice(client_count, count, replace=False)))


def attack_is_active(round_index: int, total_rounds: int, schedule: str) -> bool:
    if schedule not in ATTACK_SCHEDULES:
        raise ValueError(f"unknown attack schedule: {schedule}")
    if total_rounds < 1 or not 0 <= round_index < total_rounds:
        raise ValueError("round index is outside the experiment")
    third = math.ceil(total_rounds / 3)
    if schedule == "all_rounds":
        return True
    if schedule == "first_third":
        return round_index < third
    return round_index >= total_rounds - third


def flip_source_labels(
    targets: torch.Tensor, source_label: int, target_label: int
) -> torch.Tensor:
    if source_label == target_label:
        raise ValueError("source and target labels must differ")
    poisoned = targets.clone()
    poisoned[targets == source_label] = target_label
    return poisoned


def attack_metrics(
    labels: Sequence[int], predictions: Sequence[int], source_label: int, target_label: int
) -> dict[str, float]:
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    if labels_array.shape != predictions_array.shape:
        raise ValueError("labels and predictions must have the same shape")
    source_mask = labels_array == source_label
    if not np.any(source_mask):
        raise ValueError("evaluation data contains no source-label samples")
    unaffected_recalls = []
    for label in np.unique(labels_array):
        if label == source_label:
            continue
        mask = labels_array == label
        unaffected_recalls.append(float(np.mean(predictions_array[mask] == label)))
    return {
        "source_recall": float(np.mean(predictions_array[source_mask] == source_label)),
        "targeted_attack_success_rate": float(
            np.mean(predictions_array[source_mask] == target_label)
        ),
        "unaffected_macro_recall": float(np.mean(unaffected_recalls)),
    }
