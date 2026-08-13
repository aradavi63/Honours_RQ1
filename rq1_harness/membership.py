from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


def membership_metrics(
    member_scores: Sequence[float], nonmember_scores: Sequence[float]
) -> dict[str, float]:
    """Evaluate scores where a larger value means more likely to be a member."""
    members = np.asarray(member_scores, dtype=np.float64)
    nonmembers = np.asarray(nonmember_scores, dtype=np.float64)
    if members.ndim != 1 or nonmembers.ndim != 1 or not len(members) or not len(nonmembers):
        raise ValueError("member and nonmember scores must be non-empty vectors")
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    scores = np.concatenate([members, nonmembers])
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    true_positives = np.cumsum(sorted_labels)[group_ends - 1]
    false_positives = group_ends - true_positives
    tpr = np.r_[0.0, true_positives / len(members)]
    fpr = np.r_[0.0, false_positives / len(nonmembers)]

    def tpr_at(maximum_fpr: float) -> float:
        eligible = tpr[fpr <= maximum_fpr]
        return float(np.max(eligible)) if len(eligible) else 0.0

    return {
        "roc_auc": float(np.trapz(tpr, fpr)),
        "tpr_at_fpr_01": tpr_at(0.01),
        "tpr_at_fpr_001": tpr_at(0.001),
        "membership_advantage": float(np.max(tpr - fpr)),
    }


@torch.no_grad()
def per_sample_scores(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return negative loss and true-class confidence for each requested sample."""
    if not indices:
        raise ValueError("at least one sample index is required")
    model = model.to(device)
    model.eval()
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False)
    loss_scores, confidence_scores = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        losses = nn.functional.cross_entropy(logits, targets, reduction="none")
        confidence = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        loss_scores.extend((-losses).cpu().numpy().tolist())
        confidence_scores.extend(confidence.cpu().numpy().tolist())
    return np.asarray(loss_scores), np.asarray(confidence_scores)
