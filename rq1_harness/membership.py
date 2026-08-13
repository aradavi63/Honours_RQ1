from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def update_as_gradient_vector(
    model: nn.Module, update: Mapping[str, np.ndarray]
) -> torch.Tensor:
    """Flatten global-minus-local parameters, matching FedMIA's gradient direction."""
    values = []
    for name, parameter in model.named_parameters():
        if name not in update or tuple(update[name].shape) != tuple(parameter.shape):
            raise ValueError(f"update does not match trainable parameter {name}")
        values.append(-torch.as_tensor(update[name], dtype=parameter.dtype).reshape(-1))
    return torch.cat(values)


def per_sample_gradient_cosines(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    observed_vectors: Sequence[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Measure each sample gradient's cosine similarity with each observed update."""
    if not indices or not observed_vectors:
        raise ValueError("samples and observed updates must be non-empty")
    model = model.to(device)
    model.eval()
    parameters = tuple(model.parameters())
    vectors = [value.to(device).reshape(-1) for value in observed_vectors]
    expected_size = sum(parameter.numel() for parameter in parameters)
    if any(vector.numel() != expected_size for vector in vectors):
        raise ValueError("observed update size does not match the model")
    rows = []
    for index in indices:
        inputs, target = dataset[int(index)]
        inputs = inputs.unsqueeze(0).to(device)
        targets = torch.as_tensor([int(target)], dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        gradients = torch.autograd.grad(loss, parameters)
        sample = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
        sample_norm = torch.linalg.vector_norm(sample)
        similarities = []
        for vector in vectors:
            denominator = sample_norm * torch.linalg.vector_norm(vector)
            similarities.append(
                float(torch.dot(sample, vector).item() / denominator.item())
                if denominator.item()
                else 0.0
            )
        rows.append(similarities)
    return np.asarray(rows, dtype=np.float64)


def spatial_temporal_scores(
    round_cosines: Sequence[np.ndarray], observation: str
) -> np.ndarray:
    """Combine round/client cosine measurements under a declared observation model."""
    if not round_cosines:
        raise ValueError("at least one round of cosine measurements is required")
    arrays = [np.asarray(values, dtype=np.float64) for values in round_cosines]
    if any(values.ndim != 2 or values.shape != arrays[0].shape for values in arrays):
        raise ValueError("round cosine matrices must have equal two-dimensional shapes")
    if observation in ("individual_plaintext", "colluding_clients"):
        if arrays[0].shape[1] < 2:
            raise ValueError("spatial scoring requires a target and a non-target client")
        per_round = [values[:, 0] - values[:, 1:].mean(axis=1) for values in arrays]
    elif observation == "route_aggregate":
        if arrays[0].shape[1] != 1:
            raise ValueError("route aggregate scoring expects exactly one update")
        per_round = [values[:, 0] for values in arrays]
    else:
        raise ValueError(f"unsupported plaintext observation: {observation}")
    return np.stack(per_round, axis=1).mean(axis=1)
