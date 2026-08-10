from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class SmallMnistCNN(nn.Module):
    """Small common model whose updates are practical to encrypt during pilots."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 7 * 7, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(start_dim=1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def iid_partitions(
    dataset_size: int, client_count: int, samples_per_client: int, seed: int
) -> list[list[int]]:
    required = client_count * samples_per_client
    if required > dataset_size:
        raise ValueError(
            f"requested {required} client samples from a dataset of {dataset_size}"
        )
    rng = np.random.default_rng(seed)
    selected = rng.permutation(dataset_size)[:required]
    return [
        selected[start : start + samples_per_client].astype(int).tolist()
        for start in range(0, required, samples_per_client)
    ]


def load_or_create_iid_partitions(
    path: Path,
    dataset_name: str,
    dataset_size: int,
    client_count: int,
    samples_per_client: int,
    seed: int,
) -> list[list[int]]:
    expected = {
        "schema_version": 1,
        "dataset": dataset_name,
        "dataset_size": dataset_size,
        "client_count": client_count,
        "samples_per_client": samples_per_client,
        "seed": seed,
    }
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(
                    f"partition manifest {path} has {key}={document.get(key)!r}; "
                    f"expected {value!r}"
                )
        partitions = document.get("partitions")
        if not isinstance(partitions, list) or len(partitions) != client_count:
            raise ValueError(f"partition manifest {path} has invalid partitions")
        return [[int(index) for index in client] for client in partitions]

    partitions = iid_partitions(
        dataset_size, client_count, samples_per_client, seed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump({**expected, "partitions": partitions}, stream, indent=2)
    return partitions


def state_to_numpy(state: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() for key, value in state.items()}


def numpy_to_state(
    values: Dict[str, np.ndarray], template: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(values[key], dtype=template[key].dtype).clone()
        for key in template
    }


def model_delta(
    before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    return {
        key: (after[key].detach().cpu() - before[key].detach().cpu()).numpy()
        for key in before
    }


def train_client(
    global_model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[Dict[str, np.ndarray], float]:
    local_model = copy.deepcopy(global_model).to(device)
    before = copy.deepcopy(local_model.state_dict())
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimiser = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    local_model.train()
    total_loss = 0.0
    total_samples = 0
    for _ in range(local_epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            loss = loss_function(local_model(inputs), targets)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item()) * inputs.size(0)
            total_samples += inputs.size(0)
    return model_delta(before, local_model.state_dict()), total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device
) -> tuple[float, float]:
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    samples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += float(loss_function(logits, targets).item())
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        samples += inputs.size(0)
    return loss_sum / samples, correct / samples
