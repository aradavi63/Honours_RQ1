from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn


def parameter_gradients(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return gradients of the mean cross-entropy loss without changing the model."""
    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    return tuple(
        gradient.detach().clone()
        for gradient in torch.autograd.grad(loss, tuple(model.parameters()))
    )


def average_gradients(
    gradients: Sequence[Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    """Average equally shaped gradient observations parameter by parameter."""
    if not gradients:
        raise ValueError("at least one gradient observation is required")
    parameter_count = len(gradients[0])
    if parameter_count == 0 or any(len(item) != parameter_count for item in gradients):
        raise ValueError("gradient observations must have equal non-zero lengths")
    result = []
    for parameter_index in range(parameter_count):
        values = [item[parameter_index] for item in gradients]
        if any(value.shape != values[0].shape for value in values):
            raise ValueError("corresponding gradients must have equal shapes")
        result.append(torch.stack(values).mean(dim=0))
    return tuple(result)


def infer_single_label(gradients: Sequence[torch.Tensor]) -> int:
    """Apply iDLG's batch-size-one label rule to the final bias gradient."""
    if not gradients or gradients[-1].ndim != 1:
        raise ValueError("the model must end with a trainable class-bias vector")
    return int(torch.argmin(gradients[-1]).item())


def reconstruct_from_gradients(
    model: nn.Module,
    observed: Sequence[torch.Tensor],
    image_shape: Sequence[int],
    label: int,
    iterations: int,
    learning_rate: float,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    """Reconstruct one image by minimizing squared gradient mismatch (DLG style)."""
    if iterations < 1 or learning_rate <= 0:
        raise ValueError("iterations and learning_rate must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dummy = torch.rand((1, *image_shape), generator=generator, requires_grad=True)
    target = torch.tensor([label], dtype=torch.long)
    optimiser = torch.optim.Adam([dummy], lr=learning_rate)
    parameters = tuple(model.parameters())
    if len(observed) != len(parameters):
        raise ValueError("observed gradients do not match the model parameters")
    history = []
    model.eval()
    for _ in range(iterations):
        optimiser.zero_grad()
        loss = nn.CrossEntropyLoss()(model(dummy), target)
        candidate = torch.autograd.grad(loss, parameters, create_graph=True)
        mismatch = sum(
            torch.sum((actual - expected) ** 2)
            for actual, expected in zip(candidate, observed)
        )
        mismatch.backward()
        optimiser.step()
        with torch.no_grad():
            dummy.clamp_(0.0, 1.0)
        history.append(float(mismatch.detach().item()))
    return dummy.detach(), history


def reconstruction_metrics(
    reference: torch.Tensor, reconstruction: torch.Tensor
) -> dict[str, float]:
    """Return pixel MSE, unit-range PSNR and a global unit-range SSIM score."""
    expected = reference.detach().cpu().to(torch.float64).reshape(-1)
    actual = reconstruction.detach().cpu().to(torch.float64).reshape(-1)
    if expected.shape != actual.shape:
        raise ValueError("reference and reconstruction shapes do not match")
    mse = float(torch.mean((expected - actual) ** 2).item())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    mean_x, mean_y = expected.mean(), actual.mean()
    var_x = torch.mean((expected - mean_x) ** 2)
    var_y = torch.mean((actual - mean_y) ** 2)
    covariance = torch.mean((expected - mean_x) * (actual - mean_y))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return {"mse": mse, "psnr": psnr, "ssim": float(ssim.item())}


def ciphertext_only_result() -> dict[str, object]:
    """Describe why plaintext gradient matching cannot consume a ciphertext."""
    return {
        "applicability": "not_applicable",
        "reason": "The attacker cannot compute gradient mismatch from CKKS ciphertext coefficients without a decryption key.",
        "mse": np.nan,
        "psnr": np.nan,
        "ssim": np.nan,
        "label_recovery": np.nan,
    }
