from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from pathlib import Path

import certifi
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.inversion import (
    average_gradients,
    ciphertext_only_result,
    infer_single_label,
    parameter_gradients,
    reconstruct_from_gradients,
    reconstruction_metrics,
)
from rq1_harness.training import SmallMnistCNN, seed_everything
from scripts.run_e1 import write_rows


def load_attack_samples(name: str, count: int, data_root: Path, seed: int):
    if name == "mnist":
        ssl._create_default_https_context = lambda: ssl.create_default_context(
            cafile=certifi.where()
        )
        dataset = datasets.MNIST(
            data_root, train=True, download=True, transform=transforms.ToTensor()
        )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        examples = [dataset[index] for index in indices]
        return torch.stack([item[0] for item in examples]), torch.tensor(
            [int(item[1]) for item in examples]
        )
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(count, 1, 28, 28, generator=generator),
        torch.randint(0, 10, (count,), generator=generator),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E3 gradient-inversion baseline")
    parser.add_argument(
        "--observation",
        choices=("individual_plaintext", "aggregate_2", "aggregate_5", "aggregate_10", "ciphertext_only"),
        required=True,
    )
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="mnist")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-reconstruction", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.learning_rate <= 0:
        parser.error("iterations and learning rate must be positive")

    seed_everything(args.seed)
    aggregate_size = 1 if args.observation == "individual_plaintext" else (
        int(args.observation.split("_")[-1]) if args.observation.startswith("aggregate_") else 1
    )
    inputs, labels = load_attack_samples(
        args.dataset, aggregate_size, args.data_root, args.seed
    )
    model = SmallMnistCNN()
    started = time.perf_counter()

    if args.observation == "ciphertext_only":
        row = {
            "observation": args.observation,
            "dataset": args.dataset,
            "seed": args.seed,
            "aggregate_size": None,
            "true_target_label": None,
            "inferred_label": None,
            "iterations": 0,
            "final_gradient_mismatch": None,
            "attack_seconds": 0.0,
            **ciphertext_only_result(),
        }
    else:
        individual = [
            parameter_gradients(model, inputs[index : index + 1], labels[index : index + 1])
            for index in range(aggregate_size)
        ]
        observed = individual[0] if aggregate_size == 1 else average_gradients(individual)
        inferred_label = infer_single_label(observed)
        reconstruction, history = reconstruct_from_gradients(
            model,
            observed,
            inputs[0].shape,
            inferred_label,
            args.iterations,
            args.learning_rate,
            args.seed + 10000,
        )
        row = {
            "observation": args.observation,
            "dataset": args.dataset,
            "seed": args.seed,
            "aggregate_size": aggregate_size,
            "true_target_label": int(labels[0]),
            "inferred_label": inferred_label,
            "iterations": args.iterations,
            "final_gradient_mismatch": history[-1],
            "attack_seconds": time.perf_counter() - started,
            "applicability": "applicable",
            "reason": (
                "Direct single-example gradient observation"
                if aggregate_size == 1
                else "Single-image attacker applied to an averaged multi-example gradient; metrics compare with the first hidden example"
            ),
            **reconstruction_metrics(inputs[0:1], reconstruction),
            "label_recovery": float(inferred_label == int(labels[0])),
        }
        if args.save_reconstruction:
            args.save_reconstruction.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"reference": inputs[0:1], "reconstruction": reconstruction, "row": row},
                args.save_reconstruction,
            )

    print(json.dumps(row, allow_nan=True))
    if args.output:
        write_rows(args.output, [row])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
