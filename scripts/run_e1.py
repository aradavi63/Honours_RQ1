from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import time
from pathlib import Path

import torch
import certifi
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.aggregation import add_to_model, weighted_fedavg
from rq1_harness.e0 import tenseal_weighted_fedavg
from rq1_harness.fedshe import (
    fedshe_ckks_weighted_fedavg,
    fedshe_plain_weighted_fedavg,
)
from rq1_harness.metrics import aggregation_error
from rq1_harness.training import (
    SmallMnistCNN,
    evaluate,
    load_or_create_iid_partitions,
    numpy_to_state,
    seed_everything,
    state_to_numpy,
    train_client,
)


def load_data(name: str, data_root: Path, seed: int):
    if name == "mnist":
        # Python 3.9 fails on a malformed entry in this machine's Windows
        # certificate store. Keep TLS verification enabled using Certifi instead.
        ssl._create_default_https_context = lambda: ssl.create_default_context(
            cafile=certifi.where()
        )
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return (
            datasets.MNIST(data_root, train=True, download=True, transform=transform),
            datasets.MNIST(data_root, train=False, download=True, transform=transform),
        )
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(400, 1, 28, 28, generator=generator)
    train_targets = torch.randint(0, 10, (400,), generator=generator)
    test_inputs = torch.randn(100, 1, 28, 28, generator=generator)
    test_targets = torch.randint(0, 10, (100,), generator=generator)
    return TensorDataset(train_inputs, train_targets), TensorDataset(test_inputs, test_targets)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the shared E1 FL training pilot")
    parser.add_argument(
        "--backend",
        choices=("plaintext", "lee_ckks", "fedshe_plain", "fedshe_ckks"),
        required=True,
    )
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="synthetic")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--samples-per-client", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fedshe-security-level", default="128")
    parser.add_argument("--fedshe-multiplication-depth", default="0")
    parser.add_argument("--fedshe-polynomial-degree", default="16384")
    parser.add_argument("--fedshe-round-decimals", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partition-manifest", type=Path)
    args = parser.parse_args()
    if min(args.clients, args.samples_per_client, args.rounds, args.local_epochs, args.batch_size) < 1:
        parser.error("client, sample, round, epoch and batch values must be positive")

    seed_everything(args.seed)
    device = torch.device("cpu")
    train_data, test_data = load_data(args.dataset, args.data_root, args.seed)
    manifest = args.partition_manifest or (
        ROOT
        / "results"
        / "partitions"
        / f"{args.dataset}_iid_clients-{args.clients}_samples-{args.samples_per_client}_seed-{args.seed}.json"
    )
    manifest = manifest.resolve()
    partitions = load_or_create_iid_partitions(
        manifest,
        args.dataset,
        len(train_data),
        args.clients,
        args.samples_per_client,
        args.seed,
    )
    model = SmallMnistCNN().to(device)
    rows = []
    for round_id in range(args.rounds):
        round_started = time.perf_counter()
        updates = []
        losses = []
        for client_id, indices in enumerate(partitions):
            update, loss = train_client(
                model,
                train_data,
                indices,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                args.seed * 100000 + round_id * 1000 + client_id,
                device,
            )
            updates.append(update)
            losses.append(loss)

        counts = [len(indices) for indices in partitions]
        reference_average = weighted_fedavg(updates, counts)
        crypto = {}
        if args.backend == "plaintext":
            average = reference_average
        elif args.backend == "lee_ckks":
            average, crypto = tenseal_weighted_fedavg(updates, counts)
        elif args.backend == "fedshe_plain":
            average = fedshe_plain_weighted_fedavg(updates, counts)
        else:
            average, crypto = fedshe_ckks_weighted_fedavg(
                updates,
                counts,
                security_level=args.fedshe_security_level,
                multiplication_depth=args.fedshe_multiplication_depth,
                polynomial_degree=args.fedshe_polynomial_degree,
                round_decimals=args.fedshe_round_decimals,
            )
        correctness = aggregation_error(reference_average, average)
        current = state_to_numpy(model.state_dict())
        model.load_state_dict(numpy_to_state(add_to_model(current, average), model.state_dict()))
        test_loss, test_accuracy = evaluate(model, test_data, args.batch_size, device)
        row = {
            "backend": args.backend,
            "dataset": args.dataset,
            "seed": args.seed,
            "round": round_id + 1,
            "clients": args.clients,
            "samples_per_client": args.samples_per_client,
            "partition_manifest": str(manifest.relative_to(ROOT)),
            "mean_client_loss": sum(losses) / len(losses),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "round_seconds": time.perf_counter() - round_started,
            **correctness,
            **crypto,
        }
        rows.append(row)
        print(json.dumps(row))
    if args.output:
        write_rows(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
