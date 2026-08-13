from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.aggregation import add_to_model, weighted_fedavg
from rq1_harness.e0 import tenseal_weighted_fedavg
from rq1_harness.fedshe import fedshe_ckks_weighted_fedavg, fedshe_plain_weighted_fedavg
from rq1_harness.membership import membership_metrics, per_sample_scores
from rq1_harness.training import (
    SmallMnistCNN,
    load_or_create_iid_partitions,
    numpy_to_state,
    seed_everything,
    state_to_numpy,
    train_client,
)
from scripts.run_e1 import load_data, write_rows

BACKENDS = ("plaintext", "lee_ckks", "fedshe_plain", "fedshe_ckks")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E4b black-box membership inference")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="mnist")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fedshe-security-level", default="128")
    parser.add_argument("--fedshe-multiplication-depth", default="0")
    parser.add_argument("--fedshe-polynomial-degree", default="16384")
    parser.add_argument("--fedshe-round-decimals", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--partition-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.clients, args.samples_per_client, args.rounds, args.local_epochs, args.batch_size) < 1:
        parser.error("client, sample, round, epoch and batch values must be positive")

    seed_everything(args.seed)
    device = torch.device("cpu")
    train_data, test_data = load_data(args.dataset, args.data_root, args.seed)
    manifest = (args.partition_manifest or ROOT / "results" / "partitions" / (
        f"{args.dataset}_iid_clients-{args.clients}_samples-{args.samples_per_client}_seed-{args.seed}.json"
    )).resolve()
    partitions = load_or_create_iid_partitions(
        manifest, args.dataset, len(train_data), args.clients, args.samples_per_client, args.seed
    )
    model = SmallMnistCNN().to(device)
    crypto_totals: dict[str, float] = {}
    for round_index in range(args.rounds):
        updates = []
        for client_id, indices in enumerate(partitions):
            update, _ = train_client(
                model, train_data, indices, args.local_epochs, args.batch_size,
                args.learning_rate, args.seed * 100000 + round_index * 1000 + client_id,
                device,
            )
            updates.append(update)
        counts = [len(indices) for indices in partitions]
        if args.backend == "plaintext":
            average, crypto = weighted_fedavg(updates, counts), {}
        elif args.backend == "lee_ckks":
            average, crypto = tenseal_weighted_fedavg(updates, counts)
        elif args.backend == "fedshe_plain":
            average, crypto = fedshe_plain_weighted_fedavg(updates, counts), {}
        else:
            average, crypto = fedshe_ckks_weighted_fedavg(
                updates, counts,
                security_level=args.fedshe_security_level,
                multiplication_depth=args.fedshe_multiplication_depth,
                polynomial_degree=args.fedshe_polynomial_degree,
                round_decimals=args.fedshe_round_decimals,
            )
        for key, value in crypto.items():
            if isinstance(value, (int, float)):
                crypto_totals[key] = crypto_totals.get(key, 0.0) + float(value)
        current = state_to_numpy(model.state_dict())
        model.load_state_dict(numpy_to_state(add_to_model(current, average), model.state_dict()))

    member_indices = [index for partition in partitions for index in partition]
    rng = np.random.default_rng(args.seed + 40000)
    nonmember_indices = rng.choice(len(test_data), size=len(member_indices), replace=False).tolist()
    member_loss, member_confidence = per_sample_scores(
        model, train_data, member_indices, args.batch_size, device
    )
    nonmember_loss, nonmember_confidence = per_sample_scores(
        model, test_data, nonmember_indices, args.batch_size, device
    )
    loss_metrics = membership_metrics(member_loss, nonmember_loss)
    confidence_metrics = membership_metrics(member_confidence, nonmember_confidence)
    row = {
        "backend": args.backend,
        "dataset": args.dataset,
        "seed": args.seed,
        "clients": args.clients,
        "samples_per_client": args.samples_per_client,
        "rounds": args.rounds,
        "member_samples": len(member_indices),
        "nonmember_samples": len(nonmember_indices),
        "partition_manifest": str(manifest.relative_to(ROOT)),
        **{f"loss_{key}": value for key, value in loss_metrics.items()},
        **{f"confidence_{key}": value for key, value in confidence_metrics.items()},
        **{f"total_{key}": value for key, value in crypto_totals.items()},
    }
    print(json.dumps(row))
    if args.output:
        write_rows(args.output, [row])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
