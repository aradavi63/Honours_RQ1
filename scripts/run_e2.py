from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.aggregation import add_to_model, weighted_fedavg
from rq1_harness.e0 import tenseal_weighted_fedavg
from rq1_harness.fedshe import fedshe_ckks_weighted_fedavg, fedshe_plain_weighted_fedavg
from rq1_harness.metrics import aggregation_error
from rq1_harness.poisoning import (
    ATTACK_SCHEDULES,
    attack_is_active,
    attack_metrics,
    select_malicious_clients,
)
from rq1_harness.training import (
    SmallMnistCNN,
    dataset_labels,
    evaluate,
    load_or_create_dirichlet_partitions,
    load_or_create_iid_partitions,
    numpy_to_state,
    predict,
    seed_everything,
    state_to_numpy,
    train_client,
)
from scripts.run_e1 import load_data, write_rows


BACKENDS = ("plaintext", "lee_ckks", "fedshe_plain", "fedshe_ckks")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E2 targeted label-flipping")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="mnist")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-label", type=int, default=1)
    parser.add_argument("--target-label", type=int, default=7)
    parser.add_argument("--malicious-fraction", type=float, default=0.1)
    parser.add_argument("--attack-schedule", choices=ATTACK_SCHEDULES, default="all_rounds")
    parser.add_argument("--fedshe-security-level", default="128")
    parser.add_argument("--fedshe-multiplication-depth", default="0")
    parser.add_argument("--fedshe-polynomial-degree", default="16384")
    parser.add_argument("--fedshe-round-decimals", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--partition-manifest", type=Path)
    parser.add_argument("--partitioning", choices=("iid", "dirichlet"), default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.clients, args.samples_per_client, args.rounds, args.local_epochs, args.batch_size) < 1:
        parser.error("client, sample, round, epoch and batch values must be positive")
    if args.source_label == args.target_label:
        parser.error("source and target labels must differ")
    if args.dirichlet_alpha <= 0:
        parser.error("Dirichlet alpha must be positive")

    seed_everything(args.seed)
    device = torch.device("cpu")
    train_data, test_data = load_data(args.dataset, args.data_root, args.seed)
    partition_label = "iid" if args.partitioning == "iid" else (
        f"dirichlet-alpha-{args.dirichlet_alpha:g}"
    )
    manifest = (args.partition_manifest or ROOT / "results" / "partitions" / (
        f"{args.dataset}_{partition_label}_clients-{args.clients}_samples-{args.samples_per_client}_seed-{args.seed}.json"
    )).resolve()
    if args.partitioning == "iid":
        partitions = load_or_create_iid_partitions(
            manifest, args.dataset, len(train_data), args.clients,
            args.samples_per_client, args.seed,
        )
    else:
        partitions = load_or_create_dirichlet_partitions(
            manifest, args.dataset, dataset_labels(train_data), args.clients,
            args.samples_per_client, args.dirichlet_alpha, args.seed,
        )
    malicious_ids = select_malicious_clients(args.clients, args.malicious_fraction, args.seed)
    malicious_set = set(malicious_ids)
    train_labels = dataset_labels(train_data)
    malicious_source_samples = sum(
        int((train_labels[partitions[client_id]] == args.source_label).sum())
        for client_id in malicious_ids
    )
    model = SmallMnistCNN().to(device)
    rows = []

    for round_index in range(args.rounds):
        started = time.perf_counter()
        active = bool(malicious_ids) and attack_is_active(
            round_index, args.rounds, args.attack_schedule
        )
        updates = []
        losses = []
        for client_id, indices in enumerate(partitions):
            label_flip = (
                (args.source_label, args.target_label)
                if active and client_id in malicious_set
                else None
            )
            update, loss = train_client(
                model, train_data, indices, args.local_epochs, args.batch_size,
                args.learning_rate,
                args.seed * 100000 + round_index * 1000 + client_id,
                device, label_flip=label_flip,
            )
            updates.append(update)
            losses.append(loss)

        counts = [len(indices) for indices in partitions]
        reference = weighted_fedavg(updates, counts)
        crypto = {}
        if args.backend == "plaintext":
            average = reference
        elif args.backend == "lee_ckks":
            average, crypto = tenseal_weighted_fedavg(updates, counts)
        elif args.backend == "fedshe_plain":
            average = fedshe_plain_weighted_fedavg(updates, counts)
        else:
            average, crypto = fedshe_ckks_weighted_fedavg(
                updates, counts,
                security_level=args.fedshe_security_level,
                multiplication_depth=args.fedshe_multiplication_depth,
                polynomial_degree=args.fedshe_polynomial_degree,
                round_decimals=args.fedshe_round_decimals,
            )
        correctness = aggregation_error(reference, average)
        current = state_to_numpy(model.state_dict())
        model.load_state_dict(numpy_to_state(add_to_model(current, average), model.state_dict()))
        test_loss, test_accuracy = evaluate(model, test_data, args.batch_size, device)
        labels, predictions = predict(model, test_data, args.batch_size, device)
        row = {
            "backend": args.backend,
            "dataset": args.dataset,
            "seed": args.seed,
            "round": round_index + 1,
            "clients": args.clients,
            "samples_per_client": args.samples_per_client,
            "partition_manifest": str(manifest.relative_to(ROOT)),
            "partitioning": args.partitioning,
            "dirichlet_alpha": (
                args.dirichlet_alpha if args.partitioning == "dirichlet" else None
            ),
            "source_label": args.source_label,
            "target_label": args.target_label,
            "malicious_fraction": args.malicious_fraction,
            "malicious_client_ids": ";".join(map(str, malicious_ids)),
            "malicious_source_samples": malicious_source_samples,
            "attack_schedule": args.attack_schedule,
            "attack_active": active,
            "mean_client_loss": sum(losses) / len(losses),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            **attack_metrics(labels, predictions, args.source_label, args.target_label),
            "round_seconds": time.perf_counter() - started,
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
