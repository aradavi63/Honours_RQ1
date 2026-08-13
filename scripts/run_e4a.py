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
from rq1_harness.membership import (
    membership_metrics,
    per_sample_gradient_cosines,
    spatial_temporal_scores,
    update_as_gradient_vector,
)
from rq1_harness.training import (
    SmallMnistCNN,
    load_or_create_iid_partitions,
    numpy_to_state,
    seed_everything,
    state_to_numpy,
    train_client,
)
from scripts.run_e1 import load_data, write_rows

OBSERVATIONS = ("individual_plaintext", "route_aggregate", "colluding_clients", "ciphertext_only")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E4a FedMIA-style update attack")
    parser.add_argument("--observation", choices=OBSERVATIONS, required=True)
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="mnist")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=100)
    parser.add_argument("--attack-samples", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--partition-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.clients, args.samples_per_client, args.attack_samples, args.rounds, args.batch_size) < 1:
        parser.error("client, sample, round and batch values must be positive")
    if args.clients < 2 and args.observation in ("individual_plaintext", "colluding_clients"):
        parser.error("spatial observations require at least two clients")
    if args.attack_samples > min(args.samples_per_client, 10000):
        parser.error("attack samples exceed the target member or MNIST test pool")

    seed_everything(args.seed)
    device = torch.device("cpu")
    train_data, test_data = load_data(args.dataset, args.data_root, args.seed)
    manifest = (args.partition_manifest or ROOT / "results" / "partitions" / (
        f"{args.dataset}_iid_clients-{args.clients}_samples-{args.samples_per_client}_seed-{args.seed}.json"
    )).resolve()
    partitions = load_or_create_iid_partitions(
        manifest, args.dataset, len(train_data), args.clients, args.samples_per_client, args.seed
    )
    rng = np.random.default_rng(args.seed + 41000)
    member_indices = rng.choice(partitions[0], args.attack_samples, replace=False).tolist()
    nonmember_indices = rng.choice(len(test_data), args.attack_samples, replace=False).tolist()

    if args.observation == "ciphertext_only":
        row = {
            "observation": args.observation,
            "dataset": args.dataset,
            "seed": args.seed,
            "clients": args.clients,
            "rounds": args.rounds,
            "attack_samples_per_class": args.attack_samples,
            "applicability": "not_applicable",
            "reason": "Per-client gradient-update cosine similarity cannot be computed from CKKS ciphertext without decryption.",
            "roc_auc": np.nan,
            "tpr_at_fpr_01": np.nan,
            "tpr_at_fpr_001": np.nan,
            "membership_advantage": np.nan,
        }
    else:
        model = SmallMnistCNN().to(device)
        member_rounds, nonmember_rounds = [], []
        for round_index in range(args.rounds):
            updates = []
            for client_id, indices in enumerate(partitions):
                update, _ = train_client(
                    model, train_data, indices, 1, args.batch_size, args.learning_rate,
                    args.seed * 100000 + round_index * 1000 + client_id, device,
                )
                updates.append(update)
            average = weighted_fedavg(updates, [len(indices) for indices in partitions])
            if args.observation == "route_aggregate":
                observed = [update_as_gradient_vector(model, average)]
            else:
                # With all non-target updates and the aggregate, colluders recover the
                # target update exactly; therefore its observation equals plaintext.
                observed = [update_as_gradient_vector(model, update) for update in updates]
            member_rounds.append(
                per_sample_gradient_cosines(model, train_data, member_indices, observed, device)
            )
            nonmember_rounds.append(
                per_sample_gradient_cosines(model, test_data, nonmember_indices, observed, device)
            )
            current = state_to_numpy(model.state_dict())
            model.load_state_dict(numpy_to_state(add_to_model(current, average), model.state_dict()))

        member_scores = spatial_temporal_scores(member_rounds, args.observation)
        nonmember_scores = spatial_temporal_scores(nonmember_rounds, args.observation)
        row = {
            "observation": args.observation,
            "dataset": args.dataset,
            "seed": args.seed,
            "clients": args.clients,
            "rounds": args.rounds,
            "attack_samples_per_class": args.attack_samples,
            "partition_manifest": str(manifest.relative_to(ROOT)),
            "applicability": "applicable",
            "member_score_mean": float(member_scores.mean()),
            "nonmember_score_mean": float(nonmember_scores.mean()),
            **membership_metrics(member_scores, nonmember_scores),
        }
    print(json.dumps(row, allow_nan=True))
    if args.output:
        write_rows(args.output, [row])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
