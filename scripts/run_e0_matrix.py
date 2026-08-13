from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.aggregation import weighted_fedavg
from rq1_harness.e0 import deterministic_updates, tenseal_weighted_fedavg
from rq1_harness.fedshe import (
    fedshe_ckks_weighted_fedavg,
    fedshe_plain_weighted_fedavg,
)
from rq1_harness.metrics import aggregation_error


BACKENDS = ("plaintext", "lee_ckks", "fedshe_plain", "fedshe_ckks")


def run_one(
    backend: str,
    clients: int,
    seed: int,
    fedshe_security_level: str = "128",
    fedshe_multiplication_depth: str = "0",
    fedshe_polynomial_degree: str = "16384",
    fedshe_round_decimals: int = 3,
) -> dict[str, object]:
    updates, counts = deterministic_updates(clients, seed)
    reference = weighted_fedavg(updates, counts)
    if backend == "plaintext":
        candidate, timing = weighted_fedavg(updates, counts), {}
    elif backend == "lee_ckks":
        candidate, timing = tenseal_weighted_fedavg(updates, counts)
    elif backend == "fedshe_plain":
        candidate, timing = fedshe_plain_weighted_fedavg(updates, counts), {}
    elif backend == "fedshe_ckks":
        candidate, timing = fedshe_ckks_weighted_fedavg(
            updates,
            counts,
            security_level=fedshe_security_level,
            multiplication_depth=fedshe_multiplication_depth,
            polynomial_degree=fedshe_polynomial_degree,
            round_decimals=fedshe_round_decimals,
        )
    else:
        raise ValueError(f"unsupported backend: {backend}")
    metrics = aggregation_error(reference, candidate)
    passed = (
        metrics["relative_l2_error"] <= 1e-3
        and metrics["cosine_similarity"] >= 0.9999
    )
    row = {
        "backend": backend,
        "clients": clients,
        "seed": seed,
        "passes_acceptance": passed,
        **metrics,
        **timing,
    }
    if backend == "fedshe_ckks":
        row.update(
            {
                "security_level": fedshe_security_level,
                "multiplication_depth": fedshe_multiplication_depth,
                "polynomial_degree": fedshe_polynomial_degree,
                "round_decimals": fedshe_round_decimals,
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete deterministic E0 matrix")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "e0")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=("plaintext", "lee_ckks"))
    parser.add_argument("--fedshe-security-level", default="128")
    parser.add_argument("--fedshe-multiplication-depth", default="0")
    parser.add_argument("--fedshe-polynomial-degree", default="16384")
    parser.add_argument("--fedshe-round-decimals", type=int, default=3)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="write the complete matrix without a non-zero exit for failed acceptance checks",
    )
    args = parser.parse_args()
    clients = (1, 2, 5, 10)
    seeds = (1, 2, 3, 4, 5)
    failed = False
    for backend in args.backends:
        rows = []
        for client_count in clients:
            for seed in seeds:
                row = run_one(
                    backend,
                    client_count,
                    seed,
                    fedshe_security_level=args.fedshe_security_level,
                    fedshe_multiplication_depth=args.fedshe_multiplication_depth,
                    fedshe_polynomial_degree=args.fedshe_polynomial_degree,
                    fedshe_round_decimals=args.fedshe_round_decimals,
                )
                rows.append(row)
                passed = bool(row["passes_acceptance"])
                failed |= not passed
                print(f"{backend} clients={client_count} seed={seed} pass={passed}")
        filename = (
            f"fedshe_ckks_round{args.fedshe_round_decimals}.csv"
            if backend == "fedshe_ckks"
            else f"{backend}.csv"
        )
        write_csv(args.output_dir / filename, rows)
    return int(failed and not args.allow_failures)


if __name__ == "__main__":
    raise SystemExit(main())
