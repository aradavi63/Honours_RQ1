from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.aggregation import weighted_fedavg
from rq1_harness.e0 import deterministic_updates, tenseal_weighted_fedavg
from rq1_harness.metrics import aggregation_error


def run_one(backend: str, clients: int, seed: int) -> dict[str, object]:
    updates, counts = deterministic_updates(clients, seed)
    reference = weighted_fedavg(updates, counts)
    if backend == "plaintext":
        candidate, timing = weighted_fedavg(updates, counts), {}
    else:
        candidate, timing = tenseal_weighted_fedavg(updates, counts)
    return {
        "backend": backend,
        "clients": clients,
        "seed": seed,
        **aggregation_error(reference, candidate),
        **timing,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete deterministic E0 matrix")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "e0")
    args = parser.parse_args()
    clients = (1, 2, 5, 10)
    seeds = (1, 2, 3, 4, 5)
    failed = False
    for backend in ("plaintext", "lee_ckks"):
        rows = []
        for client_count in clients:
            for seed in seeds:
                row = run_one(backend, client_count, seed)
                rows.append(row)
                passed = (
                    float(row["relative_l2_error"]) <= 1e-3
                    and float(row["cosine_similarity"]) >= 0.9999
                )
                failed |= not passed
                print(f"{backend} clients={client_count} seed={seed} pass={passed}")
        write_csv(args.output_dir / f"{backend}.csv", rows)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
