from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.aggregation import weighted_fedavg
from rq1_harness.e0 import deterministic_updates, tenseal_weighted_fedavg
from rq1_harness.metrics import aggregation_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run E0 deterministic aggregation correctness tests"
    )
    parser.add_argument(
        "--backend", choices=("plaintext", "lee_ckks"), default="plaintext"
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients must be positive")

    updates, counts = deterministic_updates(args.clients, args.seed)
    reference = weighted_fedavg(updates, counts)
    timing = {}
    if args.backend == "plaintext":
        candidate = weighted_fedavg(updates, counts)
    else:
        candidate, timing = tenseal_weighted_fedavg(updates, counts)
    result = {
        "backend": args.backend,
        "clients": args.clients,
        "seed": args.seed,
        **aggregation_error(reference, candidate),
        **timing,
    }
    print(json.dumps(result, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        exists = args.output.exists()
        with args.output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            if not exists:
                writer.writeheader()
            writer.writerow(result)
    return int(
        result["relative_l2_error"] > 1e-3
        or result["cosine_similarity"] < 0.9999
    )


if __name__ == "__main__":
    raise SystemExit(main())

