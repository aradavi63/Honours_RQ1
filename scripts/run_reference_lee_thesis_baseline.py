"""Preview or execute Lee's exact reported MNIST baseline configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_reference_lee.py"

CONFIGURATION = {
    "dataset": "MNIST",
    "model": "cnn",
    "clients": 10,
    "samples_per_client": 100,
    "rounds": 20,
    "local_epochs": 5,
    "batch_size": 12,
    "alpha": 1.0,
    "partition_size": 10,
}


def command(mode: str, seed: int) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--run-label",
        "thesis-baseline",
        "--seed",
        str(seed),
        "--clients",
        str(CONFIGURATION["clients"]),
        "--samples-per-client",
        str(CONFIGURATION["samples_per_client"]),
        "--rounds",
        str(CONFIGURATION["rounds"]),
        "--local-epochs",
        str(CONFIGURATION["local_epochs"]),
        "--batch-size",
        str(CONFIGURATION["batch_size"]),
        "--alpha",
        str(CONFIGURATION["alpha"]),
        "--partition-size",
        str(CONFIGURATION["partition_size"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Lee's thesis MNIST baseline; defaults to a non-executing preview"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=("both", "standard", "he"), default="both")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="start the long-running reference experiment",
    )
    args = parser.parse_args()
    modes = ["standard", "he"] if args.mode == "both" else [args.mode]
    plan = {
        "status": "execution_requested" if args.execute else "dry_run_only",
        "configuration_basis": (
            "Lee thesis Section 7.2 gives 10 clients, five local epochs, one "
            "route and alpha=1; Figures 3-6 show 20 global epochs. Repository "
            "defaults supply 100 samples/client and batch size 12."
        ),
        "configuration": CONFIGURATION,
        "seed": args.seed,
        "modes": modes,
        "scope_warning": (
            "One seed recreates the reported configuration but does not establish "
            "cross-seed uncertainty. Runtime is hardware-specific."
        ),
        "commands": [subprocess.list2cmdline(command(mode, args.seed)) for mode in modes],
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to start the long-running experiment.")
        return 0
    for mode in modes:
        subprocess.run(command(mode, args.seed), cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
