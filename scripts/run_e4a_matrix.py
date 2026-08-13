from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ("individual_plaintext", "route_aggregate", "colluding_clients", "ciphertext_only")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic E4a observation matrix")
    parser.add_argument("--observations", nargs="+", choices=OBSERVATIONS, default=OBSERVATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=100)
    parser.add_argument("--attack-samples", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / "results" / "e4a" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    for observation in args.observations:
        for seed in args.seeds:
            output = output_dir / f"{observation}_seed-{seed}.csv"
            if output.exists() and not args.overwrite:
                parser.error(f"result already exists: {output.relative_to(ROOT)}")
            command = [
                sys.executable, str(ROOT / "scripts" / "run_e4a.py"),
                "--observation", observation,
                "--dataset", "mnist",
                "--clients", str(args.clients),
                "--samples-per-client", str(args.samples_per_client),
                "--attack-samples", str(args.attack_samples),
                "--rounds", str(args.rounds),
                "--seed", str(seed),
                "--output", str(output),
            ]
            print(f"Running {observation} seed={seed}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
