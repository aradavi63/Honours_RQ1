from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = (
    "individual_plaintext",
    "aggregate_2",
    "aggregate_5",
    "aggregate_10",
    "ciphertext_only",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic E3 observation matrix")
    parser.add_argument("--observations", nargs="+", choices=OBSERVATIONS, default=OBSERVATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="mnist")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.learning_rate <= 0:
        parser.error("iterations and learning rate must be positive")

    output_dir = ROOT / "results" / "e3" / "multiseed"
    reconstruction_dir = ROOT / "results" / "e3" / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    for observation in args.observations:
        for seed in args.seeds:
            stem = f"{observation}_seed-{seed}"
            output = output_dir / f"{stem}.csv"
            if output.exists() and not args.overwrite:
                parser.error(f"result already exists: {output.relative_to(ROOT)}")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_e3.py"),
                "--observation",
                observation,
                "--dataset",
                args.dataset,
                "--seed",
                str(seed),
                "--iterations",
                str(args.iterations),
                "--learning-rate",
                str(args.learning_rate),
                "--output",
                str(output),
            ]
            if observation != "ciphertext_only":
                command.extend(
                    ["--save-reconstruction", str(reconstruction_dir / f"{stem}.pt")]
                )
            print(f"Running {observation} seed={seed}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
