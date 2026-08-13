from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("plaintext", "lee_ckks", "fedshe_plain", "fedshe_ckks")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic E4b matrix")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fedshe-security-level", default="128")
    parser.add_argument("--fedshe-multiplication-depth", default="0")
    parser.add_argument("--fedshe-polynomial-degree", default="16384")
    parser.add_argument("--fedshe-round-decimals", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / "results" / "e4b" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        output = output_dir / f"{args.backend}_seed-{seed}.csv"
        if output.exists() and not args.overwrite:
            parser.error(f"result already exists: {output.relative_to(ROOT)}")
        command = [
            sys.executable, str(ROOT / "scripts" / "run_e4b.py"),
            "--backend", args.backend,
            "--dataset", "mnist",
            "--clients", str(args.clients),
            "--samples-per-client", str(args.samples_per_client),
            "--rounds", str(args.rounds),
            "--seed", str(seed),
            "--output", str(output),
            "--fedshe-security-level", args.fedshe_security_level,
            "--fedshe-multiplication-depth", args.fedshe_multiplication_depth,
            "--fedshe-polynomial-degree", args.fedshe_polynomial_degree,
            "--fedshe-round-decimals", str(args.fedshe_round_decimals),
        ]
        print(f"Running {args.backend} seed={seed}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
