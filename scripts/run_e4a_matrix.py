from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ("individual_plaintext", "route_aggregate", "colluding_clients", "ciphertext_only")


def result_filename(
    score_method: str,
    observation: str,
    seed: int,
    clients: int,
    partitioning: str = "iid",
    dirichlet_alpha: float = 1.0,
    nonmember_sampling: str = "random",
) -> str:
    """Keep the original five-client names while isolating robustness configurations."""
    score_prefix = "" if score_method == "margin" else f"{score_method}_"
    client_prefix = "" if clients == 5 else f"clients-{clients}_"
    partition_prefix = "" if partitioning == "iid" else (
        f"dirichlet-alpha-{dirichlet_alpha:g}_".replace(".", "p")
    )
    sampling_prefix = "" if nonmember_sampling == "random" else "class-matched_"
    return f"{partition_prefix}{sampling_prefix}{client_prefix}{score_prefix}{observation}_seed-{seed}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic E4a observation matrix")
    parser.add_argument("--observations", nargs="+", choices=OBSERVATIONS, default=OBSERVATIONS)
    parser.add_argument("--score-method", choices=("margin", "gaussian_cdf"), default="margin")
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--partitioning", choices=("iid", "dirichlet"), default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument(
        "--nonmember-sampling", choices=("random", "class_matched"), default="random"
    )
    parser.add_argument("--samples-per-client", type=int, default=100)
    parser.add_argument("--attack-samples", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.score_method == "gaussian_cdf" and any(
        observation not in ("individual_plaintext", "colluding_clients")
        for observation in args.observations
    ):
        parser.error("Gaussian FedMIA scoring requires client-separated plaintext observations")
    output_dir = ROOT / "results" / "e4a" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    for observation in args.observations:
        for seed in args.seeds:
            output = output_dir / result_filename(
                args.score_method, observation, seed, args.clients,
                args.partitioning, args.dirichlet_alpha, args.nonmember_sampling,
            )
            if output.exists() and not args.overwrite:
                parser.error(f"result already exists: {output.relative_to(ROOT)}")
            command = [
                sys.executable, str(ROOT / "scripts" / "run_e4a.py"),
                "--observation", observation,
                "--score-method", args.score_method,
                "--dataset", "mnist",
                "--partitioning", args.partitioning,
                "--dirichlet-alpha", str(args.dirichlet_alpha),
                "--nonmember-sampling", args.nonmember_sampling,
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
