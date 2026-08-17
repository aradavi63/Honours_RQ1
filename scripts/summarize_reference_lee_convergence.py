"""Summarise a paired multi-round Lee reference run from native CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def native_file(directory: Path, suffix: str) -> Path:
    matches = list(directory.glob(f".*_{suffix}.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one native {suffix} CSV in {directory}, found {len(matches)}"
        )
    return matches[0]


def repeated_total(rows: list[dict[str, float]], path: Path) -> float:
    totals = {row["total_time"] for row in rows}
    if len(totals) != 1:
        raise RuntimeError(f"expected Lee's repeated whole-run total in {path}")
    return totals.pop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", default="convergence-pilot")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "lee",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    standard_dir = args.results_dir / f"standard_{args.run_label}_seed-{args.seed}"
    he_dir = args.results_dir / f"he_{args.run_label}_seed-{args.seed}"
    standard_scores = read_rows(native_file(standard_dir, "scores"))
    he_scores = read_rows(native_file(he_dir, "scores"))
    standard_times_path = native_file(standard_dir, "times")
    he_times_path = native_file(he_dir, "times")
    standard_times = read_rows(standard_times_path)
    he_times = read_rows(he_times_path)
    standard_tx = read_rows(native_file(standard_dir, "transmissions"))
    he_tx = read_rows(native_file(he_dir, "transmissions"))
    row_counts = {
        len(standard_scores),
        len(he_scores),
        len(standard_times),
        len(he_times),
        len(standard_tx),
        len(he_tx),
    }
    if len(row_counts) != 1 or not row_counts:
        raise RuntimeError("native score, timing and transmission row counts differ")
    rounds = row_counts.pop()

    trajectory = []
    for index in range(rounds):
        standard_accuracy = standard_scores[index]["acc_score"]
        he_accuracy = he_scores[index]["acc_score"]
        standard_transmissions = int(standard_tx[index]["num_transmissions"])
        he_transmissions = int(
            he_tx[index]["noise_calc_num_transmissions"]
            + he_tx[index]["other_num_transmissions"]
        )
        trajectory.append(
            {
                "round": index + 1,
                "standard_accuracy_percent": standard_accuracy,
                "he_accuracy_percent": he_accuracy,
                "accuracy_difference_standard_minus_he_points": round(
                    standard_accuracy - he_accuracy, 6
                ),
                "standard_mean_client_loss": standard_scores[index]["loss_score"],
                "he_mean_client_loss": he_scores[index]["loss_score"],
                "standard_epoch_time_seconds": standard_times[index]["epoch_times"],
                "he_epoch_time_seconds": he_times[index]["epoch_times"],
                "standard_transmissions": standard_transmissions,
                "he_transmissions": he_transmissions,
            }
        )

    standard_total = repeated_total(standard_times, standard_times_path)
    he_total = repeated_total(he_times, he_times_path)
    standard_transmission_total = sum(
        row["standard_transmissions"] for row in trajectory
    )
    he_transmission_total = sum(row["he_transmissions"] for row in trajectory)
    summary = {
        "run_label": args.run_label,
        "seed": args.seed,
        "rounds": rounds,
        "interpretation_limit": (
            "One three-round paired pilot demonstrates execution and convergence; "
            "it does not estimate cross-seed uncertainty or reproduce the full "
            "20-round, five-local-epoch reference configuration."
        ),
        "native_timing_note": (
            "Lee repeats the whole-run total_time in every round row; this summary "
            "uses that repeated value once and does not sum it."
        ),
        "trajectory": trajectory,
        "whole_run": {
            "standard_total_time_seconds": standard_total,
            "he_total_time_seconds": he_total,
            "runtime_ratio_he_over_standard": round(he_total / standard_total, 6),
            "standard_transmissions": standard_transmission_total,
            "he_transmissions": he_transmission_total,
            "transmission_ratio_he_over_standard": round(
                he_transmission_total / standard_transmission_total, 6
            ),
        },
    }
    output = args.output or args.results_dir / f"{args.run_label}-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Recorded {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
