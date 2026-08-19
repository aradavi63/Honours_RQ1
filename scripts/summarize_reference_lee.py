"""Summarise paired native outputs from Lee reference reproductions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_single_row(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one data row in {path}, found {len(rows)}")
    return {key: float(value) for key, value in rows[0].items()}


def native_file(directory: Path, suffix: str) -> Path:
    matches = list(directory.glob(f".*_{suffix}.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one native {suffix} CSV in {directory}, found {len(matches)}"
        )
    return matches[0]


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 6),
        "sample_standard_deviation": round(statistics.stdev(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", default="scenario-pilot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "lee",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.seeds) < 2:
        parser.error("at least two seeds are required for a sample standard deviation")

    paired = []
    for seed in args.seeds:
        standard_dir = args.results_dir / f"standard_{args.run_label}_seed-{seed}"
        he_dir = args.results_dir / f"he_{args.run_label}_seed-{seed}"
        standard_scores = read_single_row(native_file(standard_dir, "scores"))
        he_scores = read_single_row(native_file(he_dir, "scores"))
        standard_times = read_single_row(native_file(standard_dir, "times"))
        he_times = read_single_row(native_file(he_dir, "times"))
        standard_tx = read_single_row(native_file(standard_dir, "transmissions"))
        he_tx = read_single_row(native_file(he_dir, "transmissions"))
        standard_total = standard_times["total_time"]
        he_total = he_times["total_time"]
        standard_transmissions = standard_tx["num_transmissions"]
        he_transmissions = (
            he_tx["noise_calc_num_transmissions"]
            + he_tx["other_num_transmissions"]
        )
        paired.append(
            {
                "seed": seed,
                "standard_accuracy_percent": standard_scores["acc_score"],
                "he_accuracy_percent": he_scores["acc_score"],
                "accuracy_difference_standard_minus_he_points": (
                    round(
                        standard_scores["acc_score"] - he_scores["acc_score"], 6
                    )
                ),
                "standard_total_time_seconds": standard_total,
                "he_total_time_seconds": he_total,
                "runtime_ratio_he_over_standard": round(
                    he_total / standard_total, 6
                ),
                "standard_transmissions": int(standard_transmissions),
                "he_transmissions": int(he_transmissions),
                "transmission_ratio_he_over_standard": (
                    he_transmissions / standard_transmissions
                ),
            }
        )

    summary = {
        "run_label": args.run_label,
        "seeds": args.seeds,
        "interpretation_limit": (
            "Three one-round pilot seeds estimate variability but are not a "
            "thesis-scale statistical comparison."
        ),
        "paired_results": paired,
        "aggregate": {
            "standard_accuracy_percent": describe(
                [row["standard_accuracy_percent"] for row in paired]
            ),
            "he_accuracy_percent": describe(
                [row["he_accuracy_percent"] for row in paired]
            ),
            "accuracy_difference_standard_minus_he_points": describe(
                [
                    row["accuracy_difference_standard_minus_he_points"]
                    for row in paired
                ]
            ),
            "standard_total_time_seconds": describe(
                [row["standard_total_time_seconds"] for row in paired]
            ),
            "he_total_time_seconds": describe(
                [row["he_total_time_seconds"] for row in paired]
            ),
            "runtime_ratio_he_over_standard": describe(
                [row["runtime_ratio_he_over_standard"] for row in paired]
            ),
            "transmission_ratio_he_over_standard": describe(
                [row["transmission_ratio_he_over_standard"] for row in paired]
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
