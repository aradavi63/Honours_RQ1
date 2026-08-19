"""Summarise paired FedSHE Plain and CKKS reference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", default="smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "fedshe",
    )
    args = parser.parse_args()
    plain_dir = args.results_dir / f"plain_{args.run_label}_seed-{args.seed}"
    ckks_dir = args.results_dir / f"ckks_{args.run_label}_seed-{args.seed}"
    plain_metrics = read_json(plain_dir / "reference_metrics.json")
    ckks_metrics = read_json(ckks_dir / "reference_metrics.json")
    plain_metadata = read_json(plain_dir / "reference_metadata.json")
    ckks_metadata = read_json(ckks_dir / "reference_metadata.json")

    compared_keys = (
        "dataset",
        "model",
        "iid_clients",
        "global_rounds",
        "local_epochs",
        "local_batch_size",
        "learning_rate",
        "momentum",
        "training_samples",
        "test_samples",
    )
    mismatches = [
        key
        for key in compared_keys
        if plain_metadata["configuration"][key]
        != ckks_metadata["configuration"][key]
    ]
    if mismatches:
        raise RuntimeError(f"paired configurations differ: {', '.join(mismatches)}")

    plain_accuracy = plain_metrics["test_accuracy_percent"][-1]
    ckks_accuracy = ckks_metrics["test_accuracy_percent"][-1]
    global_rounds = plain_metadata["configuration"]["global_rounds"]
    local_epochs = plain_metadata["configuration"]["local_epochs"]
    if global_rounds == 1:
        interpretation_limit = (
            "This one-round smoke verifies original-code execution and approximate "
            "CKKS utility, not paper-scale convergence. Plain ran on native Windows "
            "and CKKS under WSL, so total runtimes are recorded but must not be used "
            "as a controlled encryption-overhead comparison."
        )
    elif global_rounds == 10 and local_epochs == 10:
        interpretation_limit = (
            "This run uses the README's full 10-global-round, 10-local-epoch MNIST "
            "schedule with the original pinned code. Plain ran on native Windows "
            "and CKKS under WSL, so total runtimes are recorded but must not be used "
            "as a controlled encryption-overhead comparison."
        )
    else:
        interpretation_limit = (
            f"This {global_rounds}-round pilot verifies original-code convergence "
            "and approximate CKKS utility, not the paper's full 10-global-round, "
            "10-local-epoch schedule. Plain ran on native Windows and CKKS under "
            "WSL, so total runtimes are recorded but must not be used as a "
            "controlled encryption-overhead comparison."
        )
    summary = {
        "run_label": args.run_label,
        "seed": args.seed,
        "reference_commit": plain_metadata["reference_commit"],
        "configuration": {
            key: plain_metadata["configuration"][key] for key in compared_keys
        },
        "result": {
            "plain_final_test_accuracy_percent": plain_accuracy,
            "ckks_final_test_accuracy_percent": ckks_accuracy,
            "accuracy_difference_plain_minus_ckks_points": round(
                plain_accuracy - ckks_accuracy, 6
            ),
            "plain_total_time_seconds": plain_metrics["total_time_seconds"],
            "ckks_total_time_seconds": ckks_metrics["total_time_seconds"],
        },
        "interpretation_limit": interpretation_limit,
    }
    output = args.results_dir / f"{args.run_label}-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Recorded {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
