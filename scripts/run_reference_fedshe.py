"""Run the pinned FedSHE entry point without modifying its source tree."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import io
import json
import os
import platform
import random
import re
import runpy
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "repos" / "FedSHE"
EXPECTED_COMMIT = "7a354246782b603616b09ed05b4c3fc0cfed92e8"


class Tee(io.TextIOBase):
    def __init__(self, console: io.TextIOBase, capture: io.StringIO) -> None:
        self.console = console
        self.capture = capture

    def write(self, text: str) -> int:
        self.console.write(text)
        self.console.flush()
        self.capture.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()

    def fileno(self) -> int:
        return self.console.fileno()

    def isatty(self) -> bool:
        return self.console.isatty()


def reference_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
    ).strip()


def install_plain_import_shims() -> None:
    """Satisfy unused Unix/Pyfhel imports required by FedSHE Plain mode."""
    if os.name == "nt" and "resource" not in sys.modules:
        sys.modules["resource"] = types.ModuleType("resource")
    try:
        importlib.import_module("Pyfhel")
    except ImportError:
        module = types.ModuleType("Pyfhel")

        class UnavailablePyfhel:
            def __init__(self, *_args, **_kwargs) -> None:
                raise RuntimeError(
                    "Pyfhel is unavailable; CKKS must use rq1-fedshe-ckks in WSL"
                )

        module.Pyfhel = UnavailablePyfhel
        sys.modules["Pyfhel"] = module


def parse_list(log: str, label: str) -> list[float]:
    matches = re.findall(rf"^{re.escape(label)}:\s*(\[.*\])\s*$", log, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} list in FedSHE output")
    return [float(value) for value in ast.literal_eval(matches[0])]


def parse_total_time(log: str) -> float:
    matches = re.findall(r"^all took time\(\):\s*([0-9.eE+-]+)\s*$", log, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("expected one total runtime in FedSHE output")
    return float(matches[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute FedSHE's pinned original main.py with recorded provenance"
    )
    parser.add_argument("--mode", choices=("Plain",), default="Plain")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--run-label", default="smoke")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "fedshe",
    )
    args = parser.parse_args()
    if min(args.clients, args.rounds, args.local_epochs, args.batch_size) < 1:
        parser.error("counts must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.run_label):
        parser.error("run label must contain only lowercase letters, numbers and hyphens")
    commit = reference_commit()
    if commit != EXPECTED_COMMIT:
        parser.error(f"expected FedSHE commit {EXPECTED_COMMIT}, found {commit}")

    destination = args.output_dir.resolve() / (
        f"{args.mode.lower()}_{args.run_label}_seed-{args.seed}"
    )
    if destination.exists():
        parser.error(f"result already exists: {destination}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_plain_import_shims()
    original_cwd = Path.cwd()
    original_path = sys.path[:]
    original_argv = sys.argv[:]
    capture = io.StringIO()
    try:
        sys.path.insert(0, str(REFERENCE))
        os.chdir(REFERENCE)
        importlib.import_module("SegCKKS")
        os.chdir(ROOT)
        sys.argv = [
            str(REFERENCE / "main.py"),
            "--gpu",
            "-1",
            "--dataset",
            "mnist",
            "--model",
            "LeNet",
            "--num_channels",
            "1",
            "--epochs",
            str(args.rounds),
            "--num_users",
            str(args.clients),
            "--local_ep",
            str(args.local_epochs),
            "--local_bs",
            str(args.batch_size),
            "--lr",
            str(args.learning_rate),
            "--momentum",
            str(args.momentum),
            "--mode",
            args.mode,
            "--no-plot",
        ]
        tee = Tee(sys.stdout, capture)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            runpy.run_path(str(REFERENCE / "main.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
        os.chdir(original_cwd)

    log = capture.getvalue()
    destination.mkdir(parents=True)
    (destination / "native_stdout.log").write_text(log, encoding="utf-8")
    metrics = {
        "training_accuracy_percent": parse_list(log, "acc_train"),
        "test_accuracy_percent": parse_list(log, "acc_test"),
        "global_loss": parse_list(log, "loss_glob"),
        "test_loss": parse_list(log, "loss_test"),
        "total_time_seconds": parse_total_time(log),
    }
    expected_rows = args.rounds
    if any(
        len(metrics[key]) != expected_rows
        for key in (
            "training_accuracy_percent",
            "test_accuracy_percent",
            "global_loss",
            "test_loss",
        )
    ):
        raise RuntimeError("FedSHE output metric counts do not match requested rounds")
    (destination / "reference_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "status": "reference_run_completed",
        "reference_repo": "repos/FedSHE",
        "reference_commit": commit,
        "reference_source_modified": False,
        "mode": args.mode,
        "run_label": args.run_label,
        "compatibility_shims": {
            "resource": "unused Unix import supplied on Windows",
            "Pyfhel": "unused import-only placeholder in Plain mode",
        },
        "fidelity_note": (
            "Original main.py, client.py, server.py, LeNet, IID partitioning, "
            "local training and FedAvg execute unchanged. The wrapper injects a "
            "seed, arguments, compatible imports and result capture."
        ),
        "scale_note": (
            "Full 60,000-sample MNIST training set, but reduced rounds/local epochs; "
            "not a paper-scale result."
        ),
        "configuration": {
            "dataset": "MNIST",
            "model": "LeNetMnist",
            "iid_clients": args.clients,
            "global_rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "local_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "training_samples": 60000,
            "test_samples": 10000,
        },
        "seed_injected_by_wrapper": args.seed,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_cores": os.cpu_count(),
        },
        "result": {
            "final_test_accuracy_percent": metrics["test_accuracy_percent"][-1],
            "final_test_loss": metrics["test_loss"][-1],
            "total_time_seconds": metrics["total_time_seconds"],
        },
    }
    (destination / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Recorded {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
