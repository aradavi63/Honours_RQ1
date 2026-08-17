from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import random
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "repos" / "FedAvg"


def reference_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pinned FedAvg entry point without modifying its source"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if min(args.clients, args.shards, args.rounds, args.local_epochs) < 1:
        parser.error("client, shard, round and epoch counts must be positive")
    if args.shards != 2 * args.clients:
        parser.error("the reference non-IID sampler assigns two shards per client")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["WANDB_MODE"] = "disabled"

    original_cwd = Path.cwd()
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    transcript = io.StringIO()
    try:
        os.chdir(REFERENCE)
        sys.path.insert(0, str(REFERENCE))
        sys.argv = [
            "fed_avg.py",
            "--data_root", str(ROOT / "data"),
            "--model_name", "cnn",
            "--non_iid", "1",
            "--n_clients", str(args.clients),
            "--n_shards", str(args.shards),
            "--frac", str(args.fraction),
            "--n_epochs", str(args.rounds),
            "--n_client_epochs", str(args.local_epochs),
            "--lr", "0.01",
            "--momentum", "0.9",
            "--log_every", "1",
            "--early_stopping", "0",
            # Logger otherwise reads an unset self.wandb attribute. Disabled mode
            # initializes the original logger without sending external telemetry.
            "--wandb", "True",
        ]
        with contextlib.redirect_stdout(transcript):
            runpy.run_path(str(REFERENCE / "fed_avg.py"), run_name="__main__")
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        sys.path[:] = original_path

    native_output = transcript.getvalue()
    accuracy_matches = re.findall(r"Avg Test Accuracy: ([0-9.eE+-]+)", native_output)
    loss_matches = re.findall(r"Avg Test Loss: ([0-9.eE+-]+)", native_output)
    if not accuracy_matches or not loss_matches:
        raise RuntimeError("the original FedAvg output did not contain final metrics")
    record = {
        "status": "reference_smoke_completed",
        "reference_repo": "repos/FedAvg",
        "reference_commit": reference_commit(),
        "reference_entrypoint": "fed_avg.py",
        "reference_source_modified": False,
        "scale_note": "Reduced smoke configuration; not a paper-scale reproduction.",
        "seed_injected_by_wrapper": args.seed,
        "configuration": {
            "dataset": "MNIST",
            "model": "reference CNN",
            "non_iid": True,
            "clients": args.clients,
            "shards": args.shards,
            "client_fraction": args.fraction,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "learning_rate": 0.01,
            "momentum": 0.9,
        },
        "result": {
            "final_test_loss": float(loss_matches[-1]),
            "final_test_accuracy": float(accuracy_matches[-1]),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "platform": platform.platform(),
            "wandb_mode": "disabled",
        },
        "native_stdout": native_output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(native_output, end="")
    print(f"Recorded {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
