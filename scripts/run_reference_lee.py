from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import ssl
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import certifi
import psutil
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "repos" / "server-initiated_HE_FL"
EXPECTED_COMMIT = "80575b5ff813fb2b4ba1a7786576ca65bf5d3303"


class TextSink:
    """Minimal ScrolledText-compatible sink used by the original logger."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def insert(self, _position, message: str) -> None:
        self.messages.append(str(message))
        print(str(message), end="")

    def see(self, _position) -> None:
        return None


class RootSink:
    """Provide the quit method called by the original completion path."""

    def quit(self) -> None:
        return None


def reference_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
    ).strip()


def ensure_mnist_cache() -> None:
    """Mirror existing MNIST files into the path hard-coded by Lee's loader."""
    source = ROOT / "data" / "MNIST" / "raw"
    destination = ROOT / "data" / "mnist" / "MNIST" / "raw"
    required = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"shared MNIST cache is missing: {', '.join(missing)}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in required:
        target = destination / name
        if not target.exists():
            shutil.copy2(source / name, target)


def use_certifi_store() -> None:
    """Avoid this machine's malformed Windows certificate-store entry."""
    original = ssl.create_default_context

    def create_context(*args, **kwargs):
        kwargs.setdefault("cafile", certifi.where())
        return original(*args, **kwargs)

    ssl.create_default_context = create_context
    ssl._create_default_https_context = create_context


def reference_args(cli: argparse.Namespace, temporary_output: str) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=cli.rounds,
        num_users=cli.clients,
        num_samples=cli.samples_per_client,
        alpha=cli.alpha,
        frac=1,
        local_ep=cli.local_epochs,
        local_bs=cli.batch_size,
        bs=128,
        lr=0.01,
        momentum=0.9,
        model="cnn",
        partition_size=cli.partition_size,
        output_directory=temporary_output,
        max_seq_len=256,
        embed_dim=100,
        hidden_dim=128,
        max_vocab_size=20000,
        dataset="MNIST",
        iid=False,
        num_classes=10,
        gpu=-1,
        all_clients=True,
        device=torch.device("cpu"),
    )


def is_thesis_baseline(args: argparse.Namespace) -> bool:
    return (
        args.clients == 10
        and args.samples_per_client == 100
        and args.rounds == 20
        and args.local_epochs == 5
        and args.batch_size == 12
        and args.alpha == 1.0
        and args.partition_size == 10
    )


def load_standard_module():
    spec = importlib.util.spec_from_file_location(
        "lee_standard_reference", REFERENCE / "standard_fl_implementation.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Lee standard FL entry point")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_simulation(mode: str, args: SimpleNamespace):
    if mode == "he":
        network_module = importlib.import_module("network_node")
        server_module = importlib.import_module("server_node")
        client_module = importlib.import_module("client_node")
    else:
        module = load_standard_module()
        network_module = server_module = client_module = module
    network = network_module.NetworkSimulationClass(args)
    server = server_module.ServerNodeClass(0, network, args)
    network.addNode(server)
    for node_id in range(1, args.num_users + 1):
        network.addNode(client_module.ClientNodeClass(node_id, network, args))
    server.getNodeList(network.getNodes())
    network.root = RootSink()
    return network


def run_headless(network) -> None:
    text = TextSink()
    plots, (loss_axis, accuracy_axis) = plt.subplots(2, 1)
    plot_canvas = FigureCanvasAgg(plots)
    route_figure, route_axis = plt.subplots()
    route_canvas = FigureCanvasAgg(route_figure)
    try:
        network.initialiseLearningFixtures(
            text,
            loss_axis,
            accuracy_axis,
            plots,
            plot_canvas,
            route_canvas,
            route_axis,
        )
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        plt.close(plots)
        plt.close(route_figure)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(
        description="Run Lee's pinned GUI simulation interactively or headlessly"
    )
    parser.add_argument("--mode", choices=("he", "standard"), default="he")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--samples-per-client", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--partition-size", type=int, default=2)
    parser.add_argument(
        "--run-label",
        default="smoke",
        help="filename-safe label distinguishing reference configurations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "lee",
    )
    args = parser.parse_args()
    if min(
        args.clients,
        args.samples_per_client,
        args.rounds,
        args.local_epochs,
        args.batch_size,
        args.partition_size,
    ) < 1:
        parser.error("counts must be positive")
    if args.partition_size > args.clients:
        parser.error("partition size cannot exceed client count")
    if args.clients % args.partition_size:
        parser.error("Lee's route builder requires clients divisible by partition size")
    if args.alpha <= 0:
        parser.error("alpha must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.run_label):
        parser.error("run label must contain only lowercase letters, numbers and hyphens")
    commit = reference_commit()
    if commit != EXPECTED_COMMIT:
        parser.error(f"expected Lee commit {EXPECTED_COMMIT}, found {commit}")

    ensure_mnist_cache()
    os.environ["SSL_CERT_FILE"] = certifi.where()
    use_certifi_store()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir = args.output_dir.resolve()
    stem = f"{args.mode}_{args.run_label}_seed-{args.seed}"
    temporary_name = f".lee-reference-{stem}"
    temporary_output = ROOT / temporary_name
    if temporary_output.exists():
        parser.error(f"temporary output already exists: {temporary_output}")
    temporary_output.mkdir()

    original_cwd = Path.cwd()
    original_path = sys.path[:]
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(REFERENCE))
        simulation_args = reference_args(args, temporary_name)
        network = build_simulation(args.mode, simulation_args)
        try:
            if args.gui:
                network.create_gui()
            else:
                run_headless(network)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path

    destination = args.output_dir / stem
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        parser.error(f"result already exists: {destination}")
    shutil.move(str(temporary_output), str(destination))
    score_path = next(destination.glob("*_scores.csv"))
    with score_path.open(newline="", encoding="utf-8") as stream:
        score_rows = list(csv.DictReader(stream))
    if not score_rows:
        raise RuntimeError("Lee's native score CSV is empty")
    metadata = {
        "status": "reference_run_completed",
        "reference_repo": "repos/server-initiated_HE_FL",
        "reference_commit": commit,
        "reference_source_modified": False,
        "mode": args.mode,
        "gui": args.gui,
        "headless_adapter": not args.gui,
        "run_label": args.run_label,
        "scale_note": (
            "Matches the MNIST baseline configuration reported in Lee's thesis."
            if is_thesis_baseline(args)
            else (
                "Reduced smoke configuration; not a thesis/paper-scale result."
                if args.run_label == "smoke"
                else "Reference scenario pilot; not a full 20-round result."
            )
        ),
        "configuration_basis": (
            "Lee thesis Section 7.2 and Figures 3-6; num_samples, batch size, "
            "learning rate and momentum use the pinned repository defaults."
            if is_thesis_baseline(args)
            else "User-selected reference reproduction configuration."
        ),
        "seed_injected_by_wrapper": args.seed,
        "determinism_note": (
            "The original client threads share process-level random state; repeated "
            "runs may differ despite wrapper seeding."
        ),
        "configuration": {
            "dataset": "MNIST",
            "model": "reference Mnistcnn",
            "clients": args.clients,
            "samples_per_client": args.samples_per_client,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "dirichlet_alpha": args.alpha,
            "partition_size": args.partition_size,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": importlib.metadata.version("torchvision"),
            "tenseal_version": importlib.metadata.version("tenseal"),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "physical_memory_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        },
        "result": {
            "final_accuracy_percent": float(score_rows[-1]["acc_score"]),
            "final_mean_client_loss": float(score_rows[-1]["loss_score"]),
        },
        "native_files": sorted(path.name for path in destination.glob("*.csv")),
        "native_result_directory": str(destination.relative_to(ROOT)),
    }
    (destination / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Recorded {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
