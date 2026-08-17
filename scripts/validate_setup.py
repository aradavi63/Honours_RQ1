from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq1_harness.config import iter_environment_paths, load_config, validate_config, yaml
from scripts.validate_provenance import validate_register


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", *args], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate RQ1 submodules, configuration and Conda YAML files"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "rq1_experiments.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_config(config, ROOT)
    errors.extend(
        validate_register(ROOT / "experiments" / "source_adaptation_audit.yaml")
    )

    recorded = {}
    for line in git("ls-files", "--stage", "repos").splitlines():
        mode, sha, stage_path = line.split(maxsplit=2)
        path = stage_path.split("\t", 1)[-1]
        if mode == "160000":
            recorded[path] = sha
    for path, expected in sorted(recorded.items()):
        actual = git("-C", str(ROOT / path), "rev-parse", "HEAD")
        if actual != expected:
            errors.append(f"submodule {path}: expected {expected}, found {actual}")
        else:
            print(f"OK submodule {path} @ {actual[:12]}")

    if len(recorded) != 6:
        errors.append(f"expected 6 recorded submodules, found {len(recorded)}")
    for path in iter_environment_paths(config, ROOT):
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not document.get("name") or not document.get("dependencies"):
            errors.append(f"invalid Conda environment: {path.relative_to(ROOT)}")
        else:
            print(f"OK environment {path.relative_to(ROOT)} ({document['name']})")

    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 1
    print("Setup validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
