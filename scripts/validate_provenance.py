from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def python_symbols(path: Path) -> set[str]:
    """Return module functions plus Class.method names from a Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.add(node.name)
        elif isinstance(node, ast.ClassDef):
            found.add(node.name)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.add(f"{node.name}.{child.name}")
    return found


def recorded_submodules() -> dict[str, str]:
    output = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "ls-files", "--stage", "repos"],
        cwd=ROOT,
        text=True,
    )
    recorded = {}
    for line in output.splitlines():
        mode, commit, stage_path = line.split(maxsplit=2)
        path = stage_path.split("\t", 1)[-1].replace("\\", "/")
        if mode == "160000":
            recorded[path] = commit
    return recorded


def validate_register(path: Path) -> list[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    submodules = recorded_submodules()
    for mechanism, entry in document.get("mechanisms", {}).items():
        submodule = entry.get("submodule")
        if submodules.get(submodule) != entry.get("commit"):
            errors.append(
                f"{mechanism}: audit commit {entry.get('commit')} does not match "
                f"recorded {submodules.get(submodule)} for {submodule}"
            )
        for section in ("sources", "harness"):
            for reference in entry.get(section, []):
                reference_path = ROOT / reference["file"]
                if not reference_path.is_file():
                    errors.append(f"{mechanism}: missing {section} file {reference['file']}")
                    continue
                symbols = reference.get("symbols", [])
                if symbols and reference_path.suffix == ".py":
                    available = python_symbols(reference_path)
                    for symbol in symbols:
                        if symbol not in available:
                            errors.append(
                                f"{mechanism}: missing symbol {symbol} in {reference['file']}"
                            )
        for required in ("relationship", "retained", "changed", "rationale", "allowed_claim", "prohibited_claim"):
            if not entry.get(required):
                errors.append(f"{mechanism}: missing {required}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate source-to-adaptation provenance")
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "experiments" / "source_adaptation_audit.yaml",
    )
    args = parser.parse_args()
    errors = validate_register(args.audit)
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 1
    print("Source-to-adaptation provenance validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
