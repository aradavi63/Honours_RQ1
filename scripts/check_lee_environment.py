from __future__ import annotations

import argparse
import importlib
import platform
import sys


REQUIRED_MODULES = (
    "numpy",
    "torch",
    "torchvision",
    "torchtext",
    "tenseal",
    "phe",
    "Crypto",
    "sklearn",
    "pandas",
    "psutil",
)

FULL_REPOSITORY_MODULES = ("datasets",)


def version_of(module) -> str:
    return str(getattr(module, "__version__", "version attribute unavailable"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the Lee CKKS environment and optional full repository stack"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also require optional packages used by Lee's IMDB/data-loading path",
    )
    args = parser.parse_args()
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    if sys.version_info[:2] != (3, 9):
        print("ERROR rq1-lee-he requires Python 3.9", file=sys.stderr)
        return 1

    failures = []
    module_names = REQUIRED_MODULES + (FULL_REPOSITORY_MODULES if args.full else ())
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            print(f"OK {name} {version_of(module)}")

    if failures:
        for failure in failures:
            print(f"ERROR {failure}", file=sys.stderr)
        return 1
    print("Lee environment preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
