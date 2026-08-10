from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only before environment setup
    yaml = None


REQUIRED_REPOSITORIES = {
    "lee_repo",
    "fedshe_repo",
    "fedavg_repo",
    "gradient_attack_repo",
    "poisoning_attack_repo",
    "membership_attack_repo",
}


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Activate rq1-core or install the dependencies "
            "from environment.yml before running setup validation."
        )
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    return config


def validate_config(config: Dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    for section in ("study", "common_paths", "mechanisms", "experiments"):
        if section not in config:
            errors.append(f"missing top-level section: {section}")

    paths = config.get("common_paths", {})
    for key in sorted(REQUIRED_REPOSITORIES):
        value = paths.get(key)
        if not value:
            errors.append(f"common_paths.{key} is missing")
        elif not (root / value).is_dir():
            errors.append(f"common_paths.{key} does not exist: {value}")

    mechanisms = config.get("mechanisms", {})
    for name, mechanism in mechanisms.items():
        environment = mechanism.get("environment")
        runner = mechanism.get("runner")
        if not environment or not (root / environment).is_file():
            errors.append(f"mechanisms.{name}.environment is missing: {environment}")
        if not runner or not (root / runner).is_file():
            errors.append(f"mechanisms.{name}.runner is missing: {runner}")

    seeds = config.get("study", {}).get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds):
        errors.append("study.seeds must contain unique seed values")
    return errors


def iter_environment_paths(config: Dict[str, Any], root: Path) -> Iterable[Path]:
    seen = set()
    for mechanism in config.get("mechanisms", {}).values():
        path = root / mechanism["environment"]
        if path not in seen:
            seen.add(path)
            yield path
