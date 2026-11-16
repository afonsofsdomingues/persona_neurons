"""Small helpers shared by all scripts: configuration, files, provenance.

"Provenance" means recording where a result came from. Every output file gets
the config that produced it, the exact model commit, and the git commit of this
repository, so a number can always be traced back to the code that made it.
"""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml


def load_config(path: str | Path, smoke: bool = False) -> dict:
    """Read a YAML config. In smoke mode, the values under the ``smoke:`` key
    replace the normal ones, giving a tiny run that finishes in seconds on CPU.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    overrides = config.pop("smoke", {}) or {}
    if smoke:
        config = deep_merge(config, overrides)
    config["smoke_mode"] = smoke
    config["config_path"] = str(path)
    return config


def deep_merge(base: dict, overrides: dict) -> dict:
    """Return ``base`` with ``overrides`` applied, descending into nested dicts."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def git_commit() -> str:
    """Current git commit of this repository, with ``-dirty`` appended when
    there are uncommitted changes. Returns a readable placeholder when the
    repository has no commits yet."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit + ("-dirty" if status else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git-commit"


def run_metadata(config: dict, model_info: dict, label: str) -> dict:
    """Provenance block stored with every output file."""
    return {
        "label": label,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": model_info,
        "config": config,
    }


def save_json(path: str | Path, data: dict, meta: dict) -> None:
    """Write ``data`` as JSON with the provenance block under ``"meta"``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, **data}, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_jsonl(path: str | Path, rows: list[dict], meta: dict) -> None:
    """Write one JSON object per line. Provenance goes in a sidecar file named
    ``<name>.meta.json`` so that every line of the main file has the same shape.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
