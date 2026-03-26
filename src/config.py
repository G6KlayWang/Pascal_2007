from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = deepcopy(config)
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Override must look like key=value, got: {raw}")
        dotted_key, raw_value = raw.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = updated
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return updated
