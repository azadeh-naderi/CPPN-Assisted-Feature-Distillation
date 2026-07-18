import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply "a.b.c=value" dotted-path CLI overrides onto a nested config dict.

    Values are parsed with yaml.safe_load so ints/floats/bools/lists come
    through as their native types instead of strings.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key_path, _, raw_value = item.partition("=")
        if not _:
            raise ValueError(f"Override must be of the form key.path=value, got: {item!r}")
        value = yaml.safe_load(raw_value)
        keys = key_path.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg
