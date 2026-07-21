from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, strict: bool) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if strict:
                raise ValueError(f"Missing environment variable: {name}")
            return match.group(0)
        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, strict) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, strict) for key, item in value.items()}
    return value


def load_config(path: str | Path, *, strict_env: bool = True) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{source}: configuration must be a mapping")
    return _expand(data, strict_env)


def resolve_path(value: str | Path, root: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path

