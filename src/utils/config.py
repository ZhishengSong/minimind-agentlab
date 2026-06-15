"""Configuration file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _parse_simple_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _load_simple_yaml(text: str, path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line in {path}:{line_no}: {raw_line!r}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_simple_scalar(value.strip())
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        text = f.read()

    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _load_simple_yaml(text, config_path)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {config_path}, got {type(data).__name__}.")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        else:
            for key, value in data.items():
                if isinstance(value, bool):
                    value_text = "true" if value else "false"
                else:
                    value_text = str(value)
                f.write(f"{key}: {value_text}\n")


def parse_scalar(value: str) -> Any:
    """Parse a command-line override value using YAML scalar rules."""
    if yaml is not None:
        return yaml.safe_load(value)
    return _parse_simple_scalar(value)


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested key such as ``optimizer.lr`` inside a config dictionary."""
    if not dotted_key:
        raise ValueError("Override key cannot be empty.")

    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = target.get(part)
        if current is None:
            current = {}
            target[part] = current
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested override through non-dict key: {part}")
        target = current

    target[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply ``key=value`` command-line overrides to a config dictionary."""
    merged = dict(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, raw_value = override.split("=", 1)
        set_by_dotted_key(merged, key.strip(), parse_scalar(raw_value.strip()))
    return merged
