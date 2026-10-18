"""Read and write the ``manifest.json`` contract for generated partitions.

Core metadata uses required top-level keys. Shared dataset metadata stays at
the partition level; sizes, label distributions, and client-specific metadata
stay under ``clients``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

MANIFEST_NAME = "manifest.json"

#: Keys every partition carries. Absence means the partition is malformed.
REQUIRED = (
    "dataset",
    "partition_id",
    "schema_version",
    "settings",
    "num_clients",
    "clients",
    "splits",
    "input_spec",
    "target_spec",
    "storage",
    "has_groups",
    "group_unit",
    "provenance",
)
def write(part_dir: Path, value: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED if key not in value]
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    part_dir.mkdir(parents=True, exist_ok=True)
    with open(part_dir / MANIFEST_NAME, "w") as file:
        file.write(_format_json(value) + "\n")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_json(value: Any, level: int = 0) -> str:
    """Pretty JSON with scalar rows kept compact."""
    prefix = "  " * level
    child = "  " * (level + 1)
    if isinstance(value, dict):
        if not value or (len(value) <= 3 and all(_is_scalar(v) for v in value.values())):
            return json.dumps(value, allow_nan=False, default=str)
        entries = [
            f"{child}{json.dumps(str(key))}: {_format_json(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{prefix}}}"
    if isinstance(value, (list, tuple)):
        if all(_is_scalar(item) for item in value):
            return json.dumps(value, allow_nan=False, default=str)
        entries = [f"{child}{_format_json(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(entries) + f"\n{prefix}]"
    return json.dumps(value, allow_nan=False, default=str)


def read(part_dir: Path) -> Dict[str, Any]:
    path = part_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {part_dir}. A partition directory without one "
            "is not a partition."
        )
    with open(path) as file:
        value = json.load(file)
    missing = [key for key in REQUIRED if key not in value]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    return value


def input_names(value: Dict[str, Any]) -> List[str]:
    return [spec["name"] for spec in value["input_spec"]]


def is_multi_input(value: Dict[str, Any]) -> bool:
    """Return whether the partition declares more than one input field."""
    return len(value["input_spec"]) > 1
