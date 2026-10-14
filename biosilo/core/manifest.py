"""Read and write the ``manifest.json`` contract for generated partitions.

Core metadata uses required top-level keys. Per-client sizes, label
distributions, and dataset-specific metadata are stored together under
``clients``.
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
        json.dump(value, file, indent=2, sort_keys=True, default=str)
        file.write("\n")


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
