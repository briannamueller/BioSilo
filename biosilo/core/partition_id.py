"""Deterministic partition identifiers.

A partition directory is named ``<label>_<hash>``. The label belongs to the
dataset module, and helps a human scanning a directory listing; the hash is the
spine's, and guarantees uniqueness.

``manifest.json`` is authoritative; directory names are not parsed for metadata.
Identifiers contain only ``[A-Za-z0-9._-]``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict

#: Maximum length of the human-readable label component.
MAX_LABEL = 60

HASH_CHARS = 10

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(label: str) -> str:
    """Reduce a label to a shell-safe, length-capped fragment."""
    safe = _UNSAFE.sub("-", label).strip("-._")
    return safe[:MAX_LABEL].rstrip("-._")


def compute_hash(
    dataset: str, params: Dict[str, Any], schema_version: int,
) -> str:
    """Hash dataset identity, schema version, and output-affecting parameters."""
    _check_schema_version(schema_version)
    blob = json.dumps(
        {
            "dataset": dataset,
            "schema_version": schema_version,
            "params": params,
        },
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:HASH_CHARS]


def compose(
    dataset: str, label: str, params: Dict[str, Any], schema_version: int,
) -> str:
    """Full partition id."""
    safe = sanitize(label)
    digest = compute_hash(dataset, params, schema_version)
    return f"{safe}_{digest}" if safe else digest


def verify(
    partition_id: str, dataset: str, params: Dict[str, Any], schema_version: int,
) -> bool:
    """Return whether an ID has the digest expected for the given parameters."""
    return partition_id.endswith(compute_hash(dataset, params, schema_version))


def _check_schema_version(schema_version: int) -> None:
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("SCHEMA_VERSION must be a positive integer")
    if schema_version < 1:
        raise ValueError("SCHEMA_VERSION must be a positive integer")
