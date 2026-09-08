"""Atomic file installation for downloaded and derived cache artifacts."""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


def atomic_write(destination, writer, validator=None):
    """Write and validate a sibling temporary file, then atomically install it."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.building-{uuid4().hex}.tmp")
    try:
        writer(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"cache writer produced no data for {destination.name}")
        if validator is not None:
            validator(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

