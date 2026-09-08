"""Shared NPZ and NPY/memmap storage backends.

Dataset modules select a backend; the core owns the on-disk layout and loading.

Two backends:

``npz``
    One compressed archive per client. For data that fits in memory.

``memmap``
    One ``.npy`` per field, read back with ``mmap_mode="r"``, for data that does
    not. ``np.memmap`` subclasses ``ndarray``.
"""

from __future__ import annotations

from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .contract import Split, input_fields

NPZ = "npz"
MEMMAP = "memmap"
BACKENDS = (NPZ, MEMMAP)

_GROUPS = "__groups__"
_Y = "__y__"


@lru_cache(maxsize=None)
def inputs_type(dataset: str, names: Tuple[str, ...]):
    """Return the cached namedtuple type for a multi-input dataset."""
    return namedtuple(f"{dataset}Inputs", names)


def _assemble(dataset: str, names: Sequence[str], arrays: Sequence[np.ndarray]) -> Any:
    if len(names) == 1:
        return arrays[0]
    return inputs_type(dataset, tuple(names))(*arrays)


# ── write ─────────────────────────────────────────────────────────────────────

def write(backend: str, split_dir: Path, cid: int, split: Split) -> None:
    X, y, groups = split
    split_dir.mkdir(parents=True, exist_ok=True)

    if backend == NPZ:
        payload = {name: np.asarray(arr) for name, arr in input_fields(X)}
        payload[_Y] = np.asarray(y)
        if groups is not None:
            payload[_GROUPS] = np.asarray(groups)
        with open(split_dir / f"{cid}.npz", "wb") as f:
            np.savez_compressed(f, **payload)

    elif backend == MEMMAP:
        for name, arr in input_fields(X):
            np.save(split_dir / f"{cid}_{name}.npy", np.asarray(arr))
        np.save(split_dir / f"{cid}_y.npy", np.asarray(y))
        if groups is not None:
            np.save(split_dir / f"{cid}_groups.npy", np.asarray(groups))

    else:
        raise ValueError(f"Unknown storage backend {backend!r}. Use one of {BACKENDS}.")


# ── read ──────────────────────────────────────────────────────────────────────

def read(
    backend: str,
    split_dir: Path,
    cid: int,
    dataset: str,
    input_names: Sequence[str],
) -> Split:
    if backend == NPZ:
        path = split_dir / f"{cid}.npz"
        _require(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = [z[name] for name in input_names]
            y = z[_Y]
            groups = z[_GROUPS] if _GROUPS in z.files else None

    elif backend == MEMMAP:
        arrays = []
        for name in input_names:
            path = split_dir / f"{cid}_{name}.npy"
            _require(path)
            # Keep large input arrays on disk.
            arrays.append(np.load(path, mmap_mode="r"))
        y = np.load(_require(split_dir / f"{cid}_y.npy"))
        gpath = split_dir / f"{cid}_groups.npy"
        groups = np.load(gpath) if gpath.is_file() else None

    else:
        raise ValueError(f"Unknown storage backend {backend!r}. Use one of {BACKENDS}.")

    return _assemble(dataset, input_names, arrays), y, groups


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing partition file: {path}")
    return path
