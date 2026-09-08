"""Shared types and helpers for dataset modules and the core.

A dataset module provides this core interface:

    NAME     : str                          canonical dataset name
    STORAGE  : str                          which storage backend its arrays want
    SCHEMA_VERSION : int                    generated-data identity version
    Params   : dataclass                    everything that changes the output
    label(p) -> str                         human half of the directory name
    build(p) -> Iterator[ClientData]        the entire pipeline, module's business

``build()`` must not initiate network acquisition. It may read locally prepared
source data or generate built-in test data. A module may additionally expose
``download(p)`` for explicit source acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterator, Optional, Protocol, Tuple, Union

import numpy as np

SPLITS = ("train", "test")

#: One split's payload. ``X`` is a bare ndarray for single-input datasets, or a
#: namedtuple of arrays for multi-input ones. ``groups`` is None when samples
#: are independent.
Split = Tuple[Any, np.ndarray, Optional[np.ndarray]]


@dataclass
class ClientData:
    """Train and test data for one client yielded by ``build()``."""

    client_id: str
    train: Split
    test: Split
    meta: Dict[str, Any] = field(default_factory=dict)


class DatasetModule(Protocol):
    """Structural type for a dataset module. Modules are plain modules."""

    NAME: str
    STORAGE: str
    SCHEMA_VERSION: int

    def label(self, params: Any) -> str: ...
    def build(self, params: Any) -> Iterator[ClientData]: ...


# ── Parameters ────────────────────────────────────────────────────────────────

def unhashed(default: Any = None, **kw):
    """Declare a Params field, such as a local path, excluded from identity."""
    meta = dict(kw.pop("metadata", {}))
    meta["hashed"] = False
    return field(default=default, metadata=meta, **kw)


def hashable_params(params: Any) -> Dict[str, Any]:
    """The subset of a Params object that defines the output."""
    if not is_dataclass(params):
        raise TypeError(f"Params must be a dataclass, got {type(params).__name__}")
    return {
        f.name: getattr(params, f.name)
        for f in fields(params)
        if f.metadata.get("hashed", True)
    }


# ── Input description ─────────────────────────────────────────────────────────

def describe_inputs(X: Any) -> list[dict]:
    """Describe input names, per-sample shapes, and dtypes.

    Single inputs use the name ``x``. Namedtuple fields define multi-input names
    and order. Reported shapes exclude the sample axis.
    """
    if _is_named_tuple(X):
        return [
            {
                "name": name,
                "shape": list(np.asarray(getattr(X, name)).shape[1:]),
                "dtype": str(np.asarray(getattr(X, name)).dtype),
            }
            for name in X._fields
        ]
    arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    return [{"name": "x", "shape": list(arr.shape[1:]), "dtype": str(arr.dtype)}]


def input_fields(X: Any) -> list[Tuple[str, np.ndarray]]:
    """(name, array) pairs for X, single- or multi-input."""
    if _is_named_tuple(X):
        return [(n, getattr(X, n)) for n in X._fields]
    return [("x", X)]


def _is_named_tuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def n_samples(X: Any) -> int:
    """Sample count. First axis is the sample axis for every field."""
    if _is_named_tuple(X):
        return len(getattr(X, X._fields[0]))
    return len(X)
