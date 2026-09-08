"""Small generated dataset used to test the shared data pipeline.

It supports single or multiple inputs and optional sample groups. The values
have no scientific meaning.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..core.contract import ClientData

NAME = "synthetic"
STORAGE = "npz"
SCHEMA_VERSION = 1

#: Named multi-input fields in storage order.
Inputs = namedtuple("syntheticInputs", ("seq", "flat"))


@dataclass(frozen=True)
class Params:
    n_clients: int = 3
    n_per_client: int = 24
    n_classes: int = 3
    n_features: int = 5
    n_inputs: int = 1          # 1 -> bare ndarray, 2 -> namedtuple
    with_groups: bool = False
    group_size: int = 4        # samples per group, when grouped
    train_ratio: float = 0.75
    seed: int = 0


def label(p: Params) -> str:
    return f"n{p.n_clients}_s{p.seed}"


def build(p: Params) -> Iterator[ClientData]:
    rng = np.random.default_rng(p.seed)

    for c in range(p.n_clients):
        n = p.n_per_client
        y = rng.integers(0, p.n_classes, n).astype(np.int64)

        if p.n_inputs == 1:
            X = rng.random((n, p.n_features), dtype=np.float32)
        else:
            X = Inputs(
                seq=rng.random((n, 8, p.n_features), dtype=np.float32),
                flat=rng.random((n, 4), dtype=np.float32),
            )

        groups = (
            np.repeat(np.arange(int(np.ceil(n / p.group_size))), p.group_size)[:n]
            if p.with_groups
            else None
        )

        train_idx, test_idx = _split(n, groups, p.train_ratio, rng)

        yield ClientData(
            client_id=f"site-{c}",
            train=(_take(X, train_idx), y[train_idx], _take_1d(groups, train_idx)),
            test=(_take(X, test_idx), y[test_idx], _take_1d(groups, test_idx)),
            meta={"group_unit": "block" if p.with_groups else None},
        )


def _split(n, groups, ratio, rng):
    """Hold out whole groups when grouped; plain rows otherwise."""
    if groups is None:
        order = rng.permutation(n)
        cut = max(1, int(n * ratio))
        return order[:cut], order[cut:]

    unique = np.unique(groups)
    order = rng.permutation(unique)
    cut = max(1, int(len(unique) * ratio))
    train_groups = set(order[:cut].tolist())
    mask = np.array([g in train_groups for g in groups])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def _take(X, idx):
    if isinstance(X, tuple) and hasattr(X, "_fields"):
        return type(X)(*(getattr(X, f)[idx] for f in X._fields))
    return X[idx]


def _take_1d(arr, idx):
    return None if arr is None else arr[idx]
