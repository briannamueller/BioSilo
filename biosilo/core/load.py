"""Load partition metadata and client arrays.

Two public entry points are available:

* ``load_partition(dataset, cid, split)`` returns a plain ``(X, y, groups)``
  tuple.
* ``load(dataset)`` returns a metadata-only :class:`Partition` handle.

``Partition`` does not cache arrays, preserving lazy access to memmapped data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import manifest, storage
from .contract import SPLITS, Split

ENV_ROOT = "BIOSILO_DATA_ROOT"
DEFAULT_DATA_ROOT = "data"


def data_root(root: Optional[os.PathLike] = None) -> Path:
    """Resolve an explicit root, ``BIOSILO_DATA_ROOT``, or ``data/``."""
    if root is not None:
        resolved = Path(root)
    else:
        env = os.environ.get(ENV_ROOT)
        resolved = Path(env) if env else Path(DEFAULT_DATA_ROOT)
    return resolved


def _canonical_or_given(dataset: str) -> str:
    """Return canonical spelling for known datasets, otherwise the input name."""
    from . import registry
    try:
        return registry.canonical(dataset)
    except KeyError:
        return dataset


def partition_dir(
    dataset: str,
    root: Optional[os.PathLike] = None,
    partition: Optional[str] = None,
) -> Path:
    """Resolve one partition directory.

    Without ``partition``, resolution succeeds only when one generated
    partition exists; otherwise the candidates are reported.
    """
    base = data_root(root) / _canonical_or_given(dataset)
    if not base.is_dir():
        raise FileNotFoundError(f"No such dataset directory: {base}")

    if partition is not None:
        chosen = base / partition
        if not (chosen / manifest.MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"No {manifest.MANIFEST_NAME} in {chosen}")
        return chosen

    candidates = sorted(
        p for p in base.iterdir() if (p / manifest.MANIFEST_NAME).is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No generated partitions under {base}.")
    if len(candidates) > 1:
        listing = "\n  ".join(p.name for p in candidates)
        raise ValueError(
            f"{len(candidates)} partitions exist for {dataset}; pass "
            f"partition=... to choose one:\n  {listing}"
        )
    return candidates[0]


def check_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(
            f"split must be one of {SPLITS}, got {split!r}. BioSilo emits no "
            "validation split; carve it from train."
        )
    return split


class Partition:
    """Metadata and lazy client access for one generated partition."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = manifest.read(self.path)

    # ── metadata ──────────────────────────────────────────────────────────
    @property
    def dataset(self) -> str:
        return self.manifest["dataset"]

    @property
    def partition_id(self) -> str:
        return self.manifest["partition_id"]

    @property
    def schema_version(self) -> int:
        return self.manifest["schema_version"]

    @property
    def num_clients(self) -> int:
        return self.manifest["num_clients"]

    @property
    def num_classes(self) -> int:
        return self.manifest["target_spec"]["num_classes"]

    @property
    def client_ids(self) -> List[str]:
        """The natural units' own identifiers: hospital, study, institution."""
        return [client["client_id"] for client in self.manifest["clients"]]

    @property
    def inputs(self) -> List[dict]:
        """Per-sample shape and dtype of each input, in declared order."""
        return self.manifest["input_spec"]

    @property
    def is_multi_input(self) -> bool:
        return manifest.is_multi_input(self.manifest)

    @property
    def has_groups(self) -> bool:
        return self.manifest["has_groups"]

    @property
    def group_unit(self) -> Optional[str]:
        return self.manifest["group_unit"]

    @property
    def settings(self) -> Dict[str, Any]:
        return self.manifest["settings"]

    @property
    def provenance(self) -> Dict[str, Any]:
        return self.manifest["provenance"]

    # ── data ──────────────────────────────────────────────────────────────
    def client(self, cid: int, split: str) -> Split:
        """``(X, y, groups)`` for one client. Fresh handles every call."""
        return storage.read(
            self.manifest["storage"],
            self.path / check_split(split),
            cid,
            self.dataset,
            manifest.input_names(self.manifest),
        )

    def summary(self) -> dict:
        return {
            "dataset": self.dataset,
            "partition_id": self.partition_id,
            "schema_version": self.schema_version,
            "num_clients": self.num_clients,
            "num_classes": self.num_classes,
            "client_ids": self.client_ids,
            "input_spec": self.inputs,
            "has_groups": self.has_groups,
            "storage": self.manifest["storage"],
        }

    def __repr__(self) -> str:
        return (
            f"Partition({self.dataset}, clients={self.num_clients}, "
            f"classes={self.num_classes}, groups={self.has_groups})"
        )


def load(
    dataset: str,
    root: Optional[os.PathLike] = None,
    partition: Optional[str] = None,
) -> Partition:
    return Partition(partition_dir(dataset, root, partition))


def load_partition(
    dataset: str,
    cid: int,
    split: str,
    root: Optional[os.PathLike] = None,
    partition: Optional[str] = None,
) -> Split:
    return load(dataset, root, partition).client(cid, split)
