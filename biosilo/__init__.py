"""Naturally partitioned biomedical datasets for federated learning.

Client boundaries follow source institutions, studies, or hospitals. BioSilo
provides explicit source downloads when available, partition generation, and
loading, but no models or training strategies.

    import biosilo

    p = biosilo.load("synthetic")                 # metadata; holds no data
    X, y, groups = p.client(0, "train")

    X, y, groups = biosilo.load_partition("synthetic", 0, "train")

Generated partitions are stored under ``data/`` by default. Set
``BIOSILO_DATA_ROOT`` or pass ``root=`` to use another location.
"""

from __future__ import annotations

__version__ = "0.1.1"

from .core.contract import ClientData
from .core.download import download
from .core.generate import expected_partition, generate
from .core.load import Partition, load, load_partition
from .core.registry import available

__all__ = [
    "available",
    "download",
    "expected_partition",
    "generate",
    "load",
    "load_partition",
    "Partition",
    "ClientData",
    "__version__",
]
