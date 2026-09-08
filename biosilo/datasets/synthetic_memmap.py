"""Synthetic dataset variant stored with the memmap backend."""

from __future__ import annotations

from .synthetic import Params, build, label  # noqa: F401  (re-exported)

NAME = "synthetic_memmap"
STORAGE = "memmap"
SCHEMA_VERSION = 1
