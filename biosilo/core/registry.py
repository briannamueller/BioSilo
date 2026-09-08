"""Explicit registry and lazy import resolution for dataset modules.

Module paths are stored as strings so importing BioSilo does not load optional
dataset dependencies. Canonical name resolution also works without importing a
dataset module.
"""

from __future__ import annotations

from importlib import import_module

from .contract import DatasetModule

#: Canonical name -> the module implementing it.
_MODULES: dict[str, str] = {
    "synthetic": "biosilo.datasets.synthetic",
    "synthetic_memmap": "biosilo.datasets.synthetic_memmap",
    "eICU": "biosilo.datasets.eicu",
    "scRNAseq": "biosilo.datasets.scrnaseq",
    "BraTS": "biosilo.datasets.brats",
    "TCGA": "biosilo.datasets.tcga",
}

_LOOKUP = {name.lower(): name for name in _MODULES}


def available() -> tuple:
    return tuple(_MODULES)


def canonical(dataset: str) -> str:
    """Return a dataset's canonical spelling using case-insensitive matching."""
    name = _LOOKUP.get(dataset.lower())
    if name is None:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Available: {', '.join(available())}"
        )
    return name


def get(dataset: str) -> DatasetModule:
    """The dataset's module, imported on first use."""
    name = canonical(dataset)
    module = import_module(_MODULES[name])
    if getattr(module, "NAME", None) != name:
        raise RuntimeError(
            f"{_MODULES[name]} declares NAME={getattr(module, 'NAME', None)!r} but "
            f"is registered as {name!r}; the table and the module disagree.")
    schema_version = getattr(module, "SCHEMA_VERSION", None)
    if (isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 1):
        raise RuntimeError(
            f"{_MODULES[name]} must declare SCHEMA_VERSION as a positive integer.")
    return module
