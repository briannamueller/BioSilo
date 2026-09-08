"""Explicit acquisition of source data for datasets that provide it."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from . import registry
from .generate import _resolve_dataset_paths, _resolve_root


def download(
    dataset: str,
    params: Any = None,
    root: Optional[os.PathLike] = None,
    **param_kwargs,
) -> Path:
    """Download or prepare one dataset's source files.

    Acquisition is explicit and separate from partition generation. Dataset
    modules without an acquisition helper direct users to their dataset guide.
    """
    module = registry.get(dataset)
    handler = getattr(module, "download", None)
    if not callable(handler):
        raise NotImplementedError(
            f"{module.NAME} has no automatic download. Follow its guide under "
            "docs/datasets/ to obtain the required source data."
        )

    if params is None:
        params = module.Params(**param_kwargs)
    elif param_kwargs:
        raise TypeError("pass either a Params object or keyword arguments, not both")

    resolved_root = _resolve_root(root)
    params = _resolve_dataset_paths(module.NAME, params, resolved_root)
    return Path(handler(params))
