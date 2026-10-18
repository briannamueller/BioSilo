"""Generate and store a dataset partition.

The dataset-independent sequence is:

    resolve module -> hash params -> ask module for a label -> for each client
    the module yields: validate, write -> write manifest.json

Dataset-specific work is implemented by the module's ``build()`` iterator.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import numpy as np

from . import manifest, partition_id, registry, storage, validate
from .contract import describe_inputs, hashable_params, n_samples
from .load import data_root


def generate(
    dataset: str,
    params: Any = None,
    root: Optional[os.PathLike] = None,
    overwrite: bool = False,
    **param_kwargs,
) -> Path:
    """Build a partition and return its directory.

    The same parameters resolve to the same directory. An existing matching
    partition is reused unless ``overwrite=True``.
    """
    module, params, hashed, part_dir = _resolve_request(
        dataset, params, root, param_kwargs
    )
    pid = part_dir.name
    dataset_dir = part_dir.parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and _is_matching_partition(
            part_dir, module.NAME, hashed, module.SCHEMA_VERSION, pid):
        return part_dir

    staged = dataset_dir / f".{pid}.building-{uuid4().hex}"
    staged.mkdir()

    clients: list = []
    inputs_spec = None
    group_unit = None
    partition_meta = None
    class_names = None
    max_label = -1
    has_groups = False

    try:
        for index, client in enumerate(module.build(params)):
            validate.check_client(client)

            spec = describe_inputs(client.train[0])
            grouped = client.train[2] is not None

            if inputs_spec is None:
                inputs_spec, has_groups = spec, grouped
                group_unit = client.group_unit
                partition_meta = client.partition_meta
                class_names = client.class_names
            else:
                # A partition records one input specification shared by all clients.
                validate.check_consistent(
                    spec, inputs_spec, grouped, has_groups,
                    client_id=client.client_id)
                if client.group_unit != group_unit:
                    raise ValueError(
                        "clients must share one group_unit; "
                        f"{client.client_id!r} has {client.group_unit!r}, "
                        f"expected {group_unit!r}"
                    )
                if client.partition_meta != partition_meta:
                    raise ValueError(
                        "clients must share one partition_meta mapping; "
                        f"{client.client_id!r} differs"
                    )
                if client.class_names != class_names:
                    raise ValueError(
                        "clients must share one class_names sequence; "
                        f"{client.client_id!r} differs"
                    )

            client_manifest = {
                "client_id": str(client.client_id),
                "sizes": {},
                "label_counts": {},
            }
            for split_name in ("train", "test"):
                split = getattr(client, split_name)
                storage.write(module.STORAGE, staged / split_name, index, split)
                client_manifest["sizes"][split_name] = int(n_samples(split[0]))
                client_manifest["label_counts"][split_name] = Counter(
                    np.asarray(split[1]).tolist()
                )
            if client.meta:
                client_manifest["metadata"] = client.meta

            labels = np.concatenate([
                np.asarray(client.train[1]), np.asarray(client.test[1])])
            if labels.size:
                max_label = max(max_label, int(labels.max()))
            clients.append(client_manifest)

        if not clients:
            raise ValueError(f"{module.NAME}: build() yielded no clients.")

        num_classes = max_label + 1
        for client_manifest in clients:
            for split_name in ("train", "test"):
                counts = client_manifest["label_counts"][split_name]
                client_manifest["label_counts"][split_name] = [
                    int(counts.get(label, 0)) for label in range(num_classes)
                ]

        target_spec = {
            "dtype": "int64",
            "shape": [],
            "num_classes": num_classes,
        }
        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"class_names has {len(class_names)} entries for "
                    f"{num_classes} classes"
                )
            target_spec["class_names"] = class_names

        partition_manifest = {
            "dataset": module.NAME,
            "partition_id": pid,
            "schema_version": module.SCHEMA_VERSION,
            "settings": hashed,
            "num_clients": len(clients),
            "clients": clients,
            "splits": ["train", "test"],
            "input_spec": inputs_spec,
            "target_spec": target_spec,
            "storage": module.STORAGE,
            "has_groups": has_groups,
            "group_unit": group_unit,
        }
        if partition_meta:
            partition_manifest["metadata"] = partition_meta
        # Informational only; identity is controlled by settings and schema version.
        partition_manifest["provenance"] = {"biosilo_version": _package_version()}
        manifest.write(staged, partition_manifest)
        _install_staged_partition(staged, part_dir)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    return part_dir


def expected_partition(
    dataset: str,
    params: Any = None,
    root: Optional[os.PathLike] = None,
    **param_kwargs,
) -> Path:
    """Return the partition path determined by a generation configuration."""
    return _resolve_request(dataset, params, root, param_kwargs)[-1]


def _resolve_request(dataset: str, params, root, param_kwargs):
    module = registry.get(dataset)
    if params is None:
        params = module.Params(**param_kwargs)
    elif param_kwargs:
        raise TypeError("pass either a Params object or keyword arguments, not both")

    resolved_root = data_root(root)
    params = _resolve_dataset_paths(module.NAME, params, resolved_root)
    hashed = hashable_params(params)
    pid = partition_id.compose(
        module.NAME, module.label(params), hashed, module.SCHEMA_VERSION
    )
    return module, params, hashed, resolved_root / module.NAME / pid


def _install_staged_partition(
    staged: Path, part_dir: Path, replace=os.replace,
) -> None:
    """Install a complete sibling directory, restoring any prior on failure."""
    backup = part_dir.with_name(f".{part_dir.name}.backup-{uuid4().hex}")
    had_prior = part_dir.exists()
    if had_prior:
        replace(part_dir, backup)
    try:
        replace(staged, part_dir)
    except Exception:
        if had_prior and backup.exists():
            replace(backup, part_dir)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _is_matching_partition(
    part_dir: Path,
    dataset: str,
    params: Dict[str, Any],
    schema_version: int,
    expected_id: str,
) -> bool:
    """Whether an existing manifest describes the identity at this path."""
    try:
        existing = manifest.read(part_dir)
    except (FileNotFoundError, ValueError, OSError):
        return False
    return (
        existing["dataset"] == dataset
        and existing["partition_id"] == expected_id
        and existing["schema_version"] == schema_version
        and existing["settings"] == params
        and partition_id.verify(
            expected_id, dataset, existing["settings"], schema_version)
    )


def _resolve_root(root: Optional[os.PathLike]) -> Path:
    """Resolve and create the generation root when necessary."""
    path = data_root(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_dataset_paths(dataset: str, params: Any, root: Path) -> Any:
    """Fill conventional source/cache locations without changing identity."""
    if not is_dataclass(params):
        return params
    names = {field.name for field in fields(params)}
    updates = {}
    if "source_dir" in names and not getattr(params, "source_dir"):
        updates["source_dir"] = str(root / "_raw" / dataset)
    if "cache_dir" in names and not getattr(params, "cache_dir"):
        updates["cache_dir"] = str(root / "_cache" / dataset)
    return replace(params, **updates) if updates else params


def _package_version() -> str:
    """The package's declared version, read at call time.

    ``biosilo/__init__`` imports this module, so a module-level import of
    ``__version__`` would silently depend on the order of that file's
    statements. Reading it here keeps one declaration of the version without
    that coupling.
    """
    from .. import __version__

    return __version__
