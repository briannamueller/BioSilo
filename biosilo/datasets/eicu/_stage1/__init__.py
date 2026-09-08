"""Build and cache the eICU raw-table feature store.

Stage one builds a dense patient-hour table, assigns people to train/test once,
fits statistical preprocessing on training-visible data, and applies the frozen
transform to both partitions. Stores are task-horizon and split specific so a
test patient or a future hour cannot affect fitted preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

#: Written after all feature-store stages complete successfully.
MARKER = ".stage1-complete"

#: Horizon-independent inputs required by fitted stage-one preprocessing.
RAW_ARTIFACTS = (
    "binned_timeseries.parquet",
    "diagnoses.csv",
    "flat_features.csv",
    "labels.csv",
    "stays.txt",
    "timeseries_feature_manifest.json",
)


@dataclass(frozen=True)
class Stage1Params:
    """Everything that changes the feature store's contents."""

    schema_version: int = 3
    preprocessing_protocol: str = "inductive"
    observation_hours: int = 24
    diagnosis_hours: int = 5
    train_ratio: float = 0.75
    split_seed: int = 1
    min_dx_prevalence: float = 0.01
    within_prev: float = 0.25
    cross_prev: float = 0.70
    mask_mode: str = "exponential_decay"
    decay_rate: float = 4.0 / 3.0

    def identity(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


def store_path(processed_root: Path, params: Stage1Params) -> Path:
    return Path(processed_root) / f"stage1_{params.identity()}"


def is_complete(store: Path) -> bool:
    return (store / MARKER).is_file()


def uses_inductive_split(store: Path) -> bool:
    """Whether fitted preprocessing used only assigned training people."""
    try:
        metadata = json.loads((Path(store) / MARKER).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (metadata.get('preprocessing_protocol') == 'inductive'
            and int(metadata.get('schema_version', 0)) >= 1)


def looks_like_store(path: Path) -> bool:
    """Whether a directory is already a usable feature store.

    Recognises the layout stage 1 produces, wherever it came from, including a
    store produced outside BioSilo.
    """
    return any((Path(path) / split / "stays.txt").is_file()
               for split in ("train", "val", "test"))


def ensure(
    processed_root: Path,
    eicu_dir: Path,
    params: Optional[Stage1Params] = None,
    force: bool = False,
) -> Tuple[Path, str]:
    """Return a reusable feature store and its provenance.

    Existing stores supplied directly use ``"external"`` provenance because
    their preprocessing parameters cannot be verified. Parameter-keyed stores
    use ``"cached"`` or ``"built"`` provenance. Stores without a completion
    marker are rebuilt.
    """
    params = params or Stage1Params()
    processed_root = Path(processed_root)

    if not force and looks_like_store(processed_root):
        return processed_root, "external"

    store = store_path(processed_root, params)
    if is_complete(store) and not force:
        return store, "cached"

    staged = store.with_name(f'.{store.name}.building-{uuid4().hex}')
    staged.mkdir(parents=True)

    from . import (apply_filters, diagnoses, extract_tables, flat_and_labels,
                   split_train_test, timeseries)

    data_dir = f"{staged}/"
    eicu_dir = str(eicu_dir)

    try:
        raw_seed = _find_raw_seed(
            processed_root, exclude=store,
            schema_version=params.schema_version)
        if raw_seed is None:
            extract_tables.main(eicu_dir, str(staged))
            timeseries.timeseries_main(data_dir, test=False)
        else:
            print(f'==> Reusing horizon-independent eICU inputs from {raw_seed}')
            _copy_raw_artifacts(raw_seed, staged)
        partitions = split_train_test.create_person_partitions(
            data_dir, train_ratio=params.train_ratio, seed=params.split_seed)
        apply_filters.apply_filters_main(
            data_dir=data_dir,
            eicu_dir=eicu_dir,
            within_threshold=params.within_prev,
            cross_threshold=params.cross_prev,
            mask_mode=params.mask_mode,
            decay_rate=params.decay_rate,
            fit_patients=partitions['train'],
            fit_max_hour=params.observation_hours,
        )
        diagnoses.diagnoses_main(
            data_dir, params.min_dx_prevalence,
            fit_patients=partitions['train'],
            max_offset_minutes=params.diagnosis_hours * 60)
        flat_and_labels.flat_and_labels_main(
            data_dir, fit_patients=partitions['train'])
        split_train_test.split_train_test(
            data_dir, is_test=False, partitions=partitions)

        (staged / MARKER).write_text(
            json.dumps(asdict(params), indent=2, sort_keys=True) + "\n")
        _install_staged_store(staged, store)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    return store, "built"


def _find_raw_seed(
    processed_root: Path,
    exclude: Optional[Path] = None,
    schema_version: int = 3,
):
    """Find a complete parameter-keyed store with reusable raw-stage artifacts."""
    for marker in sorted(Path(processed_root).glob(f"stage1_*/{MARKER}")):
        candidate = marker.parent
        if exclude is not None and candidate == Path(exclude):
            continue
        try:
            metadata = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("schema_version") != schema_version:
            continue
        if all((candidate / name).is_file()
               and (candidate / name).stat().st_size > 0
               for name in RAW_ARTIFACTS):
            return candidate
    return None


def _copy_raw_artifacts(source: Path, destination: Path) -> None:
    """Copy immutable raw-stage inputs into an isolated horizon build."""
    source, destination = Path(source), Path(destination)
    for name in RAW_ARTIFACTS:
        shutil.copy2(source / name, destination / name)


def _install_staged_store(staged: Path, store: Path, replace=os.replace) -> None:
    """Install a complete sibling store, restoring the prior store on failure."""
    backup = store.with_name(f'.{store.name}.backup-{uuid4().hex}')
    had_prior = store.exists()
    if had_prior:
        replace(store, backup)
    try:
        replace(staged, store)
    except Exception:
        if had_prior and backup.exists():
            replace(backup, store)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
