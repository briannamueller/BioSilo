"""BraTS/FeTS 2022 slice-level tumour classification by institution.

Each sample is a float32 ``(4, H, W)`` axial slice in T1, T1Gd, T2, and
T2-FLAIR order, optionally area-resampled to ``image_size``. Labels indicate
whether the segmentation contains tumour on the slice. Train/test splits keep
subjects intact. Generation streams subjects into disk-backed arrays because
individual institutions can exceed available RAM.
"""

from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from ...core.contract import ClientData, hashed_unless_default, unhashed
from . import _volumes

NAME = "BraTS"
STORAGE = "memmap"      # far too large to hold in memory
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Params:
    # Fixed release marker; this adapter supports FeTS 2022 only.
    source: str = field(default="FeTS2022", init=False)
    slice_stride: int = 2
    min_brain_fraction: float = 0.01
    min_subjects_per_institution: int = 5
    min_slices_per_institution: int = 100
    max_subjects_per_institution: int = hashed_unless_default(0)  # 0 = no cap
    image_size: int = hashed_unless_default(0)  # 0 = native resolution
    stratify: bool = hashed_unless_default(False)  # spread test subjects by tumour share
    num_clients: int = 0            # 0 = every qualifying institution
    train_ratio: float = 0.80
    seed: int = 1

    source_dir: str = unhashed("")  # location, not identity


def label(p: Params) -> str:
    bits = [f"s{p.seed}"]
    if p.num_clients:
        bits.append(f"n{p.num_clients}")
    return "_".join(bits)


def build(p: Params) -> Iterator[ClientData]:
    root = Path(p.source_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"source_dir not found: {root}. Expected the extracted FeTS training "
            "data, containing subject directories and a partitioning CSV."
        )

    institutions = _scan(root, p)
    if not institutions:
        raise RuntimeError(
            "No institution qualified "
            f"(min_subjects={p.min_subjects_per_institution}, "
            f"min_slices={p.min_slices_per_institution})."
        )

    order = sorted(institutions)
    if p.num_clients > len(order):
        noun = "client" if len(order) == 1 else "clients"
        print(
            f"[BioSilo] Only {len(order)} {noun} survived the filters; "
            f"requested {p.num_clients}."
        )
    if p.num_clients:
        order = order[:p.num_clients]

    rng = np.random.default_rng(p.seed)
    workspace = Path(tempfile.mkdtemp(prefix="biosilo-brats-"))

    try:
        for institution in order:
            subjects = institutions[institution]
            cap = p.max_subjects_per_institution
            if cap and len(subjects) > cap:
                keep = np.sort(rng.choice(len(subjects), cap, replace=False))
                subjects = [subjects[i] for i in keep]
            train, test = _split_subjects(subjects, p.train_ratio, rng, p.stratify)

            yield ClientData(
                client_id=str(institution),
                train=_materialize(workspace, "train", institution, train,
                                   p.image_size),
                test=_materialize(workspace, "test", institution, test,
                                  p.image_size),
                meta={
                    "n_subjects": len(subjects),
                    "n_train_subjects": len(train),
                    "n_test_subjects": len(test),
                },
                group_unit="subject",
            )
            # The core writes each yielded client before requesting the next one.
            for stale in workspace.glob(f"{institution}_*"):
                stale.unlink()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# Internal helpers

#: One subject's contribution: directory, surviving slice indices, plane shape.
Subject = Tuple[Path, List[int], Tuple[int, int]]


def _scan(root: Path, p: Params) -> Dict[int, List[Subject]]:
    """Collect usable subjects and slice indices by institution.

    The T1-only scan determines output sizes before four-channel materialization.
    Subjects must match the cohort's dominant in-plane geometry in addition to
    having internally consistent modalities.
    """
    mapping = _volumes.read_institutions(root)

    found: Dict[int, List[Subject]] = defaultdict(list)
    for subject_dir in _volumes.find_subject_dirs(root):
        institution = mapping.get(subject_dir.name)
        if institution is None:
            for key, value in mapping.items():
                if key in subject_dir.name or subject_dir.name in key:
                    institution = value
                    break
        if institution is None:
            continue

        indices = _volumes.kept_slice_indices(
            subject_dir, p.slice_stride, p.min_brain_fraction)
        if indices:
            found[institution].append(
                (subject_dir, indices, _plane_shape(subject_dir)))

    shape = _dominant_shape(found)
    conforming: Dict[int, List[Subject]] = {
        institution: [s for s in subjects if s[2] == shape]
        for institution, subjects in found.items()
    }

    return {
        institution: subjects
        for institution, subjects in conforming.items()
        if len(subjects) >= p.min_subjects_per_institution
        and sum(len(idx) for _, idx, _ in subjects) >= p.min_slices_per_institution
    }


def _dominant_shape(found: Dict[int, List[Subject]]) -> Tuple[int, int]:
    """The in-plane geometry the cohort is built on, i.e. the most common one."""
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for subjects in found.values():
        for _, indices, shape in subjects:
            counts[shape] += len(indices)
    if not counts:
        raise RuntimeError("No subject produced any usable slice.")
    return max(counts, key=counts.get)


def _split_subjects(subjects: List[Subject], train_ratio: float, rng,
                    stratify: bool = False):
    """Split a client by subject.

    With ``stratify``, subjects are ranked by their share of tumour slices and
    test subjects are drawn at evenly spaced ranks, so both splits cover the
    same range of tumour shares.
    """
    n = len(subjects)
    n_train = max(1, int(n * train_ratio))
    if n_train == n and n > 1:
        n_train = n - 1
    if not stratify:
        order = rng.permutation(n)
        return ([subjects[i] for i in order[:n_train]],
                [subjects[i] for i in order[n_train:]])

    shares = [_volumes.slice_labels(d, idx).mean() for d, idx, _ in subjects]
    ranked = np.argsort(shares, kind="stable")
    n_test = n - n_train
    picks = np.floor((np.arange(n_test) + rng.random()) * n / n_test).astype(int)
    test = set(ranked[picks].tolist())
    return ([s for i, s in enumerate(subjects) if i not in test],
            [s for i, s in enumerate(subjects) if i in test])


def _materialize(workspace: Path, split: str, institution: int,
                 subjects: List[Subject], image_size: int = 0):
    """Build one split as a disk-backed array, one subject at a time."""
    total = sum(len(indices) for _, indices, _ in subjects)
    if total == 0:
        raise RuntimeError(
            f"institution {institution} {split}: no slices. Loosen "
            "min_brain_fraction or lower train_ratio."
        )

    height, width = subjects[0][2]
    resize = None
    if image_size:
        resize = (_area_weights(height, image_size),
                  _area_weights(width, image_size))
        height = width = image_size
    path = workspace / f"{institution}_{split}.npy"

    x = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(total, 4, height, width))
    y = np.empty(total, dtype=np.int64)
    groups = np.empty(total, dtype=object)

    at = 0
    for subject_dir, indices, _ in subjects:
        block, labels = _volumes.slices_and_labels(subject_dir, indices)
        stop = at + len(indices)
        if resize is not None:
            block = resize[0] @ block @ resize[1].T
        x[at:stop] = block
        y[at:stop] = labels
        groups[at:stop] = subject_dir.name
        at = stop
        del block

    x.flush()
    return x, y, groups.astype("<U64")


def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """``(n_out, n_in)`` matrix averaging each output pixel's input footprint."""
    edges = np.linspace(0, n_in, n_out + 1)
    lo = np.maximum(edges[:-1, None], np.arange(n_in)[None, :])
    hi = np.minimum(edges[1:, None], np.arange(1, n_in + 1)[None, :])
    overlap = np.clip(hi - lo, 0, None)
    return (overlap / overlap.sum(axis=1, keepdims=True)).astype(np.float32)


def _plane_shape(subject_dir: Path) -> Tuple[int, int]:
    import nibabel as nib

    files = _volumes.find_modalities(subject_dir)
    shape = nib.load(str(files["t1"])).shape
    return int(shape[0]), int(shape[1])
