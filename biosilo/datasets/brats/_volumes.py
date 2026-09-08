"""Read FeTS/BraTS NIfTI volumes and produce normalized axial slices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Fixed input channel order.
CHANNELS = ("t1", "t1ce", "t2", "flair")
REQUIRED = CHANNELS + ("seg",)

SUFFIXES = {
    "t1": ("_t1.nii", "_T1.nii", "_t1.nii.gz", "_T1.nii.gz"),
    "t1ce": ("_t1ce.nii", "_T1ce.nii", "_t1gd.nii", "_T1Gd.nii",
             "_t1ce.nii.gz", "_T1ce.nii.gz", "_t1gd.nii.gz", "_T1Gd.nii.gz"),
    "t2": ("_t2.nii", "_T2.nii", "_t2.nii.gz", "_T2.nii.gz"),
    "flair": ("_flair.nii", "_FLAIR.nii", "_t2f.nii", "_T2f.nii",
              "_flair.nii.gz", "_FLAIR.nii.gz", "_t2f.nii.gz", "_T2f.nii.gz"),
    "seg": ("_seg.nii", "_Seg.nii", "_seg.nii.gz", "_Seg.nii.gz"),
}

SUBJECT_PREFIXES = ("FeTS", "BraTS", "UPENN", "Case")


def find_subject_dirs(data_root: Path) -> List[Path]:
    """Subject directories, in sorted order."""
    dirs = [e for e in sorted(data_root.iterdir())
            if e.is_dir() and any(p in e.name for p in SUBJECT_PREFIXES)]
    if dirs:
        return dirs
    for subdir in sorted(data_root.iterdir()):
        if subdir.is_dir():
            dirs.extend(e for e in sorted(subdir.iterdir())
                        if e.is_dir() and any(p in e.name for p in SUBJECT_PREFIXES))
    return dirs


def find_modalities(subject_dir: Path) -> Dict[str, Path]:
    """Locate each modality file. Missing entries mean the subject is unusable."""
    files = list(subject_dir.glob("*.nii.gz")) + list(subject_dir.glob("*.nii"))
    found: Dict[str, Path] = {}
    for modality, suffixes in SUFFIXES.items():
        for path in files:
            if any(path.name.endswith(s) for s in suffixes):
                # "_t1" must not also match "_t1ce"/"_t1gd"
                if modality == "t1" and any(
                    path.name.endswith(s) for s in SUFFIXES["t1ce"]
                ):
                    continue
                found[modality] = path
                break
    return found


def read_institutions(data_root: Path) -> Dict[str, int]:
    """Subject id -> institution, from the partitioning CSV shipped with FeTS."""
    import csv

    for name in ("partitioning_1.csv", "partitioning.csv", "partition_1.csv"):
        path = data_root / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No partitioning CSV under {data_root}. FeTS ships one; it is what "
            "defines the natural federation."
        )

    mapping: Dict[str, int] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        subject_col = institution_col = None
        for column in reader.fieldnames or []:
            low = column.lower().strip()
            if "subject" in low:
                subject_col = column
            elif any(k in low for k in ("institution", "partition", "center", "site")):
                institution_col = column
        if subject_col is None or institution_col is None:
            columns = list(reader.fieldnames or [])
            subject_col, institution_col = columns[0], columns[1]
        for row in reader:
            mapping[row[subject_col].strip()] = int(row[institution_col].strip())
    return mapping


def _load(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def normalize(volume: np.ndarray) -> np.ndarray:
    """Percentile-clip and z-score one modality over nonzero 3D voxels."""
    brain = volume[volume > 0]
    if brain.size == 0:
        return np.zeros_like(volume)

    lo, hi = np.percentile(brain, [0.5, 99.5])
    clipped_brain = np.clip(brain, lo, hi)
    mu = float(clipped_brain.mean())
    std = float(clipped_brain.std())
    if std <= 0:
        return np.zeros_like(volume)

    return np.where(volume > 0, (np.clip(volume, lo, hi) - mu) / std, 0).astype(np.float32)


def kept_slice_indices(
    subject_dir: Path, slice_stride: int, min_brain_fraction: float,
) -> Optional[List[int]]:
    """Return T1-filtered axial indices, or ``None`` for an unusable subject.

    All modalities must be present with matching shapes. Only T1 is loaded for
    the brain-fraction calculation.
    """
    files = find_modalities(subject_dir)
    if any(m not in files for m in REQUIRED):
        return None

    import nibabel as nib

    shapes = {nib.load(str(files[m])).shape for m in REQUIRED}
    if len(shapes) != 1:
        return None

    normalized = normalize(_load(files["t1"]))
    plane = normalized.shape[0] * normalized.shape[1]

    kept = [
        s for s in range(0, normalized.shape[-1], slice_stride)
        if np.count_nonzero(normalized[:, :, s]) / plane >= min_brain_fraction
    ]
    return kept or None


def slices_and_labels(
    subject_dir: Path, indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """``(N, 4, H, W)`` slices and their binary labels, for the given indices.

    The label collapses the segmentation mask: 1 if the slice contains any
    tumour voxel of any subtype, 0 otherwise.
    """
    files = find_modalities(subject_dir)
    volume = np.stack([normalize(_load(files[c])) for c in CHANNELS], axis=0)
    seg = _load(files["seg"])

    x = np.stack([volume[:, :, :, s] for s in indices], axis=0).astype(np.float32)
    y = np.array([int(np.any(seg[:, :, s] > 0)) for s in indices], dtype=np.int64)
    return x, y
