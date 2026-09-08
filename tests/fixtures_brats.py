"""Create a small FeTS-compatible NIfTI fixture.

The fixture includes the expected modality files and partitioning CSV, plus:

* an institution below the subject floor
* a subject missing a modality
* a subject whose volumes disagree in shape
* mostly empty slices
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

H = W = 16
DEPTH = 12

#: (institution, n_subjects). Institution 9 is below the default floor of 5.
COHORT = [(1, 6), (2, 5), (9, 2)]

#: Slices with brain content. The rest are blank and must be filtered out.
BRAIN_SLICES = (2, 3, 4, 5, 6, 7, 8, 9)


def build(root: Path, stride: int = 1) -> Path:
    data = root / "fets"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    counter = 0

    for institution, n_subjects in COHORT:
        for k in range(n_subjects):
            counter += 1
            name = f"FeTS2022_{counter:05d}"
            subject = data / name
            subject.mkdir(exist_ok=True)
            rows.append((name, institution))

            # One subject in institution 2 is missing FLAIR; another has a
            # mismatched shape. Both must be skipped entirely.
            skip_flair = institution == 2 and k == 0
            odd_shape = institution == 2 and k == 1

            shape = (H + 2, W, DEPTH) if odd_shape else (H, W, DEPTH)

            for modality in ("t1", "t1ce", "t2", "flair"):
                if modality == "flair" and skip_flair:
                    continue
                volume = np.zeros(shape, dtype=np.float32)
                for s in BRAIN_SLICES:
                    volume[:, :, s] = rng.random(shape[:2]) * 500 + 1
                _save(volume, subject / f"{name}_{modality}.nii.gz")

            # Tumour in some slices of some subjects, so labels are not constant.
            seg = np.zeros(shape, dtype=np.float32)
            if k % 2 == 0:
                for s in BRAIN_SLICES[:3]:
                    seg[2:5, 2:5, s] = 1.0
            _save(seg, subject / f"{name}_seg.nii.gz")

    with open(data / "partitioning_1.csv", "w") as f:
        f.write("Subject_ID,Partition_ID\n")
        for name, institution in rows:
            f.write(f"{name},{institution}\n")

    return data


def _save(volume: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(path))
