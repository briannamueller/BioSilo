"""Recover subject group IDs for legacy BraTS partitions.

The utility replays the legacy subject ordering and seeded split. It reads T1
to reproduce retained slice counts and checks all modality shapes. Before
writing, every reconstructed client must match the stored slice counts and one
selected client's segmentation-derived labels must match ``{cid}_y.npy``.
Existing arrays are unchanged; ``--write`` adds ``{cid}_groups.npy`` files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")

#: Suffix patterns per modality, as the original matches them.
SUFFIXES = {
    "t1": ["_t1.nii", "_T1.nii", "_t1.nii.gz", "_T1.nii.gz"],
    "t1ce": ["_t1ce.nii", "_T1ce.nii", "_t1gd.nii", "_T1Gd.nii",
             "_t1ce.nii.gz", "_T1ce.nii.gz", "_t1gd.nii.gz", "_T1Gd.nii.gz"],
    "t2": ["_t2.nii", "_T2.nii", "_t2.nii.gz", "_T2.nii.gz"],
    "flair": ["_flair.nii", "_FLAIR.nii", "_t2f.nii", "_T2f.nii",
              "_flair.nii.gz", "_FLAIR.nii.gz", "_t2f.nii.gz", "_T2f.nii.gz"],
    "seg": ["_seg.nii", "_Seg.nii", "_seg.nii.gz", "_Seg.nii.gz"],
}

SUBJECT_PREFIXES = ("FeTS", "BraTS", "UPENN", "Case")


# ── discovery, mirroring the original ─────────────────────────────────────────

def find_subject_dirs(data_root: Path) -> List[Path]:
    dirs = [e for e in sorted(data_root.iterdir())
            if e.is_dir() and any(p in e.name for p in SUBJECT_PREFIXES)]
    if dirs:
        return dirs
    for subdir in sorted(data_root.iterdir()):
        if subdir.is_dir():
            dirs.extend(e for e in sorted(subdir.iterdir())
                        if e.is_dir() and any(p in e.name for p in SUBJECT_PREFIXES))
    return dirs


def read_institution_mapping(data_root: Path) -> Dict[str, int]:
    import csv

    for name in ("partitioning_1.csv", "partitioning.csv", "partition_1.csv"):
        path = data_root / name
        if path.exists():
            break
    else:
        sys.exit(f"No partitioning CSV in {data_root}")

    mapping = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        subj_col = inst_col = None
        for col in reader.fieldnames:
            low = col.lower().strip()
            if "subject" in low:
                subj_col = col
            elif any(k in low for k in ("institution", "partition", "center", "site")):
                inst_col = col
        if subj_col is None or inst_col is None:
            cols = list(reader.fieldnames)
            subj_col, inst_col = cols[0], cols[1]
        for row in reader:
            mapping[row[subj_col].strip()] = int(row[inst_col].strip())
    return mapping


def find_modality_files(subject_dir: Path) -> Dict[str, Path]:
    files = list(subject_dir.glob("*.nii.gz")) + list(subject_dir.glob("*.nii"))
    found: Dict[str, Path] = {}
    for modality, suffixes in SUFFIXES.items():
        for path in files:
            for suffix in suffixes:
                if path.name.endswith(suffix):
                    # Exclude T1-contrast suffixes from the T1 match.
                    if modality == "t1" and any(
                        path.name.endswith(s) for s in SUFFIXES["t1ce"]
                    ):
                        continue
                    found[modality] = path
                    break
            if modality in found:
                break
    return found


# ── the part that decides how many slices a subject contributes ───────────────

def slice_profile(
    subject_dir: Path,
    slice_stride: int,
    min_brain_fraction: float,
    with_labels: bool = False,
) -> Optional[Tuple[int, Optional[np.ndarray]]]:
    """Return retained slice count and optional labels for a legacy subject.

    Float32 normalization matches the legacy generator's brain-voxel count.
    """
    import nibabel as nib

    files = find_modality_files(subject_dir)
    if any(m not in files for m in MODALITIES):
        return None

    # Shapes must all match; headers are enough to check that.
    shapes = {m: nib.load(str(files[m])).shape for m in MODALITIES}
    if len(set(shapes.values())) != 1:
        return None

    t1 = np.asarray(nib.load(str(files["t1"])).dataobj, dtype=np.float32)

    brain = t1[t1 > 0]
    if brain.size > 0:
        lo, hi = np.percentile(brain, [0.5, 99.5])
        clipped_brain = np.clip(brain, lo, hi)
        mu = float(clipped_brain.mean())
        std = float(clipped_brain.std())
    else:
        mu, std, lo, hi = 0.0, 1.0, 0.0, 0.0

    normalized = np.zeros_like(t1)
    if std > 0:
        mask = t1 > 0
        normalized = np.where(mask, (np.clip(t1, lo, hi) - mu) / std, 0).astype(np.float32)

    seg = None
    if with_labels:
        seg = np.asarray(nib.load(str(files["seg"])).dataobj, dtype=np.float32)

    plane = normalized.shape[0] * normalized.shape[1]
    kept = 0
    labels: List[int] = []

    for s in range(0, normalized.shape[-1], slice_stride):
        if np.count_nonzero(normalized[:, :, s]) / plane < min_brain_fraction:
            continue
        kept += 1
        if with_labels:
            labels.append(int(np.any(seg[:, :, s] > 0)))

    return kept, (np.array(labels, dtype=np.int64) if with_labels else None)


# ── replaying the split ───────────────────────────────────────────────────────

def replay_split(n_subjects: int, train_ratio: float) -> Tuple[List[int], List[int]]:
    """One institution's train/test subject indices, consuming the RNG as the
    original does. Must be called once per institution, in the same order."""
    order = list(range(n_subjects))
    random.shuffle(order)
    n_train = max(1, int(n_subjects * train_ratio))
    if n_train == n_subjects and n_subjects > 1:
        n_train = n_subjects - 1
    return order[:n_train], order[n_train:]


def subject_ids(indices: List[int], counts: List[int], names: List[str]) -> np.ndarray:
    """Per-slice subject id, in the order the original concatenated them."""
    return np.concatenate([
        np.repeat(names[j], counts[j]) for j in indices
    ]) if indices else np.array([], dtype="<U64")


# ── driver ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-dir", required=True,
                    help="Directory holding subject dirs + the partitioning CSV.")
    ap.add_argument("--partition-dir", required=True,
                    help="Generated partition (contains manifest.json, train/, test/).")
    ap.add_argument("--verify-client", type=int, default=0,
                    help="Client index whose labels are recomputed and compared "
                         "against disk (default 0). Length checks run on all.")
    ap.add_argument("--write", action="store_true",
                    help="Write {cid}_groups.npy. Without this, nothing is written.")
    args = ap.parse_args()

    data_root = Path(args.source_dir)
    part = Path(args.partition_dir)
    manifest = json.loads((part / "manifest.json").read_text())
    settings = manifest["settings"]

    seed = int(settings["seed"])
    stride = int(settings["slice_stride"])
    min_brain = float(settings["min_brain_fraction"])
    train_ratio = float(settings["train_ratio"])
    institutions = [int(client["client_id"]) for client in manifest["clients"]]

    print(f"partition: {part.name}")
    print(f"  seed={seed} stride={stride} min_brain={min_brain} "
          f"train_ratio={train_ratio}")
    print(f"  {len(institutions)} institutions: {institutions}")

    mapping = read_institution_mapping(data_root)
    by_institution: Dict[int, List[Path]] = defaultdict(list)
    for subj in find_subject_dirs(data_root):
        institution = mapping.get(subj.name)
        if institution is None:
            for key, value in mapping.items():
                if key in subj.name or subj.name in key:
                    institution = value
                    break
        if institution is not None:
            by_institution[institution].append(subj)

    # Replay one shared RNG stream in legacy institution order.
    random.seed(seed)

    # Validate all reconstructed clients before writing any group files.
    pending: List[Tuple[int, str, np.ndarray]] = []
    failures: List[str] = []
    label_checked = None

    for position, institution in enumerate(institutions):
        subjects = by_institution.get(institution, [])
        want_labels = position == args.verify_client

        names, counts, labels = [], [], []
        for subj in subjects:
            profile = slice_profile(subj, stride, min_brain, with_labels=want_labels)
            if profile is None:
                continue
            kept, subj_labels = profile
            names.append(subj.name)
            counts.append(kept)
            if want_labels:
                labels.append(subj_labels)

        # Consume the shared RNG stream for every institution in order.
        train_idx, test_idx = replay_split(len(names), train_ratio)

        for split, indices in (("train", train_idx), ("test", test_idx)):
            ids = subject_ids(indices, counts, names)
            saved_y = np.load(part / split / f"{position}_y.npy")

            if len(ids) != len(saved_y):
                failures.append(
                    f"client {position} (institution {institution}) {split}: "
                    f"reconstructed {len(ids)} slices, {len(saved_y)} on disk")
                continue

            if want_labels:
                recomputed = np.concatenate([labels[j] for j in indices]) \
                    if indices else np.array([], dtype=np.int64)
                if not np.array_equal(recomputed, saved_y):
                    wrong = int((recomputed != saved_y).sum())
                    failures.append(
                        f"client {position} {split}: {wrong} of {len(saved_y)} "
                        "labels differ, so subject ordering is wrong")
                    continue
                label_checked = (position, institution)

            pending.append((position, split, ids))

        print(f"  client {position:2} (institution {institution:2}): "
              f"{len(names):3} subjects, "
              f"{sum(counts):5} slices"
              f"{'  [labels verified]' if want_labels else ''}")

    if failures:
        print("\nFAILED, nothing written:")
        for line in failures:
            print(f"  {line}")
        print("\nThe replay does not reproduce this partition. Regeneration is required.")
        return 1

    print(f"\nAll {len(institutions)} clients reconstruct to the exact slice "
          "counts on disk.")
    if label_checked:
        print(f"Client {label_checked[0]} (institution {label_checked[1]}) also "
              "reproduces its labels exactly, so the slice-to-subject mapping "
              "and every shuffle up to it are correct.")

    if not args.write:
        print("\nNothing written (pass --write to save subject ids).")
        return 0

    for position, split, ids in pending:
        np.save(part / split / f"{position}_groups.npy", ids)
    print(f"\nWrote subject ids for {len(institutions)} clients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
