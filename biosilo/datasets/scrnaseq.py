"""scRNA-seq cell-type annotation with one client per sequencing study.

The default pancreas cohort uses Baron Human, Muraro, Segerstolpe, and Xin from
Zenodo record 3357167. Inputs contain the shared gene set after CPM, log2, and
per-study min-max normalization. Cell-type names are harmonized across studies;
cells have no group IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from ..core.contract import ClientData, unhashed

NAME = "scRNAseq"
STORAGE = "npz"
SCHEMA_VERSION = 2

#: study -> (subdirectory under Intra-dataset/, expression csv)
STUDIES = {
    "BaronHuman":  ("Pancreatic_data/Baron Human",  "Filtered_Baron_HumanPancreas_data.csv"),
    "BaronMouse":  ("Pancreatic_data/Baron Mouse",  "Filtered_MousePancreas_data.csv"),
    "Muraro":      ("Pancreatic_data/Muraro",       "Filtered_Muraro_HumanPancreas_data.csv"),
    "Segerstolpe": ("Pancreatic_data/Segerstolpe",  "Filtered_Segerstolpe_HumanPancreas_data.csv"),
    "Xin":         ("Pancreatic_data/Xin",          "Filtered_Xin_HumanPancreas_data.csv"),
    "TM":          ("TM",                           "Filtered_TM_data.csv"),
    "Zhengsorted": ("Zheng sorted",                 "Filtered_DownSampled_SortedPBMC_data.csv"),
    "Zheng68K":    ("Zheng 68K",                    "Filtered_68K_PBMC_data.csv"),
    "AMB":         ("AMB",                          "Filtered_mouse_allen_brain_data.csv"),
}

#: Supported multi-study cohorts.
COHORTS = {
    "pancreas": ("BaronHuman", "Muraro", "Segerstolpe", "Xin"),
}

#: Collections excluded because they contain only one study.
SINGLE_STUDY = ("TM", "Zhengsorted", "Zheng68K", "AMB", "BaronMouse")

#: Cross-study cell-type aliases.
SYNONYMS = {
    "pp": "gamma",                  # Muraro
    "duct": "ductal",               # Muraro
    "mesenchymal": "stellate",      # Muraro
    "PSC": "stellate",              # Segerstolpe; pancreatic stellate cell
    "activated_stellate": "stellate",   # Baron
    "quiescent_stellate": "stellate",   # Baron
}


#: Segerstolpe quality-control annotations excluded by default.
NON_BIOLOGICAL = ("co-expression", "unclassified endocrine")


@dataclass(frozen=True)
class Params:
    cohort: str = "pancreas"
    shared_labels_only: bool = False
    min_cells_per_type: int = 10
    exclude_types: Tuple[str, ...] = NON_BIOLOGICAL
    top_genes: int = 0              # 0 = every shared gene
    train_ratio: float = 0.80
    seed: int = 1

    source_dir: str = unhashed("")  # location, not identity


def label(p: Params) -> str:
    bits = [p.cohort, f"s{p.seed}"]
    if p.top_genes:
        bits.append(f"g{p.top_genes}")
    if p.shared_labels_only:
        bits.append("shared")
    if p.min_cells_per_type != 10:
        bits.append(f"min{p.min_cells_per_type}")
    if not p.exclude_types:
        bits.append("keepQC")
    return "_".join(bits)


def build(p: Params) -> Iterator[ClientData]:
    studies = _cohort_studies(p.cohort)
    raw = Path(p.source_dir)
    if not raw.is_dir():
        raise FileNotFoundError(
            f"source_dir not found: {raw}. Expected the extracted Zenodo 3357167 "
            "archive, containing an Intra-dataset/ directory."
        )

    counts, labels, origin = _load(raw, studies)
    counts, labels, origin, excluded = _exclude_types(
        counts, labels, origin, p.exclude_types)
    train_mask, test_mask = _split_masks(
        labels, origin, studies, p.train_ratio, p.seed)
    counts, labels, origin, train_mask, test_mask = _filter_types_from_training(
        counts, labels, origin, train_mask, test_mask, studies,
        p.shared_labels_only, p.min_cells_per_type)

    encoding = {name: i for i, name in enumerate(sorted(labels.unique()))}
    y_all = labels.map(encoding).to_numpy(np.int64)
    x_all, _ = _preprocess(counts, origin, train_mask, p.top_genes)

    class_names = {str(i): n for n, i in encoding.items()}

    for study in studies:
        rows = (origin == study).to_numpy()
        x, y = x_all[rows], y_all[rows]
        train_i = np.flatnonzero(train_mask[rows])
        test_i = np.flatnonzero(test_mask[rows])

        yield ClientData(
            client_id=study,
            train=(x[train_i], y[train_i], None),
            test=(x[test_i], y[test_i], None),
            meta={
                "n_cells": int(len(y)),
                "n_genes": int(x.shape[1]),
                "types_present": sorted({class_names[str(v)] for v in np.unique(y)}),
                "class_names": class_names,
                # Cohort-wide counts removed by ``exclude_types``.
                "excluded_types": excluded,
                "preprocessing_protocol": "inductive",
            },
        )


# ── internals ─────────────────────────────────────────────────────────────────

def _cohort_studies(cohort: str) -> Tuple[str, ...]:
    if cohort in COHORTS:
        return COHORTS[cohort]
    if cohort in SINGLE_STUDY:
        raise ValueError(
            f"{cohort!r} is a single study, so it has no natural federation. "
            "BioSilo only produces real-world partitions; synthetic splits "
            "(Dirichlet and friends) belong to the consuming framework. "
            f"Multi-study cohorts: {sorted(COHORTS)}."
        )
    raise ValueError(f"Unknown cohort {cohort!r}. Available: {sorted(COHORTS)}.")


def _strip_chr(gene: str) -> str:
    """Muraro names genes 'ACTB__chr7'. Every other study just says 'ACTB'."""
    return gene.split("__")[0] if "__chr" in gene else gene


def _gene_columns(raw: Path, study: str) -> dict:
    """Map normalized gene names to expression-column positions.

    Muraro suffix removal creates several duplicate names, so values are lists
    of positions. Position zero is the unnamed cell-ID column and must remain
    available to ``index_col=0``.
    """
    import pandas as pd

    subdir, csv_name = STUDIES[study]
    header = pd.read_csv(raw / "Intra-dataset" / subdir / csv_name, nrows=0)

    mapping: dict = {}
    for position, original in enumerate(header.columns):
        if position == 0:
            continue        # the cell-id column
        mapping.setdefault(_strip_chr(original), []).append(position)
    return mapping


def _load(raw: Path, studies: Tuple[str, ...]):
    """Load harmonized labels and the gene intersection across studies.

    Headers are scanned first so each matrix loads only shared columns. Loading
    all full-width matrices exceeded 6 GB for the pancreas cohort.
    """
    import pandas as pd

    columns = {s: _gene_columns(raw, s) for s in studies}
    shared = sorted(set.intersection(*(set(m) for m in columns.values())))
    if not shared:
        raise RuntimeError(
            f"No genes are shared across {studies}. Gene identifiers probably "
            "differ in convention between these studies. Check whether they "
            "use the same annotation release."
        )

    frames, labels, origin = [], [], []

    for study in studies:
        subdir, csv_name = STUDIES[study]
        directory = raw / "Intra-dataset" / subdir
        mapping = columns[study]

        # Position zero contains cell IDs.
        positions = sorted({0} | {p for gene in shared for p in mapping[gene]})
        x = pd.read_csv(directory / csv_name, index_col=0, usecols=positions)

        x.columns = [_strip_chr(g) for g in x.columns]
        if x.columns.duplicated().any():
            # Sum columns that collide after suffix removal.
            x = x.T.groupby(level=0).sum().T
        x = x[shared]

        y = pd.read_csv(directory / "Labels.csv", header=0).iloc[:, 0]
        y.index = x.index

        frames.append(x)
        labels.append(y.map(lambda v: SYNONYMS.get(v, v)))
        origin.append(pd.Series(study, index=x.index))

    return pd.concat(frames), pd.concat(labels), pd.concat(origin)


def _exclude_types(counts, labels, origin, exclude: Tuple[str, ...] = ()):
    """Apply fixed annotation exclusions before establishing the split."""
    removed: dict = {}
    if exclude:
        hit = labels.isin(exclude)
        if hit.any():
            removed = {str(k): int(v) for k, v in labels[hit].value_counts().items()}
            keep = ~hit
            counts, labels, origin = counts[keep], labels[keep], origin[keep]
    return counts, labels, origin, removed


def _split_masks(labels, origin, studies, train_ratio: float, seed: int):
    """Create deterministic per-study splits before any fitted preprocessing."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    train = np.zeros(len(labels), dtype=bool)
    test = np.zeros(len(labels), dtype=bool)
    label_values = labels.to_numpy()
    origin_values = origin.to_numpy()

    for study in studies:
        study_rows = np.flatnonzero(origin_values == study)
        if len(study_rows) < 2:
            raise ValueError(f"Study {study!r} needs at least two retained cells")

        # Allocate each type independently when possible. This keeps rare types
        # represented on both sides without requiring scikit-learn.
        for cell_type in sorted(set(label_values[study_rows])):
            rows = study_rows[label_values[study_rows] == cell_type]
            rows = rng.permutation(rows)
            if len(rows) == 1:
                train[rows] = True
                continue
            cut = int(np.floor(len(rows) * train_ratio))
            cut = min(max(cut, 1), len(rows) - 1)
            train[rows[:cut]] = True
            test[rows[cut:]] = True

        # A study made entirely of singleton labels still needs a test split.
        if not test[study_rows].any():
            candidates = study_rows[train[study_rows]]
            moved = candidates[int(rng.integers(len(candidates)))]
            train[moved] = False
            test[moved] = True

    return train, test


def _filter_types_from_training(
    counts, labels, origin, train_mask, test_mask, studies,
    shared_only: bool, min_cells: int,
):
    """Fit label-presence rules on training cells, then apply them to both splits."""
    training_labels = labels.iloc[np.flatnonzero(train_mask)]
    allowed = set(training_labels.unique())
    if shared_only:
        per_study = [
            set(labels[(origin == study).to_numpy() & train_mask].unique())
            for study in studies
        ]
        allowed &= set.intersection(*per_study)

    frequency = training_labels.value_counts()
    allowed &= set(frequency[frequency >= min_cells].index)
    if not allowed:
        raise ValueError(
            "No cell types survived the training-only prevalence filters")

    keep = labels.isin(allowed).to_numpy()
    filtered_train = train_mask[keep]
    filtered_test = test_mask[keep]
    for study in studies:
        study_rows = (origin[keep] == study).to_numpy()
        if not filtered_train[study_rows].any() or not filtered_test[study_rows].any():
            raise ValueError(
                f"Study {study!r} has an empty split after cell-type filtering")
    return (counts.iloc[np.flatnonzero(keep)], labels.iloc[np.flatnonzero(keep)],
            origin.iloc[np.flatnonzero(keep)], filtered_train, filtered_test)


def _preprocess(counts, origin, train_mask, top_genes: int):
    """Apply per-cell transforms and training-fitted scaling/gene selection.

    NumPy operations run in place to avoid several copies of the full matrix.
    Intermediate arithmetic remains float64 before one final float32 cast.
    """
    x = counts.to_numpy(dtype=np.float64, copy=True)

    depth = x.sum(axis=1, keepdims=True)
    np.maximum(depth, 1.0, out=depth)      # a cell with no counts must not divide by zero
    x /= depth
    x *= 1e6
    x += 1.0
    np.log2(x, out=x)

    origin_values = origin.to_numpy()
    for study in origin.unique():
        rows = origin_values == study
        fit_rows = rows & train_mask
        if not fit_rows.any():
            raise ValueError(f"Study {study!r} has no training cells for scaling")
        fit = x[fit_rows]
        lo = fit.min(axis=0)
        span = fit.max(axis=0) - lo
        span[span == 0] = 1.0              # a gene constant within a study
        block = x[rows]
        block -= lo
        block /= span
        np.clip(block, 0.0, 1.0, out=block)
        x[rows] = block

    keep = np.arange(x.shape[1])
    if top_genes and top_genes < x.shape[1]:
        keep = np.sort(np.argsort(x[train_mask].var(axis=0))[::-1][:top_genes])
        x = x[:, keep]

    return x.astype(np.float32, copy=False), keep
