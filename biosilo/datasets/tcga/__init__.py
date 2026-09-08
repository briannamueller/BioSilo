"""TCGA bulk tumour transcriptomics with one client per contributing hospital.

Supported tasks are cancer type, advanced stage, and thresholded survival.
Inputs are filtered gene-expression vectors loaded from Xena or GDC. TCGA
tissue-source-site codes map samples to hospitals, while train/test splits keep
all samples from a patient together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from ...core.contract import ClientData, unhashed
from ._omics import gene_filter, natural, registry

NAME = "TCGA"
STORAGE = "npz"
SCHEMA_VERSION = 3

TASKS = ("cancer_type", "stage", "survival")
SOURCES = ("xena", "gdc")
EXPRESSION_TRANSFORMS = ("source_default", "none")

@dataclass(frozen=True)
class Params:
    cohorts: Tuple[str, ...] = ("LUAD", "LUSC", "BRCA", "COAD")
    task: str = "cancer_type"
    source: str = "xena"
    expression_transform: str = "source_default"
    top_genes: int = 2000
    min_samples: int = 20           # per hospital
    min_nonzero_fraction: float = 0.10
    survival_threshold_days: int = 365 * 3
    train_ratio: float = 0.80
    seed: int = 1

    source_dir: str = unhashed("")  # prepared source data, not identity


def label(p: Params) -> str:
    return f"{p.task}_s{p.seed}"


def build(p: Params) -> Iterator[ClientData]:
    _check(p)
    source = Path(p.source_dir) / p.source

    expression, phenotype = _read_source(p, source)
    labels, class_map, valid = _encode_labels(p, expression, phenotype, source)

    expression = expression.loc[valid]
    labels = np.asarray(labels)[valid.to_numpy()]
    expression = _transform_expression(expression, p.source, p.expression_transform)

    clients = natural.partition_by_center(
        expression, labels, min_samples=p.min_samples, include_expression=False)
    if not clients:
        raise RuntimeError(
            f"No hospital contributed at least {p.min_samples} samples for "
            f"cohorts={list(p.cohorts)}."
        )

    rng = np.random.default_rng(p.seed)
    names = {str(v): k for k, v in class_map.items()} if isinstance(
        class_map, dict) else {}

    prepared = []
    training_samples = []
    for client in clients:
        y = np.asarray(client["y"], dtype=np.int64)
        groups = np.array([_patient(s) for s in client["sample_ids"]], dtype="<U32")
        train_i, test_i = _split(y, groups, p.train_ratio, rng)
        prepared.append((client, y, groups, train_i, test_i))
        training_samples.extend(np.asarray(client["sample_ids"])[train_i].tolist())

    genes = _fit_genes(
        expression, training_samples, p.min_nonzero_fraction, p.top_genes)
    expression = expression.loc[:, genes]

    for client, y, groups, train_i, test_i in prepared:
        x = expression.loc[client["sample_ids"]].to_numpy(dtype=np.float32)

        yield ClientData(
            client_id=str(client["center_id"]),
            train=(x[train_i], y[train_i], groups[train_i]),
            test=(x[test_i], y[test_i], groups[test_i]),
            meta={
                "group_unit": "patient",
                "tss_codes": client.get("tss_codes", []),
                "n_samples": int(len(y)),
                "n_patients": int(len(np.unique(groups))),
                "class_names": names,
                "preprocessing_protocol": "inductive",
                "expression_transform": _resolved_transform(
                    p.source, p.expression_transform),
            },
        )


# ── internals ─────────────────────────────────────────────────────────────────

def download(p: Params) -> Path:
    """Explicitly download and prepare the selected TCGA source."""
    if p.source not in SOURCES:
        raise ValueError(f"Unknown source {p.source!r}. Choose from {list(SOURCES)}.")
    cohorts = registry.validate_cohorts(list(p.cohorts))
    destination = Path(p.source_dir) / p.source

    if p.source == "xena":
        from ._omics import xena
        xena.download(cohorts, output_dir=destination)
    else:
        from ._omics import gdc
        gdc.download(cohorts, output_dir=destination)
    return destination


def _check(p: Params) -> None:
    if p.task not in TASKS:
        raise ValueError(f"Unknown task {p.task!r}. Choose from {list(TASKS)}.")
    if p.source not in SOURCES:
        raise ValueError(f"Unknown source {p.source!r}. Choose from {list(SOURCES)}.")
    if p.expression_transform not in EXPRESSION_TRANSFORMS:
        raise ValueError(
            f"Unknown expression_transform {p.expression_transform!r}. "
            f"Choose from {list(EXPRESSION_TRANSFORMS)}.")
    if p.task == "cancer_type" and len(p.cohorts) < 2:
        raise ValueError(
            "cancer_type needs at least two cohorts; with one there is nothing "
            "to distinguish. Use task='stage' or 'survival' for a single cohort."
        )
    if p.task == "survival" and p.source == "gdc":
        raise ValueError(
            "survival needs source='xena'. GDC's clinical download carries no "
            "survival endpoints."
        )
    registry.validate_cohorts(list(p.cohorts))


def _resolved_transform(source: str, requested: str) -> str:
    if requested == "none":
        return "none"
    return "cpm_log2" if source == "gdc" else "none"


def _transform_expression(expression, source: str, requested: str):
    """Apply the source-aware, per-sample transform that requires no fitting."""
    mode = _resolved_transform(source, requested)
    if mode == "none":
        return expression

    values = expression.to_numpy(dtype=np.float64, copy=True)
    depth = values.sum(axis=1, keepdims=True)
    np.maximum(depth, 1.0, out=depth)
    values /= depth
    values *= 1e6
    np.log2(values + 1.0, out=values)
    return expression.__class__(
        values, index=expression.index, columns=expression.columns)


def _fit_genes(expression, training_samples, min_nonzero_fraction, top_genes):
    """Fit nonzero and variance filters using training expression only."""
    if not training_samples:
        raise ValueError("TCGA feature fitting received no training samples")
    fit = expression.loc[training_samples]
    keep = gene_filter.nonzero_genes(
        fit, min_fraction=min_nonzero_fraction)
    fit = fit.loc[:, keep]
    if fit.shape[1] == 0:
        raise ValueError("No genes survived the training-only nonzero filter")
    return gene_filter.high_variance_genes(fit, top_k=top_genes)


def _patient(barcode: str) -> str:
    """Extract the three-field patient ID from a TCGA sample barcode."""
    parts = str(barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(barcode)


def _split(y, groups, train_ratio: float, rng):
    """Split samples with stratification and patient grouping when available."""
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    n = len(y)
    random_state = int(rng.integers(0, np.iinfo(np.int32).max))
    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 else None

    if len(np.unique(groups)) == n:          # one sample per patient
        train_i, test_i = train_test_split(
            np.arange(n), train_size=train_ratio,
            stratify=stratify, random_state=random_state)
        return np.sort(train_i), np.sort(test_i)

    n_splits = max(2, min(int(round(1.0 / max(1e-9, 1.0 - train_ratio))),
                          len(np.unique(groups))))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    train_i, test_i = next(splitter.split(np.zeros(n), y, groups))
    return np.sort(train_i), np.sort(test_i)


def _read_source(p: Params, source: Path):
    import pandas as pd

    cohorts = list(p.cohorts)

    if p.source == "xena":
        from ._omics import xena

        paths = xena.local_files(source, require_survival=p.task == "survival")
        expression = xena.load_expression(
            paths["expression_path"], cohorts, paths["phenotype_path"])
        phenotype = xena.load_phenotype(
            paths["phenotype_path"], samples=expression.index.tolist())
        return expression, phenotype

    from ._omics import gdc

    results = gdc.local_files(cohorts, output_dir=source)
    frames, phenotype = [], []
    for cohort, paths in results.items():
        expression = pd.read_parquet(paths["counts"])
        clin = pd.read_csv(paths["clinical"])
        sample_cases = pd.read_csv(paths["sample_cases"])
        frames.append(expression)
        phenotype.append(
            _align_gdc_clinical(expression, sample_cases, clin, cohort))

    if len(frames) > 1:
        shared = sorted(set.intersection(*(set(f.columns) for f in frames)))
        frames = [f[shared] for f in frames]

    return pd.concat(frames, axis=0), pd.concat(phenotype, axis=0)


def _align_gdc_clinical(expression, sample_cases, clinical, cohort: str):
    """Expand GDC case-level clinical rows onto expression sample barcodes."""
    import pandas as pd

    required_mapping = {"sample_barcode", "case_id"}
    missing = required_mapping - set(sample_cases.columns)
    if missing:
        raise ValueError(
            f"GDC sample-case mapping is missing columns: {sorted(missing)}")
    if "case_id" not in clinical.columns:
        raise ValueError("GDC clinical data is missing column 'case_id'")
    if not expression.index.is_unique:
        raise ValueError("GDC expression sample barcodes must be unique")

    mapping = sample_cases[["sample_barcode", "case_id"]].copy()
    if mapping.isna().any(axis=None) or mapping.eq("").any(axis=None):
        raise ValueError("GDC sample-case mapping contains a missing identifier")
    conflicts = mapping.groupby("sample_barcode")["case_id"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "GDC mapped a sample barcode to multiple cases: "
            + ", ".join(conflicts.index.astype(str)))
    mapping = mapping.drop_duplicates().set_index("sample_barcode")

    case_ids = mapping["case_id"].reindex(expression.index)
    if case_ids.isna().any():
        absent = case_ids.index[case_ids.isna()].astype(str).tolist()
        raise ValueError(
            "GDC expression samples have no case mapping: "
            + ", ".join(absent[:10]))

    if clinical["case_id"].isna().any() or clinical["case_id"].duplicated().any():
        raise ValueError("GDC clinical case IDs must be present and unique")
    by_case = clinical.set_index("case_id")
    aligned = by_case.reindex(case_ids.to_numpy()).copy()
    aligned.index = expression.index
    aligned.index.name = expression.index.name
    aligned.insert(0, "case_id", case_ids.to_numpy())
    aligned["cohort"] = cohort
    return aligned


def _encode_labels(p: Params, expression, phenotype, source: Path):
    import pandas as pd

    from ._omics import labels as encode

    if p.task == "cancer_type":
        if "cohort" not in phenotype.columns:
            # Cancer-type labels require explicit cohort assignments.
            raise ValueError(
                "Phenotype data has no 'cohort' column, so samples cannot be "
                "assigned to a cancer type. This usually means the download "
                "returned clinical data without cohort labels. Check the "
                f"{p.source!r} source data in {source}."
            )
        assignments = phenotype["cohort"].reindex(expression.index)
        y, class_map = encode.encode_cancer_type(
            expression.index.tolist(), assignments, list(p.cohorts))
        return y, class_map, pd.Series(True, index=expression.index)

    if p.task == "stage":
        if "stage" not in phenotype.columns:
            raise ValueError("Phenotype data has no 'stage' column.")
        y, class_map = encode.encode_stage(
            phenotype["stage"].reindex(expression.index))
        return y, class_map, pd.Series(~np.isnan(y), index=expression.index)

    survival = Path(source) / "TCGA_survival_data.tsv"
    if not survival.exists():
        raise FileNotFoundError(
            f"Survival data not found at {survival}. It ships with the Xena "
            "download; re-run the download if it is missing.")
    table = pd.read_csv(survival, sep="\t", index_col="sample").reindex(expression.index)
    y, class_map = encode.encode_survival(
        table["OS.time"], table["OS"].map({1: "Dead", 0: "Alive"}),
        p.survival_threshold_days)
    return y, class_map, pd.Series(~np.isnan(y), index=expression.index)
