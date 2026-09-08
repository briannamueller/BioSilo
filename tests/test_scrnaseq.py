"""scRNA-seq gene, label, cohort, and split tests.

Set ``SCRNASEQ_LABELS_DIR`` to run the optional source-label cases.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import biosilo
import fixtures_scrnaseq as fx

def _env_dir(name):
    """The directory ``name`` points at, or None if the variable is unset.

    ``Path("")`` is ``Path(".")``, which always exists, so an unset variable
    would otherwise pass the skip guard and run the case against the repo root.
    """
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


LABELS_DIR = _env_dir("SCRNASEQ_LABELS_DIR")

PASSED, FAILED = [], []


def case(fn):
    def run(root, **kw):
        try:
            fn(root, **kw)
        except Exception as exc:  # noqa: BLE001
            FAILED.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            PASSED.append(fn.__name__)
    return run


def _generate(root, labels_dir=None, **overrides):
    raw = fx.build(root, labels_dir=labels_dir)
    kwargs = dict(cohort="pancreas", seed=1, source_dir=str(raw))
    kwargs.update(overrides)
    out = biosilo.generate("scRNAseq", root=root / "data", **kwargs)
    return biosilo.load("scRNAseq", root=root / "data", partition=out.name)


# ── federation ────────────────────────────────────────────────────────────────

@case
def one_client_per_study(root):
    p = _generate(root)
    assert p.num_clients == 4, p.num_clients
    assert p.client_ids == list(fx.PANCREAS), p.client_ids
    assert p.has_groups is False
    assert p.group_unit is None


@case
def single_study_cohorts_are_refused(root):
    try:
        _generate(root, cohort="Zheng68K")
    except ValueError as exc:
        assert "no natural federation" in str(exc)
        assert "synthetic splits" in str(exc)
    else:
        raise AssertionError("a single-study cohort was accepted")


# ── genes ─────────────────────────────────────────────────────────────────────

@case
def genes_are_intersected_across_studies(root):
    p = _generate(root)
    assert p.inputs[0]["shape"] == [len(fx.SHARED_GENES)], p.inputs[0]
    # Study-specific genes are excluded from the intersection.
    X, _, _ = p.client(0, "train")
    assert X.shape[1] == len(fx.SHARED_GENES), X.shape


@case
def chromosome_suffixes_are_stripped(root):
    """Muraro names genes 'INS__chr11'. Without stripping, nothing intersects."""
    p = _generate(root)
    # Muraro is client 1; if its suffixes survived, the intersection would be
    # empty and generation would have raised.
    X, y, _ = p.client(1, "train")
    assert X.shape[1] == len(fx.SHARED_GENES), X.shape
    assert len(y) > 0


@case
def top_genes_selects_a_subset(root):
    p = _generate(root, top_genes=3)
    assert p.inputs[0]["shape"] == [3], p.inputs[0]


@case
def test_cells_cannot_change_fitted_preprocessing(root):
    """Test-only expression must not affect scalers, selected genes, or train X."""
    from biosilo.datasets import scrnaseq

    raw = fx.build(root)
    counts, labels, origin = scrnaseq._load(raw, fx.PANCREAS)
    counts, labels, origin, _ = scrnaseq._exclude_types(
        counts, labels, origin, scrnaseq.NON_BIOLOGICAL)
    train, test = scrnaseq._split_masks(
        labels, origin, fx.PANCREAS, 0.8, 1)
    counts, labels, origin, train, test = scrnaseq._filter_types_from_training(
        counts, labels, origin, train, test, fx.PANCREAS, False, 1)

    baseline, baseline_genes = scrnaseq._preprocess(
        counts, origin, train, top_genes=3)
    changed = counts.copy()
    changed.iloc[np.flatnonzero(test), 0] *= 1_000_000
    perturbed, perturbed_genes = scrnaseq._preprocess(
        changed, origin, train, top_genes=3)

    assert np.array_equal(baseline_genes, perturbed_genes)
    assert np.array_equal(baseline[train], perturbed[train])
    assert np.isfinite(perturbed).all()
    assert perturbed.min() >= 0.0 and perturbed.max() <= 1.0


@case
def cell_ids_are_not_mistaken_for_a_gene(root):
    """Keep the unnamed cell-ID column when selecting expression columns."""
    from biosilo.datasets.scrnaseq import _load

    with tempfile.TemporaryDirectory() as tmp:
        raw = fx.build(Path(tmp))
        counts, labels, origin = _load(raw, fx.PANCREAS)

    assert list(counts.columns) == fx.SHARED_GENES, list(counts.columns)
    assert len(counts) == len(labels) == len(origin)
    # cell ids are strings the fixture generated, not expression values
    assert all(isinstance(i, str) and "_cell" in i for i in counts.index[:5]), \
        list(counts.index[:5])


# ── labels ────────────────────────────────────────────────────────────────────

@case
def synonyms_collapse_to_one_class(root):
    p = _generate(root, min_cells_per_type=1)
    names = set(p.manifest["clients"][0]["metadata"]["class_names"].values())
    # every synonym key must be gone, replaced by its canonical form
    for synonym in ("pp", "duct", "PSC", "activated_stellate",
                    "quiescent_stellate", "mesenchymal"):
        assert synonym not in names, f"{synonym} survived harmonization"
    assert "gamma" in names and "ductal" in names and "stellate" in names


@case
def rare_types_are_dropped(root):
    # 12 cells per type in the fixture; a threshold above that removes everything
    loose = _generate(root, min_cells_per_type=1)
    with tempfile.TemporaryDirectory() as other:
        strict = _generate(Path(other), min_cells_per_type=13)
    assert strict.num_classes < loose.num_classes, (strict.num_classes, loose.num_classes)


@case
def shared_labels_only_reduces_to_the_intersection(root):
    p = _generate(root, shared_labels_only=True, min_cells_per_type=1)
    names = set(p.manifest["clients"][0]["metadata"]["class_names"].values())
    assert names == fx.SHARED_TYPES, names


# ── splits ────────────────────────────────────────────────────────────────────

@case
def splits_partition_each_client(root):
    p = _generate(root)
    assert all(client["metadata"]["preprocessing_protocol"] == "inductive"
               for client in p.manifest["clients"])
    for cid in range(p.num_clients):
        Xtr, ytr, gtr = p.client(cid, "train")
        Xte, yte, gte = p.client(cid, "test")
        assert gtr is None and gte is None
        assert len(Xtr) == len(ytr) and len(Xte) == len(yte)
        total = p.manifest["clients"][cid]["metadata"]["n_cells"]
        assert len(ytr) + len(yte) == total, (len(ytr), len(yte), total)


# ── against the real label vocabulary ─────────────────────────────────────────

@case
def qc_annotations_are_excluded_and_recorded(root):
    """Excluded quality-control labels and counts must be recorded."""
    p = _generate(root, min_cells_per_type=1)
    metadata = p.manifest["clients"][0]["metadata"]
    names = set(metadata["class_names"].values())
    assert "co-expression" not in names, names

    # the choice must be legible from the partition alone
    assert "co-expression" in p.settings["exclude_types"], p.settings
    removed = metadata["excluded_types"]
    assert removed.get("co-expression", 0) > 0, removed


@case
def qc_exclusion_can_be_turned_off(root):
    p = _generate(root, min_cells_per_type=1, exclude_types=())
    metadata = p.manifest["clients"][0]["metadata"]
    names = set(metadata["class_names"].values())
    assert "co-expression" in names, names
    assert metadata["excluded_types"] == {}


@case
def real_labels_give_twelve_classes(root):
    """Source labels produce 12 classes after configured QC exclusions."""
    p = _generate(root, labels_dir=LABELS_DIR)
    assert p.num_clients == 4, p.num_clients
    assert p.num_classes == 12, p.num_classes

    metadata = p.manifest["clients"][0]["metadata"]
    names = set(metadata["class_names"].values())
    for gone in ("MHC class II", "t_cell", "unclassified endocrine",
                 "co-expression"):
        assert gone not in names, f"{gone} should not be a class"

    assert metadata["excluded_types"] == {
        "co-expression": 39, "unclassified endocrine": 5,
    }, metadata["excluded_types"]


@case
def real_labels_shared_only_gives_four(root):
    p = _generate(root, labels_dir=LABELS_DIR, shared_labels_only=True)
    assert p.num_classes == 4, p.num_classes
    names = set(p.manifest["clients"][0]["metadata"]["class_names"].values())
    assert names == {"alpha", "beta", "delta", "gamma"}, names


@case
def real_labels_every_synonym_fires(root):
    """Every configured synonym must occur in the source labels."""
    import pandas as pd

    from biosilo.datasets.scrnaseq import SYNONYMS

    seen = set()
    for study in fx.PANCREAS:
        seen |= set(pd.read_csv(LABELS_DIR / f"{study}.csv", header=0).iloc[:, 0].unique())
    unused = [k for k in SYNONYMS if k not in seen]
    assert not unused, f"synonym map has entries nothing uses: {unused}"


FIXTURE_CASES = [
    one_client_per_study,
    single_study_cohorts_are_refused,
    genes_are_intersected_across_studies,
    chromosome_suffixes_are_stripped,
    top_genes_selects_a_subset,
    test_cells_cannot_change_fitted_preprocessing,
    cell_ids_are_not_mistaken_for_a_gene,
    synonyms_collapse_to_one_class,
    rare_types_are_dropped,
    shared_labels_only_reduces_to_the_intersection,
    splits_partition_each_client,
    qc_annotations_are_excluded_and_recorded,
    qc_exclusion_can_be_turned_off,
]

REAL_LABEL_CASES = [
    real_labels_give_twelve_classes,
    real_labels_shared_only_gives_four,
    real_labels_every_synonym_fires,
]


def main() -> int:
    for fn in FIXTURE_CASES:
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))

    if LABELS_DIR is not None and LABELS_DIR.is_dir():
        for fn in REAL_LABEL_CASES:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp))
    else:
        print("  skipped real-label cases: set SCRNASEQ_LABELS_DIR")

    for name in PASSED:
        print(f"  ok    {name}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
