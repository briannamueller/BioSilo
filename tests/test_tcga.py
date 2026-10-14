"""TCGA label, gene, hospital, grouping, and split tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import biosilo
import fixtures_tcga as fx
from biosilo.datasets import tcga


def _generate(root, **overrides):
    """Generate with local source loading swapped for the fixture."""
    original = tcga._read_source
    tcga._read_source = lambda p, source: fx.build()
    try:
        kwargs = dict(cohorts=fx.COHORTS, task="cancer_type", top_genes=10,
                      min_samples=5, train_ratio=0.75, seed=1,
                      source_dir="/tmp")
        kwargs.update(overrides)
        out = biosilo.generate("TCGA", root=root / "data", **kwargs)
        return biosilo.load("TCGA", root=root / "data", partition=out.name)
    finally:
        tcga._read_source = original


# Federation

def test_source_defaults_under_the_partition_root(root):
    original = tcga._read_source
    observed = []

    def fixture_source(p, source):
        observed.append(Path(source))
        return fx.build()

    tcga._read_source = fixture_source
    data_root = root / "data"
    try:
        biosilo.generate(
            "TCGA", root=data_root, cohorts=fx.COHORTS,
            task="cancer_type", top_genes=10, min_samples=5,
            train_ratio=0.75, seed=91,
        )
    finally:
        tcga._read_source = original

    assert observed == [data_root / "_raw" / "TCGA" / "xena"], observed


def test_download_is_an_explicit_step(root):
    from biosilo.datasets.tcga._omics import xena

    original = xena.download
    observed = []

    def fixture_download(cohorts, output_dir):
        observed.append((tuple(cohorts), Path(output_dir)))

    xena.download = fixture_download
    data_root = root / "data"
    try:
        destination = biosilo.download(
            "TCGA", root=data_root, source="xena", cohorts=fx.COHORTS)
    finally:
        xena.download = original

    expected = data_root / "_raw" / "TCGA" / "xena"
    assert destination == expected
    assert observed == [(fx.COHORTS, expected)], observed


def test_generation_refuses_missing_source_without_downloading(root):
    data_root = root / "data"
    with pytest.raises(FileNotFoundError) as exc_info:
        biosilo.generate(
            "TCGA", root=data_root, cohorts=fx.COHORTS,
            min_samples=5, top_genes=10,
        )
    message = str(exc_info.value)
    assert "biosilo.download" in message
    assert str(data_root / "_raw" / "TCGA" / "xena") in message


def test_clients_are_hospitals_not_sites(root):
    """Map barcode TSS codes to hospital names."""
    p = _generate(root)
    assert "Indivumed" in p.client_ids, p.client_ids
    assert "Fox Chase Cancer Center" in p.client_ids, p.client_ids
    # no raw TSS fallback names leaked through
    assert not any(c.startswith("TSS_") for c in p.client_ids), p.client_ids


def test_small_hospitals_are_dropped(root):
    p = _generate(root)
    # Mayo contributed 3 patients, below min_samples=5
    assert "Mayo Clinic - Rochester" not in p.client_ids, p.client_ids


# Grouping

def test_a_patients_samples_never_straddle_a_split(root):
    p = _generate(root)
    assert p.has_groups and p.group_unit == "patient", p.group_unit
    assert all(client["metadata"]["preprocessing_protocol"] == "inductive"
               for client in p.manifest["clients"])
    for cid in range(p.num_clients):
        _, _, gtr = p.client(cid, "train")
        _, _, gte = p.client(cid, "test")
        assert np.intersect1d(gtr, gte).size == 0, f"client {cid} leaks a patient"


def test_fixture_contains_a_repeat_patient(root):
    """Confirm that the fixture exercises repeated-patient grouping."""
    p = _generate(root)
    idx = p.client_ids.index("Indivumed")
    meta = p.manifest["clients"][idx]["metadata"]
    assert meta["n_patients"] < meta["n_samples"], meta


def test_patient_id_uses_the_first_three_barcode_fields():
    assert tcga._patient("TCGA-A7-A0CG-01A-11R-A00Z-07") == "TCGA-A7-A0CG"
    assert tcga._patient("TCGA-A7-A0CG-06A") == "TCGA-A7-A0CG"


# Genes

def test_silent_genes_are_dropped_before_variance_ranking(root):
    p = _generate(root, top_genes=0)
    kept = p.inputs[0]["shape"][0]
    # GENE0 is always zero and GENE1 is nonzero in only 5% of samples
    assert kept == len(fx.GENES) - 2, (kept, len(fx.GENES))


def test_top_genes_caps_the_feature_count(root):
    p = _generate(root, top_genes=10)
    assert p.inputs[0]["shape"] == [10], p.inputs[0]
    X, _, _ = p.client(0, "train")
    assert X.shape[1] == 10, X.shape


def test_samples_cannot_change_fitted_gene_set():
    """Variance present only in test samples must not drive gene selection."""
    import pandas as pd

    expression = pd.DataFrame(
        {
            "TRAIN_SIGNAL": [0.0, 10.0, 5.0, 5.0],
            "TEST_SIGNAL": [2.0, 2.0, 0.0, 1000.0],
        },
        index=["train-a", "train-b", "test-a", "test-b"],
    )
    selected = tcga._fit_genes(
        expression, ["train-a", "train-b"], 0.0, 1)
    changed = expression.copy()
    changed.loc[["test-a", "test-b"], "TEST_SIGNAL"] *= 1_000_000
    selected_after = tcga._fit_genes(
        changed, ["train-a", "train-b"], 0.0, 1)
    assert selected.tolist() == ["TRAIN_SIGNAL"]
    assert selected_after.tolist() == selected.tolist()


def test_gdc_source_default_removes_library_size():
    """Proportional GDC count vectors should match after CPM-log2."""
    import pandas as pd

    counts = pd.DataFrame(
        [[1.0, 3.0], [10.0, 30.0]],
        index=["sample-a", "sample-b"], columns=["g1", "g2"])
    transformed = tcga._transform_expression(
        counts, source="gdc", requested="source_default")
    assert np.allclose(transformed.iloc[0], transformed.iloc[1])
    assert np.array_equal(
        tcga._transform_expression(counts, "gdc", "none").to_numpy(),
        counts.to_numpy())


# Tasks

def test_cancer_type_labels_match_the_cohorts(root):
    p = _generate(root)
    assert p.num_classes == len(fx.COHORTS), p.num_classes


def test_stage_task_is_binary(root):
    p = _generate(root, task="stage")
    assert p.num_classes == 2, p.num_classes


def test_gdc_clinical_rows_align_by_case_id():
    """Case-level clinical values must be expanded to each mapped sample."""
    expression, sample_cases, clinical = fx.build_gdc()
    aligned = tcga._align_gdc_clinical(
        expression, sample_cases, clinical, "LUAD")

    assert aligned.index.tolist() == expression.index.tolist()
    assert aligned["case_id"].tolist() == [
        "TCGA-05-0500", "TCGA-05-0500", "TCGA-21-2101"]
    assert aligned["stage"].tolist() == ["Stage IV", "Stage IV", "Stage II"]
    assert aligned["cohort"].eq("LUAD").all()


def test_gdc_download_contract_aligns_file_shaped_tables(root):
    """Exercise the parquet/CSV path contract returned by the GDC backend."""
    fx.write_gdc(root / "LUAD")
    expression, phenotype = tcga._read_source(
        tcga.Params(cohorts=("LUAD",), task="stage", source="gdc"), root)

    assert phenotype.index.equals(expression.index)
    assert phenotype["case_id"].tolist() == [
        "TCGA-05-0500", "TCGA-05-0500", "TCGA-21-2101"]
    assert phenotype["stage"].tolist() == ["Stage IV", "Stage IV", "Stage II"]


def test_gdc_rejects_conflicting_case_mapping():
    expression, sample_cases, clinical = fx.build_gdc()
    conflict = sample_cases.iloc[[0]].copy()
    conflict["case_id"] = "TCGA-99-9999"
    sample_cases = __import__("pandas").concat(
        [sample_cases, conflict], ignore_index=True)
    with pytest.raises(ValueError, match="multiple cases"):
        tcga._align_gdc_clinical(expression, sample_cases, clinical, "LUAD")


def test_gdc_count_files_are_aligned_by_gene_id(root):
    """File row order must not determine which count belongs to which gene."""
    import pandas as pd
    from biosilo.datasets.tcga._omics import gdc

    infos = []
    rows = {
        "file-a": [("g1", 1), ("g2", 2)],
        "file-b": [("g2", 20), ("g1", 10)],
    }
    for file_id, sample in (("file-a", "sample-a"), ("file-b", "sample-b")):
        directory = root / file_id
        directory.mkdir()
        pd.DataFrame(rows[file_id], columns=["gene", "unstranded"]).set_index(
            "gene").to_csv(directory / "counts.tsv", sep="\t")
        infos.append({
            "file_id": file_id, "file_name": "counts.tsv",
            "sample_barcode": sample,
        })

    matrix = gdc._build_counts_matrix(infos, root)
    assert matrix.columns.tolist() == ["g1", "g2"]
    assert matrix.loc["sample-b"].tolist() == [10, 20]


def test_gdc_expression_files_use_unique_aliquot_barcodes():
    from biosilo.datasets.tcga._omics import gdc

    hit = {
        "file_id": "file-a",
        "cases": [{
            "submitter_id": "TCGA-05-0500",
            "samples": [{
                "submitter_id": "TCGA-05-0500-01A",
                "sample_type": "Primary Tumor",
                "portions": [{"analytes": [{"aliquots": [{
                    "submitter_id": "TCGA-05-0500-01A-11R-A00Z-07",
                }]}]}],
            }],
        }],
    }
    case, sample, aliquot = gdc._file_relationship(hit)
    assert case["submitter_id"] == "TCGA-05-0500"
    assert sample["submitter_id"] == "TCGA-05-0500-01A"
    assert aliquot == "TCGA-05-0500-01A-11R-A00Z-07"


def test_gdc_rejects_old_sample_level_mapping_caches(root):
    import pandas as pd
    from biosilo.datasets.tcga._omics import gdc

    mapping = root / "sample_cases.csv"
    pd.DataFrame([{
        "sample_barcode": "TCGA-05-0500-01A",
        "case_id": "TCGA-05-0500",
    }]).to_csv(mapping, index=False)
    with pytest.raises(ValueError, match="aliquot"):
        gdc._validate_sample_cases(mapping, require_aliquots=True)


def test_gdc_counts_index_reads_rows_from_columnless_parquet(root):
    import pandas as pd
    from biosilo.datasets.tcga._omics import gdc

    path = root / "counts.parquet"
    barcodes = [
        "TCGA-05-0500-01A-11R-A00Z-07",
        "TCGA-21-2101-01A-11R-A00Z-07",
    ]
    pd.DataFrame({"gene": [1, 2]}, index=barcodes).to_parquet(path)
    assert gdc._counts_index(path).tolist() == barcodes


def test_gdc_refuses_duplicate_count_files_for_one_sample(root):
    import pandas as pd
    from biosilo.datasets.tcga._omics import gdc

    infos = []
    for file_id in ("file-a", "file-b"):
        directory = root / file_id
        directory.mkdir()
        pd.DataFrame({"unstranded": [1]}, index=["g1"]).to_csv(
            directory / "counts.tsv", sep="\t")
        infos.append({
            "file_id": file_id, "file_name": "counts.tsv",
            "sample_barcode": "same-sample",
        })
    with pytest.raises(ValueError, match="multiple count files for aliquot"):
        gdc._build_counts_matrix(infos, root)


# Cache integrity

def test_atomic_cache_write_preserves_existing_file_on_failure(root):
    from biosilo.datasets.tcga._omics._cache import atomic_write

    destination = root / "artifact.bin"
    destination.write_bytes(b"complete-old-cache")

    def fail_after_partial_write(temporary):
        temporary.write_bytes(b"partial-new-cache")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_write(destination, fail_after_partial_write)

    assert destination.read_bytes() == b"complete-old-cache"
    assert not list(root.glob(".artifact.bin.building-*.tmp"))


def test_interrupted_xena_download_never_looks_cached(root):
    from biosilo.datasets.tcga._omics import xena

    destination = root / "expression.gz"
    original = xena.urllib.request.urlretrieve

    def interrupted(url, temporary):
        Path(temporary).write_bytes(b"partial")
        raise OSError("connection lost")

    xena.urllib.request.urlretrieve = interrupted
    try:
        with pytest.raises(OSError, match="connection lost"):
            xena._download_file("https://example.invalid/expression", destination)
    finally:
        xena.urllib.request.urlretrieve = original

    assert not destination.exists()
    assert not list(root.glob(".expression.gz.building-*.tmp"))


def test_interrupted_gdc_parquet_write_leaves_no_counts_cache(root):
    import pandas as pd
    from biosilo.datasets.tcga._omics import gdc

    file_infos = [{
        "file_id": "file-1", "file_name": "file.tsv",
        "case_id": "TCGA-05-0500",
        "sample_barcode": "TCGA-05-0500-01A",
        "sample_type": "Primary Tumor",
    }]
    matrix = pd.DataFrame(
        [[1, 2]], index=["TCGA-05-0500-01A"], columns=["g1", "g2"])

    originals = (
        gdc._query_expression_files, gdc._download_batch,
        gdc._build_counts_matrix, pd.DataFrame.to_parquet,
    )
    gdc._query_expression_files = lambda project: file_infos
    gdc._download_batch = lambda *args, **kwargs: None
    gdc._build_counts_matrix = lambda infos, directory: matrix

    def interrupted(self, temporary, *args, **kwargs):
        Path(temporary).write_bytes(b"partial parquet")
        raise RuntimeError("disk interrupted")

    pd.DataFrame.to_parquet = interrupted
    try:
        with pytest.raises(RuntimeError, match="disk interrupted"):
            gdc.download(["LUAD"], root)
    finally:
        (gdc._query_expression_files, gdc._download_batch,
         gdc._build_counts_matrix, pd.DataFrame.to_parquet) = originals

    cohort = root / "LUAD"
    assert not (cohort / "counts_matrix.parquet").exists()
    assert not (cohort / "sample_cases.csv").exists()
    assert not list(cohort.glob(".*.building-*.tmp"))


# Refusals

def test_single_cohort_cancer_type_is_refused(root):
    with pytest.raises(ValueError, match="at least two cohorts"):
        _generate(root, cohorts=("LUAD",))


def test_survival_from_gdc_is_refused(root):
    with pytest.raises(ValueError, match="no survival endpoints"):
        _generate(root, task="survival", source="gdc")


def test_unknown_cohort_is_refused(root):
    with pytest.raises((ValueError, KeyError)):
        _generate(root, cohorts=("LUAD", "NOTACOHORT"))


def test_partition_label_uses_task_and_seed():
    params = tcga.Params(
        cohorts=("BRCA", "COAD"), task="stage", source="gdc",
        top_genes=500, seed=6,
    )
    assert tcga.label(params) == "stage_s6"
