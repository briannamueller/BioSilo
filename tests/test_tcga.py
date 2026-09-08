"""TCGA label, gene, hospital, grouping, and split tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import biosilo
import fixtures_tcga as fx
from biosilo.datasets import tcga

PASSED, FAILED = [], []


def case(fn):
    def run(root):
        try:
            fn(root)
        except Exception as exc:  # noqa: BLE001
            FAILED.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            PASSED.append(fn.__name__)
    return run


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


# ── federation ────────────────────────────────────────────────────────────────

@case
def source_defaults_under_the_partition_root(root):
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


@case
def download_is_an_explicit_step(root):
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


@case
def generation_refuses_missing_source_without_downloading(root):
    data_root = root / "data"
    try:
        biosilo.generate(
            "TCGA", root=data_root, cohorts=fx.COHORTS,
            min_samples=5, top_genes=10,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        assert "biosilo.download" in message
        assert str(data_root / "_raw" / "TCGA" / "xena") in message
    else:
        raise AssertionError("TCGA generation accepted missing source data")


@case
def clients_are_hospitals_not_sites(root):
    """Map barcode TSS codes to hospital names."""
    p = _generate(root)
    assert "Indivumed" in p.client_ids, p.client_ids
    assert "Fox Chase Cancer Center" in p.client_ids, p.client_ids
    # no raw TSS fallback names leaked through
    assert not any(c.startswith("TSS_") for c in p.client_ids), p.client_ids


@case
def small_hospitals_are_dropped(root):
    p = _generate(root)
    # Mayo contributed 3 patients, below min_samples=5
    assert "Mayo Clinic - Rochester" not in p.client_ids, p.client_ids


# ── grouping ──────────────────────────────────────────────────────────────────

@case
def a_patients_samples_never_straddle_a_split(root):
    p = _generate(root)
    assert p.has_groups and p.group_unit == "patient", p.group_unit
    assert all(client["metadata"]["preprocessing_protocol"] == "inductive"
               for client in p.manifest["clients"])
    for cid in range(p.num_clients):
        _, _, gtr = p.client(cid, "train")
        _, _, gte = p.client(cid, "test")
        assert np.intersect1d(gtr, gte).size == 0, f"client {cid} leaks a patient"


@case
def the_repeat_patient_is_actually_present(root):
    """Confirm that the fixture exercises repeated-patient grouping."""
    p = _generate(root)
    idx = p.client_ids.index("Indivumed")
    meta = p.manifest["clients"][idx]["metadata"]
    assert meta["n_patients"] < meta["n_samples"], meta


@case
def patient_id_is_the_first_three_barcode_fields(root):
    assert tcga._patient("TCGA-A7-A0CG-01A-11R-A00Z-07") == "TCGA-A7-A0CG"
    assert tcga._patient("TCGA-A7-A0CG-06A") == "TCGA-A7-A0CG"


# ── genes ─────────────────────────────────────────────────────────────────────

@case
def silent_genes_are_dropped_before_variance_ranking(root):
    p = _generate(root, top_genes=0)
    kept = p.inputs[0]["shape"][0]
    # GENE0 is always zero and GENE1 is nonzero in only 5% of samples
    assert kept == len(fx.GENES) - 2, (kept, len(fx.GENES))


@case
def top_genes_caps_the_feature_count(root):
    p = _generate(root, top_genes=10)
    assert p.inputs[0]["shape"] == [10], p.inputs[0]
    X, _, _ = p.client(0, "train")
    assert X.shape[1] == 10, X.shape


@case
def test_samples_cannot_change_fitted_gene_set(root):
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


@case
def gdc_source_default_removes_library_size(root):
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


# ── tasks ─────────────────────────────────────────────────────────────────────

@case
def cancer_type_labels_match_the_cohorts(root):
    p = _generate(root)
    assert p.num_classes == len(fx.COHORTS), p.num_classes


@case
def stage_task_is_binary(root):
    p = _generate(root, task="stage")
    assert p.num_classes == 2, p.num_classes


@case
def gdc_clinical_rows_follow_samples_through_case_ids(root):
    """Case-level clinical values must be expanded to each mapped sample."""
    expression, sample_cases, clinical = fx.build_gdc()
    aligned = tcga._align_gdc_clinical(
        expression, sample_cases, clinical, "LUAD")

    assert aligned.index.tolist() == expression.index.tolist()
    assert aligned["case_id"].tolist() == [
        "TCGA-05-0500", "TCGA-05-0500", "TCGA-21-2101"]
    assert aligned["stage"].tolist() == ["Stage IV", "Stage IV", "Stage II"]
    assert aligned["cohort"].eq("LUAD").all()


@case
def gdc_download_contract_aligns_file_shaped_tables(root):
    """Exercise the parquet/CSV path contract returned by the GDC backend."""
    fx.write_gdc(root / "LUAD")
    expression, phenotype = tcga._read_source(
        tcga.Params(cohorts=("LUAD",), task="stage", source="gdc"), root)

    assert phenotype.index.equals(expression.index)
    assert phenotype["case_id"].tolist() == [
        "TCGA-05-0500", "TCGA-05-0500", "TCGA-21-2101"]
    assert phenotype["stage"].tolist() == ["Stage IV", "Stage IV", "Stage II"]


@case
def gdc_refuses_one_sample_mapped_to_two_cases(root):
    expression, sample_cases, clinical = fx.build_gdc()
    conflict = sample_cases.iloc[[0]].copy()
    conflict["case_id"] = "TCGA-99-9999"
    sample_cases = __import__("pandas").concat(
        [sample_cases, conflict], ignore_index=True)
    try:
        tcga._align_gdc_clinical(expression, sample_cases, clinical, "LUAD")
    except ValueError as exc:
        assert "multiple cases" in str(exc), str(exc)
    else:
        raise AssertionError("one sample mapped to two cases was accepted")


@case
def gdc_count_files_are_aligned_by_gene_id(root):
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


@case
def gdc_expression_files_use_their_unique_aliquot_barcode(root):
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


@case
def gdc_rejects_old_sample_level_mapping_caches(root):
    import pandas as pd
    from biosilo.datasets.tcga._omics import gdc

    mapping = root / "sample_cases.csv"
    pd.DataFrame([{
        "sample_barcode": "TCGA-05-0500-01A",
        "case_id": "TCGA-05-0500",
    }]).to_csv(mapping, index=False)
    try:
        gdc._validate_sample_cases(mapping, require_aliquots=True)
    except ValueError as exc:
        assert "aliquot" in str(exc)
    else:
        raise AssertionError("sample-level GDC cache was accepted as aliquot-unique")


@case
def gdc_counts_index_reads_rows_from_columnless_parquet(root):
    import pandas as pd
    from biosilo.datasets.tcga._omics import gdc

    path = root / "counts.parquet"
    barcodes = [
        "TCGA-05-0500-01A-11R-A00Z-07",
        "TCGA-21-2101-01A-11R-A00Z-07",
    ]
    pd.DataFrame({"gene": [1, 2]}, index=barcodes).to_parquet(path)
    assert gdc._counts_index(path).tolist() == barcodes


@case
def gdc_refuses_duplicate_count_files_for_one_sample(root):
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
    try:
        gdc._build_counts_matrix(infos, root)
    except ValueError as exc:
        assert "multiple count files for aliquot" in str(exc)
    else:
        raise AssertionError("duplicate GDC sample count files were accepted")


# ── cache integrity ──────────────────────────────────────────────────────────

@case
def atomic_cache_write_preserves_existing_file_on_failure(root):
    from biosilo.datasets.tcga._omics._cache import atomic_write

    destination = root / "artifact.bin"
    destination.write_bytes(b"complete-old-cache")

    def fail_after_partial_write(temporary):
        temporary.write_bytes(b"partial-new-cache")
        raise RuntimeError("interrupted")

    try:
        atomic_write(destination, fail_after_partial_write)
    except RuntimeError as exc:
        assert "interrupted" in str(exc)
    else:
        raise AssertionError("interrupted cache overwrite succeeded")

    assert destination.read_bytes() == b"complete-old-cache"
    assert not list(root.glob(".artifact.bin.building-*.tmp"))


@case
def interrupted_xena_download_never_looks_cached(root):
    from biosilo.datasets.tcga._omics import xena

    destination = root / "expression.gz"
    original = xena.urllib.request.urlretrieve

    def interrupted(url, temporary):
        Path(temporary).write_bytes(b"partial")
        raise OSError("connection lost")

    xena.urllib.request.urlretrieve = interrupted
    try:
        try:
            xena._download_file("https://example.invalid/expression", destination)
        except OSError as exc:
            assert "connection lost" in str(exc)
        else:
            raise AssertionError("interrupted download succeeded")
    finally:
        xena.urllib.request.urlretrieve = original

    assert not destination.exists()
    assert not list(root.glob(".expression.gz.building-*.tmp"))


@case
def interrupted_gdc_parquet_write_leaves_no_counts_cache(root):
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
        try:
            gdc.download(["LUAD"], root)
        except RuntimeError as exc:
            assert "disk interrupted" in str(exc)
        else:
            raise AssertionError("interrupted GDC cache write succeeded")
    finally:
        (gdc._query_expression_files, gdc._download_batch,
         gdc._build_counts_matrix, pd.DataFrame.to_parquet) = originals

    cohort = root / "LUAD"
    assert not (cohort / "counts_matrix.parquet").exists()
    assert not (cohort / "sample_cases.csv").exists()
    assert not list(cohort.glob(".*.building-*.tmp"))


# ── refusals ──────────────────────────────────────────────────────────────────

@case
def single_cohort_cancer_type_is_refused(root):
    try:
        _generate(root, cohorts=("LUAD",))
    except ValueError as exc:
        assert "at least two cohorts" in str(exc)
    else:
        raise AssertionError("single-cohort cancer_type was accepted")


@case
def survival_from_gdc_is_refused(root):
    try:
        _generate(root, task="survival", source="gdc")
    except ValueError as exc:
        assert "no survival endpoints" in str(exc)
    else:
        raise AssertionError("survival+gdc was accepted")


@case
def unknown_cohort_is_refused(root):
    try:
        _generate(root, cohorts=("LUAD", "NOTACOHORT"))
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("unknown cohort accepted")


@case
def partition_label_uses_task_and_seed(root):
    params = tcga.Params(
        cohorts=("BRCA", "COAD"), task="stage", source="gdc",
        top_genes=500, seed=6,
    )
    assert tcga.label(params) == "stage_s6"


CASES = [
    source_defaults_under_the_partition_root,
    download_is_an_explicit_step,
    generation_refuses_missing_source_without_downloading,
    clients_are_hospitals_not_sites,
    small_hospitals_are_dropped,
    a_patients_samples_never_straddle_a_split,
    the_repeat_patient_is_actually_present,
    patient_id_is_the_first_three_barcode_fields,
    silent_genes_are_dropped_before_variance_ranking,
    top_genes_caps_the_feature_count,
    test_samples_cannot_change_fitted_gene_set,
    gdc_source_default_removes_library_size,
    cancer_type_labels_match_the_cohorts,
    stage_task_is_binary,
    gdc_clinical_rows_follow_samples_through_case_ids,
    gdc_download_contract_aligns_file_shaped_tables,
    gdc_refuses_one_sample_mapped_to_two_cases,
    gdc_count_files_are_aligned_by_gene_id,
    gdc_expression_files_use_their_unique_aliquot_barcode,
    gdc_rejects_old_sample_level_mapping_caches,
    gdc_counts_index_reads_rows_from_columnless_parquet,
    gdc_refuses_duplicate_count_files_for_one_sample,
    atomic_cache_write_preserves_existing_file_on_failure,
    interrupted_xena_download_never_looks_cached,
    interrupted_gdc_parquet_write_leaves_no_counts_cache,
    single_cohort_cancer_type_is_refused,
    survival_from_gdc_is_refused,
    unknown_cohort_is_refused,
    partition_label_uses_task_and_seed,
]


def main() -> int:
    for fn in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))

    for name in PASSED:
        print(f"  ok    {name}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
