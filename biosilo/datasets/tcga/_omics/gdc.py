"""GDC (Genomic Data Commons) API download backend.

Downloads STAR-Counts expression and clinical metadata from the NCI GDC
REST API. Uses only stdlib urllib.
"""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import gdc_project_id, validate_cohorts
from ._cache import atomic_write

_GDC_API = "https://api.gdc.cancer.gov"
_BATCH_SIZE = 30
_MAX_RETRIES = 3


def _gdc_post(endpoint, payload, timeout=120):
    url = f"{_GDC_API}/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _gdc_get(endpoint, params, timeout=60):
    query = "&".join(
        f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items()
    )
    url = f"{_GDC_API}/{endpoint}?{query}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _query_expression_files(project_id):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.project_id",
                "value": [project_id],
            }},
            {"op": "in", "content": {
                "field": "data_type",
                "value": ["Gene Expression Quantification"],
            }},
            {"op": "in", "content": {
                "field": "analysis.workflow_type",
                "value": ["STAR - Counts"],
            }},
        ],
    }

    files = []
    page_from = 0
    page_size = 500

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ("file_id,file_name,"
                       "cases.submitter_id,"
                       "cases.samples.submitter_id,"
                       "cases.samples.sample_type,"
                       "cases.samples.portions.analytes.aliquots.submitter_id"),
            "size": str(page_size),
            "from": str(page_from),
            "format": "json",
        }
        resp = _gdc_get("files", params)
        hits = resp["data"]["hits"]
        if not hits:
            break

        for hit in hits:
            case, sample, aliquot_barcode = _file_relationship(hit)
            files.append({
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "case_id": case["submitter_id"],
                # Count files are aliquot-specific. Using the four-field sample
                # ID here silently overwrites legitimate replicate aliquots.
                "sample_barcode": aliquot_barcode,
                "sample_type": sample["sample_type"],
            })

        total = resp["data"]["pagination"]["total"]
        page_from += page_size
        if page_from >= total:
            break

    return files


def _file_relationship(hit):
    """Return the one case, sample, and aliquot attached to a count file."""
    file_id = hit.get("file_id", "<unknown>")
    cases = hit.get("cases", [])
    if len(cases) != 1:
        raise ValueError(f"GDC file {file_id} links to {len(cases)} cases")
    samples = cases[0].get("samples", [])
    if len(samples) != 1:
        raise ValueError(f"GDC file {file_id} links to {len(samples)} samples")

    aliquots = []
    for portion in samples[0].get("portions", []):
        for analyte in portion.get("analytes", []):
            aliquots.extend(
                item.get("submitter_id")
                for item in analyte.get("aliquots", [])
                if item.get("submitter_id")
            )
    if len(aliquots) != 1:
        raise ValueError(f"GDC file {file_id} links to {len(aliquots)} aliquots")
    return cases[0], samples[0], aliquots[0]


def _download_batch(file_ids, output_dir, batch_num, total_batches):
    print(f"    Batch {batch_num}/{total_batches} "
          f"({len(file_ids)} files) ...", end=" ", flush=True)

    payload = json.dumps({"ids": file_ids}).encode("utf-8")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                f"{_GDC_API}/data",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            extracted = []
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read()

                if len(file_ids) == 1:
                    dest = output_dir / file_ids[0]
                    dest.mkdir(parents=True, exist_ok=True)
                    out_path = dest / f"{file_ids[0]}.tsv"
                    with open(out_path, "wb") as f:
                        f.write(raw)
                    extracted.append(out_path)
                else:
                    tar_data = io.BytesIO(raw)
                    with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
                        tar.extractall(path=str(output_dir))
                        extracted = [
                            output_dir / m.name
                            for m in tar.getmembers() if m.isfile()
                        ]

            print(f"done ({len(extracted)} files)")
            return extracted

        except (EOFError, tarfile.ReadError, urllib.error.URLError,
                ConnectionError, TimeoutError, OSError) as exc:
            if attempt < _MAX_RETRIES:
                wait = 10 * attempt
                print(f"\n      Retry {attempt} ({exc}), "
                      f"waiting {wait}s ...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"\n      FAILED after {_MAX_RETRIES} attempts")
                raise


def _build_counts_matrix(file_infos, download_dir):
    print("  Building counts matrix ...")
    counts_dict = {}
    gene_names = None

    for i, info in enumerate(file_infos):
        fid = info["file_id"]
        fname = info["file_name"]
        barcode = info["sample_barcode"]

        tsv_path = download_dir / fid / fname
        if not tsv_path.exists():
            fid_dir = download_dir / fid
            if fid_dir.exists():
                tsvs = list(fid_dir.glob("*.tsv"))
                if tsvs:
                    tsv_path = tsvs[0]
                else:
                    continue
            else:
                continue

        df = pd.read_csv(tsv_path, sep="\t", comment="#",
                         index_col=0, header=0)
        if "unstranded" in df.columns:
            counts = df["unstranded"]
        else:
            counts = df.iloc[:, 0]

        counts = counts[~counts.index.str.startswith(("N_", "__"))]
        if counts.index.duplicated().any():
            raise ValueError(f"GDC count file {fname} contains duplicate gene IDs")

        if gene_names is None:
            gene_names = counts.index.tolist()
        else:
            expected_genes = set(gene_names)
            actual_genes = set(counts.index)
            if actual_genes != expected_genes:
                raise ValueError(
                    f"GDC count file {fname} has a different gene set")
            counts = counts.reindex(gene_names)

        if barcode in counts_dict:
            raise ValueError(
                f"GDC returned multiple count files for aliquot {barcode}")
        counts_dict[barcode] = counts.values

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(file_infos)} files processed")

    matrix = pd.DataFrame(counts_dict, index=gene_names).T
    expected = {item["sample_barcode"] for item in file_infos}
    actual = set(matrix.index)
    if actual != expected:
        missing = sorted(expected - actual)
        raise RuntimeError(
            f"GDC expression download is missing {len(missing)} samples: "
            + ", ".join(missing[:10]))
    print(f"  Matrix: {matrix.shape[0]} samples x {matrix.shape[1]} genes")
    return matrix


def _sample_case_table(file_infos):
    """Return the validated GDC sample-barcode to case-ID relationship."""
    mapping = pd.DataFrame(
        [{"sample_barcode": item.get("sample_barcode", ""),
          "case_id": item.get("case_id", "")}
         for item in file_infos],
        columns=["sample_barcode", "case_id"],
    )
    if mapping.empty:
        raise ValueError("GDC returned no sample-to-case mappings")
    missing = mapping.eq("").any(axis=1) | mapping.isna().any(axis=1)
    if missing.any():
        raise ValueError("GDC returned a file without both sample and case IDs")
    conflicts = mapping.groupby("sample_barcode")["case_id"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "GDC mapped a sample barcode to multiple cases: "
            + ", ".join(conflicts.index.astype(str)))
    return mapping.drop_duplicates().sort_values(
        ["sample_barcode", "case_id"]).reset_index(drop=True)


def _download_clinical(project_id):
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": [project_id],
        },
    }
    fields = [
        "submitter_id",
        "demographic.gender",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_stage",
    ]

    cases = []
    page_from = 0
    page_size = 500

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "size": str(page_size),
            "from": str(page_from),
            "format": "json",
        }
        resp = _gdc_get("cases", params)
        hits = resp["data"]["hits"]
        if not hits:
            break

        for hit in hits:
            case_id = hit.get("submitter_id", "")
            gender = hit.get("demographic", {}).get("gender", "")
            diagnoses = hit.get("diagnoses", [{}])
            stage = ""
            if diagnoses:
                stage = diagnoses[0].get("ajcc_pathologic_stage", "")
                if not stage:
                    stage = diagnoses[0].get("tumor_stage", "")

            cases.append({
                "case_id": case_id,
                "gender": gender,
                "stage": stage,
            })

        total = resp["data"]["pagination"]["total"]
        page_from += page_size
        if page_from >= total:
            break

    return pd.DataFrame(cases)


def _validate_sample_cases(
    path, expected_samples=None, require_aliquots=False,
):
    mapping = pd.read_csv(path)
    mapping = _sample_case_table(mapping.to_dict(orient="records"))
    if require_aliquots:
        complete = mapping["sample_barcode"].astype(str).str.split("-").str.len() >= 7
        if not complete.all():
            raise ValueError(
                "GDC sample-case data use non-unique sample-level barcodes; "
                "aliquot barcodes are required")
    if expected_samples is not None:
        actual = set(mapping["sample_barcode"])
        expected = set(expected_samples)
        if actual != expected:
            raise ValueError(
                "GDC sample-case mapping does not exactly cover the counts matrix")


def _counts_index(path):
    frame = pd.read_parquet(path, columns=[])
    # A frame loaded with ``columns=[]`` has no columns and therefore reports
    # ``empty`` even when it has rows. Validate the row axis directly.
    if len(frame.index) == 0 or not frame.index.is_unique:
        raise ValueError("GDC counts data must have unique, non-empty sample rows")
    index = frame.index.astype(str)
    if (index.str.strip() == "").any():
        raise ValueError("GDC counts data contain an empty sample barcode")
    return index


def _validate_counts(path, expected_rows: int):
    index = _counts_index(path)
    if len(index) != expected_rows:
        raise ValueError(
            f"GDC counts data have {len(index)} rows; expected {expected_rows}")


def _validate_clinical(path):
    clinical = pd.read_csv(path)
    if "case_id" not in clinical.columns or clinical.empty:
        raise ValueError("GDC clinical data have no case records")
    if clinical["case_id"].isna().any() or clinical["case_id"].duplicated().any():
        raise ValueError("GDC clinical case IDs must be present and unique")


def local_files(
    cohorts: list[str], output_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    """Return prepared GDC source files, refusing missing or invalid data."""
    cohorts = validate_cohorts(cohorts)
    output_dir = Path(output_dir)
    results = {}
    for cohort in cohorts:
        cohort_dir = output_dir / cohort
        paths = {
            "counts": cohort_dir / "counts_matrix.parquet",
            "clinical": cohort_dir / "clinical.csv",
            "sample_cases": cohort_dir / "sample_cases.csv",
        }
        missing = [path.name for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"TCGA GDC source data for {cohort} are incomplete under "
                f"{cohort_dir}; missing {', '.join(missing)}. Run "
                "biosilo.download('TCGA', source='gdc', ...) before generating "
                "the partition."
            )
        try:
            samples = _counts_index(paths["counts"])
            _validate_sample_cases(
                paths["sample_cases"], samples, require_aliquots=True)
            _validate_clinical(paths["clinical"])
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"TCGA GDC source data for {cohort} are invalid under "
                f"{cohort_dir}. Run biosilo.download('TCGA', source='gdc', "
                "...) to prepare them again."
            ) from exc
        results[cohort] = paths
    return results


def download(
    cohorts: list[str],
    output_dir: str | Path = "data/_raw/TCGA/gdc",
) -> dict[str, dict[str, Path]]:
    """Download TCGA expression counts and clinical data from GDC.

    Returns a mapping per cohort containing expression counts, case-level
    clinical data, and the explicit sample-to-case relationship needed to join
    them safely.
    """
    cohorts = validate_cohorts(cohorts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[biosilo.tcga.gdc] Downloading from GDC API")
    print(f"  Cohorts: {', '.join(cohorts)}")
    print(f"  Output: {output_dir}")
    print()

    results = {}

    for cohort in cohorts:
        project_id = gdc_project_id(cohort)
        cohort_dir = output_dir / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)

        counts_file = cohort_dir / "counts_matrix.parquet"
        clinical_file = cohort_dir / "clinical.csv"
        sample_cases_file = cohort_dir / "sample_cases.csv"

        print(f"[{cohort}] {project_id}")

        # Validate the relationship as a pair. Older downloads used a four-field
        # sample barcode, which can collapse distinct aliquot count files.
        file_infos = None
        sample_cases = None
        pair_is_valid = False
        if counts_file.exists() and sample_cases_file.exists():
            try:
                cached_samples = _counts_index(counts_file)
                _validate_sample_cases(
                    sample_cases_file, cached_samples,
                    require_aliquots=True)
            except (OSError, ValueError):
                print("  Cached counts/mapping need refresh")
            else:
                pair_is_valid = True

        if not pair_is_valid:
            print("  Querying expression files ...")
            file_infos = _query_expression_files(project_id)
            print(f"  Found {len(file_infos)} files")
            sample_cases = _sample_case_table(file_infos)

        expected_samples = (
            set(sample_cases["sample_barcode"]) if sample_cases is not None
            else set(_counts_index(counts_file))
        )
        counts_are_reusable = False
        if counts_file.exists():
            try:
                counts_are_reusable = set(_counts_index(counts_file)) == expected_samples
            except (OSError, ValueError):
                counts_are_reusable = False

        if counts_are_reusable:
            print(f"  Counts already prepared: {counts_file.name}")
        else:
            assert file_infos is not None

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                total_batches = (
                    (len(file_infos) + _BATCH_SIZE - 1) // _BATCH_SIZE
                )

                for batch_idx in range(total_batches):
                    start = batch_idx * _BATCH_SIZE
                    end = min(start + _BATCH_SIZE, len(file_infos))
                    batch_ids = [f["file_id"] for f in file_infos[start:end]]
                    _download_batch(
                        batch_ids, tmpdir, batch_idx + 1, total_batches,
                    )

                matrix = _build_counts_matrix(file_infos, tmpdir)

            atomic_write(
                counts_file,
                lambda temporary: matrix.to_parquet(temporary),
                lambda temporary: _validate_counts(temporary, len(matrix)),
            )
            size_mb = counts_file.stat().st_size / (1024 * 1024)
            print(f"  Saved: {counts_file.name} ({size_mb:.1f} MB)")

        if sample_cases is not None:
            expected_samples = _counts_index(counts_file)
            atomic_write(
                sample_cases_file,
                lambda temporary: sample_cases.to_csv(temporary, index=False),
                lambda temporary: _validate_sample_cases(
                    temporary, expected_samples, require_aliquots=True),
            )
            print(f"  Saved: {sample_cases_file.name} "
                  f"({len(sample_cases)} mappings)")

        # Clinical data
        clinical_is_valid = False
        if clinical_file.exists():
            try:
                _validate_clinical(clinical_file)
            except (OSError, ValueError):
                print("  Cached clinical data need refresh")
            else:
                clinical_is_valid = True
        if clinical_is_valid:
            print(f"  Clinical already prepared: {clinical_file.name}")
        else:
            print(f"  Downloading clinical data ...")
            clinical = _download_clinical(project_id)
            atomic_write(
                clinical_file,
                lambda temporary: clinical.to_csv(temporary, index=False),
                _validate_clinical,
            )
            print(f"  Saved: {clinical_file.name} ({len(clinical)} records)")

        results[cohort] = {
            "counts": counts_file,
            "clinical": clinical_file,
            "sample_cases": sample_cases_file,
        }
        print()

    return results
