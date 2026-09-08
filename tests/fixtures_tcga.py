"""Create small TCGA expression and phenotype tables.

The fixture replaces network downloads while retaining source TSS codes for
label encoding, gene filtering, hospital mapping, grouping, and split tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: TSS code, expected hospital, and contributed patient count.
SITES = [("05", "Indivumed", 14), ("21", "Fox Chase Cancer Center", 11),
         ("22", "Mayo Clinic - Rochester", 3)]

COHORTS = ("LUAD", "LUSC")
GENES = [f"GENE{i}" for i in range(40)]

#: One patient at site 05 contributes a primary and a metastatic sample.
REPEAT_PATIENT = ("05", 0)


def build(seed: int = 0):
    """Return (expression, phenotype), shaped as the download layer returns them."""
    rng = np.random.default_rng(seed)

    barcodes, cohorts = [], []
    for tss, _, n_patients in SITES:
        for k in range(n_patients):
            patient = f"{tss}{k:02d}"
            cohort = COHORTS[k % len(COHORTS)]
            barcodes.append(f"TCGA-{tss}-{patient}-01A-11R-A00Z-07")
            cohorts.append(cohort)
            if (tss, k) == REPEAT_PATIENT:
                # same patient, second sample
                barcodes.append(f"TCGA-{tss}-{patient}-06A-11R-A00Z-07")
                cohorts.append(cohort)

    n = len(barcodes)
    values = rng.random((n, len(GENES))).astype(np.float32) * 10

    # Two genes are near-silent, so the nonzero-fraction filter has something to
    # remove; one is constant, so the variance filter ranks it last.
    values[:, 0] = 0.0
    values[: int(n * 0.95), 1] = 0.0
    values[:, 2] = 5.0

    expression = pd.DataFrame(values, index=barcodes, columns=GENES)
    phenotype = pd.DataFrame(
        {"cohort": cohorts,
         "stage": ["Stage IV" if i % 3 == 0 else "Stage II" for i in range(n)]},
        index=barcodes,
    )
    return expression, phenotype


def build_gdc(seed: int = 7):
    """Return GDC-shaped expression, sample-case mapping, and clinical rows.

    The three tables intentionally use different orders. One case contributes
    two expression samples, exercising the valid one-case-to-many-samples join.
    """
    rng = np.random.default_rng(seed)
    samples = [
        "TCGA-05-0500-01A-11R-A00Z-07",
        "TCGA-05-0500-06A-11R-A00Z-07",
        "TCGA-21-2101-01A-11R-A00Z-07",
    ]
    expression = pd.DataFrame(
        rng.random((3, 4)), index=samples,
        columns=["ENSG0001", "ENSG0002", "ENSG0003", "ENSG0004"],
    )
    sample_cases = pd.DataFrame([
        {"sample_barcode": samples[2], "case_id": "TCGA-21-2101"},
        {"sample_barcode": samples[1], "case_id": "TCGA-05-0500"},
        {"sample_barcode": samples[0], "case_id": "TCGA-05-0500"},
    ])
    clinical = pd.DataFrame([
        {"case_id": "TCGA-21-2101", "gender": "male", "stage": "Stage II"},
        {"case_id": "TCGA-05-0500", "gender": "female", "stage": "Stage IV"},
    ])
    return expression, sample_cases, clinical


def write_gdc(root, cohort: str = "LUAD"):
    """Write the GDC-shaped fixture and return the backend's path contract."""
    root = __import__("pathlib").Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expression, sample_cases, clinical = build_gdc()
    counts_path = root / "counts_matrix.parquet"
    clinical_path = root / "clinical.csv"
    sample_cases_path = root / "sample_cases.csv"
    expression.to_parquet(counts_path)
    clinical.to_csv(clinical_path, index=False)
    sample_cases.to_csv(sample_cases_path, index=False)
    return {cohort: {
        "counts": counts_path,
        "clinical": clinical_path,
        "sample_cases": sample_cases_path,
    }}
