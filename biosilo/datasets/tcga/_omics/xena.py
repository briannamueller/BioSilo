"""UCSC Xena download backend: pre-normalized log2(TPM+1) matrices."""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import (
    TCGA_COHORTS, TCGA_COHORT_TO_DISEASE, TCGA_DISEASE_TO_COHORT,
    validate_cohorts,
)
from ._cache import atomic_write

_TOIL_HUB = "https://toil.xenahubs.net/download"
_EXPRESSION_URL = f"{_TOIL_HUB}/tcga_RSEM_gene_tpm.gz"
_PHENOTYPE_URL = f"{_TOIL_HUB}/TcgaTargetGTEX_phenotype.txt.gz"
_SURVIVAL_URL = f"{_TOIL_HUB}/TCGA_survival_data"


def local_files(
    output_dir: str | Path, require_survival: bool = False,
) -> dict[str, Path]:
    """Return prepared Xena source files, refusing an incomplete source."""
    output_dir = Path(output_dir)
    paths = {
        "expression_path": output_dir / "tcga_RSEM_gene_tpm.gz",
        "phenotype_path": output_dir / "TcgaTargetGTEX_phenotype.txt.gz",
        "survival_path": output_dir / "TCGA_survival_data.tsv",
        "output_dir": output_dir,
    }
    required = [paths["expression_path"], paths["phenotype_path"]]
    if require_survival:
        required.append(paths["survival_path"])
    missing = [
        path.name for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"TCGA Xena source data are incomplete under {output_dir}; missing "
            f"{', '.join(missing)}. Run biosilo.download('TCGA', "
            "source='xena', ...) before generating the partition."
        )
    return paths


def _download_file(url, dest, desc=""):
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name} ...")
    print(f"    URL: {url}")
    atomic_write(dest, lambda temporary: urllib.request.urlretrieve(
        url, str(temporary)))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"    Done ({size_mb:.1f} MB)")


def download(
    cohorts: list[str],
    output_dir: str | Path = "data/_raw/TCGA/xena",
) -> dict[str, Path]:
    """Download expression, phenotype, and survival files from Xena TOIL hub."""
    cohorts = validate_cohorts(cohorts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "expression_path": output_dir / "tcga_RSEM_gene_tpm.gz",
        "phenotype_path": output_dir / "TcgaTargetGTEX_phenotype.txt.gz",
        "survival_path": output_dir / "TCGA_survival_data.tsv",
        "output_dir": output_dir,
    }

    print("[biosilo.tcga.xena] Downloading TCGA data from Xena TOIL hub")
    print(f"  Cohorts: {', '.join(cohorts)}")
    print(f"  Output: {output_dir}")
    print()

    _download_file(
        _EXPRESSION_URL, paths["expression_path"], "expression matrix (~741 MB)")
    _download_file(
        _PHENOTYPE_URL, paths["phenotype_path"], "phenotype data (~136 KB)")
    _download_file(
        _SURVIVAL_URL, paths["survival_path"], "survival data (~407 KB)")

    return local_files(output_dir, require_survival=True)


def load_expression(
    expression_path: Path,
    cohorts: list[str],
    phenotype_path: Path,
) -> pd.DataFrame:
    """Load expression matrix filtered to the given cohorts (samples x genes)."""
    cohorts = validate_cohorts(cohorts)

    print("  Loading phenotype data for sample filtering ...")
    pheno = pd.read_csv(phenotype_path, sep="\t", index_col=0,
                        encoding="latin-1")

    tcga_mask = pheno.index.str.startswith("TCGA-")
    pheno = pheno.loc[tcga_mask]

    disease_col = None
    for col in ["detailed_category", "primary disease or tissue"]:
        if col in pheno.columns:
            disease_col = col
            break

    if disease_col:
        target_diseases = {
            TCGA_COHORT_TO_DISEASE[c] for c in cohorts
            if c in TCGA_COHORT_TO_DISEASE
        }
        cohort_mask = pheno[disease_col].isin(target_diseases)
        cohort_samples = pheno.index[cohort_mask].tolist()
    else:
        # Fallback: use all TCGA samples
        cohort_samples = pheno.index.tolist()

    # Primary tumor samples only (sample code 01-09)
    tumor_samples = []
    for s in cohort_samples:
        parts = s.split("-")
        if len(parts) >= 4:
            try:
                sample_code = int(parts[3][:2])
                if sample_code < 10:
                    tumor_samples.append(s)
            except ValueError:
                continue

    print(f"  Found {len(tumor_samples)} tumor samples across "
          f"{len(cohorts)} cohorts")

    print("  Loading expression matrix (this may take a minute) ...")
    expr = pd.read_csv(expression_path, sep="\t", index_col=0)
    available = [s for s in tumor_samples if s in expr.columns]
    print(f"  {len(available)} samples with expression data")

    expr_filtered = expr[available].T  # -> samples x genes
    expr_filtered.index.name = "sample_id"

    return expr_filtered


def load_phenotype(
    phenotype_path: Path,
    samples: list[str] | None = None,
) -> pd.DataFrame:
    """Load phenotype data, returning cohort/gender/sample_type per sample."""
    pheno = pd.read_csv(phenotype_path, sep="\t", index_col=0,
                        encoding="latin-1")

    pheno = pheno.loc[pheno.index.str.startswith("TCGA-")]

    if samples is not None:
        pheno = pheno.loc[pheno.index.isin(samples)]

    result = pd.DataFrame(index=pheno.index)

    for col in ["detailed_category", "primary disease or tissue"]:
        if col in pheno.columns:
            result["cohort"] = pheno[col].map(
                lambda x: TCGA_DISEASE_TO_COHORT.get(x, x)
                if isinstance(x, str) else x
            )
            break

    for col in ["_gender", "gender", "Gender"]:
        if col in pheno.columns:
            result["gender"] = pheno[col].str.lower()
            break

    for col in ["_sample_type", "sample_type"]:
        if col in pheno.columns:
            result["sample_type"] = pheno[col]
            break

    return result


def extract_tss(barcode: str) -> str:
    """Extract TSS code from a TCGA barcode (e.g. TCGA-A7-A0CG-01 -> A7)."""
    parts = barcode.split("-")
    if len(parts) >= 2 and parts[0] == "TCGA":
        return parts[1]
    return ""
