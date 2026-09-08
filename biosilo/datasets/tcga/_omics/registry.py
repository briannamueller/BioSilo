"""TCGA cohort names and source-specific identifiers."""
from __future__ import annotations

# Every TCGA cohort BioSilo can build.
TCGA_COHORTS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]

# Mapping: TCGA abbreviation -> Xena phenotype disease name
TCGA_COHORT_TO_DISEASE = {
    "ACC": "Adrenocortical Cancer",
    "BLCA": "Bladder Urothelial Carcinoma",
    "BRCA": "Breast Invasive Carcinoma",
    "CESC": "Cervical & Endocervical Cancer",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon Adenocarcinoma",
    "DLBC": "Diffuse Large B-Cell Lymphoma",
    "ESCA": "Esophageal Carcinoma",
    "GBM": "Glioblastoma Multiforme",
    "HNSC": "Head & Neck Squamous Cell Carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney Clear Cell Carcinoma",
    "KIRP": "Kidney Papillary Cell Carcinoma",
    "LAML": "Acute Myeloid Leukemia",
    "LGG": "Brain Lower Grade Glioma",
    "LIHC": "Liver Hepatocellular Carcinoma",
    "LUAD": "Lung Adenocarcinoma",
    "LUSC": "Lung Squamous Cell Carcinoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian Serous Cystadenocarcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
    "PCPG": "Pheochromocytoma & Paraganglioma",
    "PRAD": "Prostate Adenocarcinoma",
    "READ": "Rectum Adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach Adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumor",
    "THCA": "Thyroid Carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine Corpus Endometrioid Carcinoma",
    "UCS": "Uterine Carcinosarcoma",
    "UVM": "Uveal Melanoma",
}

# Reverse mapping
TCGA_DISEASE_TO_COHORT = {v: k for k, v in TCGA_COHORT_TO_DISEASE.items()}

# GDC project ID format
def gdc_project_id(cohort: str) -> str:
    """Convert short cohort name to GDC project ID."""
    return f"TCGA-{cohort.upper()}"


def validate_cohorts(cohorts: list[str]) -> list[str]:
    """Validate and normalize cohort names. Returns uppercase list."""
    normalized = []
    for c in cohorts:
        c_upper = c.upper().replace("TCGA-", "")
        if c_upper not in TCGA_COHORTS:
            raise ValueError(
                f"Unknown TCGA cohort: '{c}'. "
                f"Available: {', '.join(TCGA_COHORTS)}"
            )
        normalized.append(c_upper)
    return sorted(set(normalized))
