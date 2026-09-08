"""Label extraction and encoding from clinical/phenotype data."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Stage IV variants considered "advanced"
_ADVANCED_STAGES = {
    "Stage IV", "Stage IVA", "Stage IVB", "Stage IVC",
}


def encode_cancer_type(sample_ids, cohort_assignments, cohorts):
    """Integer-encode cancer type. Returns (labels, class_map)."""
    cohort_to_int = {c: i for i, c in enumerate(sorted(cohorts))}
    labels = np.array(
        [cohort_to_int[cohort_assignments[s]] for s in sample_ids],
        dtype=np.int64,
    )
    class_map = {i: c for c, i in cohort_to_int.items()}
    return labels, class_map


def encode_stage(stages):
    """Binary stage: advanced (Stage IV+) = 1, else = 0, NaN for missing."""
    def _encode(s):
        if not isinstance(s, str) or not s.strip():
            return np.nan
        s = s.strip()
        if s in _ADVANCED_STAGES:
            return 1.0
        if s.startswith("Stage"):
            return 0.0
        return np.nan

    labels = stages.apply(_encode).values
    class_map = {0: "Non-advanced", 1: "Advanced"}
    return labels, class_map


def encode_gender(genders):
    """Binary gender: male = 0, female = 1, NaN for missing."""
    mapping = {
        "male": 0.0, "female": 1.0,
        "MALE": 0.0, "FEMALE": 1.0,
        "m": 0.0, "f": 1.0,
        "M": 0.0, "F": 1.0,
    }
    labels = genders.map(mapping).values
    class_map = {0: "Male", 1: "Female"}
    return labels, class_map


def encode_survival(days_to_death, vital_status, threshold_days=365 * 3):
    """Binary survival: 1 if survived past threshold_days, 0 if not. NaN for censored."""
    labels = np.full(len(days_to_death), np.nan)

    for i, (days, status) in enumerate(zip(days_to_death, vital_status)):
        if isinstance(status, str) and status.lower() == "alive":
            # If alive and follow-up >= threshold, label as survived
            if isinstance(days, (int, float)) and days >= threshold_days:
                labels[i] = 1.0
            # If alive but short follow-up, censored -> NaN
        elif isinstance(status, str) and status.lower() == "dead":
            if isinstance(days, (int, float)):
                labels[i] = 0.0 if days < threshold_days else 1.0

    class_map = {0: "Deceased", 1: "Survived"}
    return labels, class_map
