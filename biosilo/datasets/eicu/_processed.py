"""Read preprocessed eICU features and derive task labels.

Expects the layout stage 1 writes:

    <processed>/{train,val,test}/
        stays.txt          patient ids, one per line
        labels.csv         indexed by patient
        flat.csv           indexed by patient
        diagnoses.csv      indexed by patient (or diagnoses_<window>.csv)
        timeseries.parquet indexed by patient (falls back to .csv)

BioSilo-generated inductive stores preserve their stage-one person assignment
when forming each hospital client. External stores without verifiable protocol
metadata are pooled and split globally by person at generation time.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

#: Observation window per task, in hours. LOS task names describe the outcome
#: threshold, not the input duration: both use the first 24 hours.
TASKS = {
    "mortality_24h": 24,
    "mortality_48h": 48,
    "los_3day": 24,
    "los_7day": 24,
    "shock_4h": 4,
    "shock_12h": 12,
    "arf_4h": 4,
    "arf_12h": 12,
}

#: Tasks labelled from raw-table event onset times.
EVENT_TASKS = {"shock_4h", "shock_12h", "arf_4h", "arf_12h"}

#: Hospital-level flat columns dropped by default.
HOSPITAL_LEVEL_PREFIXES = (
    "teachingstatus", "numbedscategory_", "region_", "physicianspeciality_",
)


def load_split(
    split_dir: Path,
    max_seq_len: int,
    include_diagnoses: bool,
    drop_hospital_vars: bool,
):
    """Load one preprocessed split into per-patient arrays."""
    patient_ids = [int(x) for x in (split_dir / "stays.txt").read_text().strip().split("\n")]
    patient_set = set(patient_ids)

    labels_df = _read_patient_indexed_csv(split_dir / "labels.csv")
    labels_df = labels_df.loc[labels_df.index.isin(patient_set)]

    flat_df = _read_patient_indexed_csv(split_dir / "flat.csv")
    if drop_hospital_vars:
        drop = [c for c in flat_df.columns
                if any(c.startswith(p) for p in HOSPITAL_LEVEL_PREFIXES)]
        if drop:
            flat_df = flat_df.drop(columns=drop)

    diag_df = None
    if include_diagnoses:
        path = split_dir / "diagnoses.csv"
        if path.exists():
            diag_df = _read_patient_indexed_csv(path)

    patient_data = _read_timeseries(split_dir, patient_set)

    n_flat = len(flat_df.columns)
    n_diag = len(diag_df.columns) if diag_df is not None else 0

    ts_features, static_features, valid_pids = [], [], []
    for pid in patient_ids:
        if pid not in patient_data or pid not in labels_df.index:
            continue

        # Select explicit modeled hours, never merely the first N source rows.
        hours, values = patient_data[pid]
        visible = (hours >= 1) & (hours <= max_seq_len)
        ts = values[visible]

        flat_vec = (flat_df.loc[pid].to_numpy(dtype=np.float32)
                    if pid in flat_df.index else np.zeros(n_flat, dtype=np.float32))
        np.nan_to_num(flat_vec, copy=False)
        parts = [flat_vec]

        if diag_df is not None:
            diag_vec = (diag_df.loc[pid].to_numpy(dtype=np.float32)
                        if pid in diag_df.index else np.zeros(n_diag, dtype=np.float32))
            np.nan_to_num(diag_vec, copy=False)
            parts.append(diag_vec)

        ts_features.append(ts)
        static_features.append(np.concatenate(parts).astype(np.float32))
        valid_pids.append(pid)

    return valid_pids, ts_features, static_features, labels_df, n_flat, n_diag


def _read_patient_indexed_csv(path: Path) -> pd.DataFrame:
    """Read a split CSV whose patient identifier may be an unnamed index.

    Depending on how the table was written, the serialized index can appear as
    ``Unnamed: 0``. Accept it alongside the explicit patient column names.
    """
    frame = pd.read_csv(path)
    for column in ("patient", "patientunitstayid", "Unnamed: 0"):
        if column in frame.columns:
            frame = frame.set_index(column)
            frame.index.name = "patient"
            return frame
    raise ValueError(
        f"{path} has no patient identifier column; expected 'patient', "
        "'patientunitstayid', or a serialized unnamed index"
    )


def _read_timeseries(split_dir: Path, patient_set: set) -> dict:
    """Validated per-patient ``(hours, values)`` pairs."""
    parquet = split_dir / "timeseries.parquet"
    csv = split_dir / "timeseries.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
        if "patient" in df.columns:
            df = df.set_index("patient")
        out = {}
        for pid, group in df.groupby(level=0, sort=False):
            pid = int(pid)
            if pid in patient_set:
                out[pid] = _validated_patient_timeseries(group, pid)
        return out

    pieces = {}
    for chunk in pd.read_csv(csv, chunksize=500_000):
        for pid, group in chunk.groupby("patient"):
            pid = int(pid)
            if pid not in patient_set:
                continue
            pieces.setdefault(pid, []).append(group.drop(columns=['patient']))
    return {
        pid: _validated_patient_timeseries(pd.concat(groups, ignore_index=True), pid)
        for pid, groups in pieces.items()
    }


def _validated_patient_timeseries(group: pd.DataFrame, patient: int):
    """Sort and validate one patient's explicit hourly key."""
    if 'time' not in group.columns:
        raise ValueError(
            f'eICU timeseries for patient {patient} has no time column; '
            'the store does not satisfy the patient-hour contract')

    group = group.copy()
    group['time'] = pd.to_numeric(group['time'], errors='raise').astype(np.int64)
    group.sort_values('time', kind='stable', inplace=True)
    hours = group['time'].to_numpy()
    if len(np.unique(hours)) != len(hours):
        raise ValueError(f'eICU patient {patient} has duplicate hourly rows')
    if len(hours) and hours[0] != 1:
        raise ValueError(f'eICU patient {patient} starts at hour {hours[0]}, expected 1')
    if len(hours) > 1 and not np.all(np.diff(hours) == 1):
        raise ValueError(f'eICU patient {patient} has missing or unsorted hours')

    cols = _feature_columns(group.columns, meta={'time', 'hour'})
    return hours, group[cols].to_numpy(dtype=np.float32)


def _feature_columns(columns, meta: set) -> List[str]:
    values = [c for c in columns if c not in meta and not c.endswith("_mask")]
    masks = [c for c in columns if c.endswith("_mask")]
    return values + masks


def make_labels(labels_df, patient_ids, ts_features, task: str, paradigm: str):
    """Per-patient labels.

    Returns ``(summary, entries)``. ``summary`` is always a scalar per patient,
    and is what filtering and stratification use regardless of paradigm.
    ``entries`` is the label actually stored: the same scalars under
    ``single_horizon``, or one array per patient under ``rolling``.
    """
    # pandas 3 uses a dedicated string dtype, so test numeric capability directly.
    base = "mortality" if task.startswith("mortality") else task
    mort_binary = None
    if base == "mortality":
        mort_col = labels_df["actualhospitalmortality"]
        numeric = pd.to_numeric(mort_col, errors="coerce")
        textual = mort_col.astype("string").str.upper().map(
            {"ALIVE": 0, "EXPIRED": 1})
        mort_binary = numeric.where(numeric.notna(), textual)
        mort_binary = mort_binary.where(mort_binary.isin({0, 1}))
        if mort_binary.reindex(patient_ids).isna().any():
            raise ValueError("mortality task contains stays with unknown outcomes")

    n = len(patient_ids)
    summary = np.zeros(n, dtype=np.int64)
    rolling: List[np.ndarray] = [] if paradigm == "rolling" else None

    for i, (pid, ts) in enumerate(zip(patient_ids, ts_features)):
        offset_min = float(labels_df.loc[pid, "unitdischargeoffset"])
        T_i = int(ts.shape[0])

        if base == "mortality":
            y = int(mort_binary.loc[pid])
            summary[i] = y
            if paradigm == "rolling":
                rolling.append(np.full(T_i, y, dtype=np.int64))

        elif base in ("los_3day", "los_7day"):
            threshold = 3 if base == "los_3day" else 7
            if paradigm == "single_horizon":
                summary[i] = 1 if offset_min / 1440.0 > threshold else 0
            else:
                hours = np.arange(1, T_i + 1, dtype=np.float64)
                remaining = (offset_min - hours * 60.0) / 1440.0
                arr = (remaining > threshold).astype(np.int64)
                rolling.append(arr)
                summary[i] = int(arr.max()) if T_i > 0 else 0

        elif task in EVENT_TASKS:
            summary[i] = 0      # replaced once onset times are known

        else:
            raise ValueError(f"Unknown task: {task}")

    return summary, (summary if paradigm == "single_horizon" else rolling)


def observation_window_filter(max_seq_len: int, ts_features) -> np.ndarray:
    """Return patients with a complete fixed-length observation window."""
    return np.array(
        [i for i, ts in enumerate(ts_features) if ts.shape[0] >= max_seq_len],
        dtype=np.int64,
    )


def take(indices, *sequences):
    """Subset several parallel sequences by index."""
    out = []
    for seq in sequences:
        if isinstance(seq, np.ndarray):
            out.append(seq[indices])
        else:
            out.append([seq[i] for i in indices])
    return out
