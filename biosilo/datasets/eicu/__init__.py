"""eICU outcome prediction with one client per hospital.

Tasks cover mortality, length of stay, vasopressor-infusion onset, and a
respiratory-support onset proxy.
Each sample contains fixed-length hourly ``ts`` data and admission-level
``static`` features. Short stays are excluded, and train/test splits group ICU
stays by ``uniquepid``. Raw-table preprocessing is cached under
``cache_dir`` and its provenance is recorded in the partition metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ...core.contract import ClientData, unhashed
from . import _cohort, _events, _processed, _stage1
from ._processed import EVENT_TASKS, TASKS
from ._stage1 import Stage1Params

NAME = "eICU"
STORAGE = "npz"
SCHEMA_VERSION = 2

#: Multi-input fields in storage order.
Inputs = __import__("collections").namedtuple("eICUInputs", ("ts", "static"))


@dataclass(frozen=True)
class Params:
    task: str = "mortality_24h"
    num_clients: int = 20          # 0 = every qualifying hospital
    sort_mode: str = "size"        # size | positives | prevalence
    min_size: int = 10
    min_prev: float = 0.0
    min_minority: int = 0
    train_ratio: float = 0.75
    seed: int = 1
    include_diagnoses: bool = True
    drop_hospital_vars: bool = True
    diag_window: str = "5h"
    paradigm: str = "single_horizon"

    # Stage-one parameters affect the feature store and partition identity.
    min_dx_prevalence: float = 0.01
    within_prev: float = 0.25
    cross_prev: float = 0.70
    mask_mode: str = "exponential_decay"
    decay_rate: float = 4.0 / 3.0

    # Local storage locations do not affect partition identity.
    source_dir: str = unhashed("")
    cache_dir: str = unhashed("")

    def stage1(self) -> Stage1Params:
        return Stage1Params(
            observation_hours=TASKS[self.task],
            diagnosis_hours=_diagnosis_hours(self),
            train_ratio=self.train_ratio,
            split_seed=self.seed,
            min_dx_prevalence=self.min_dx_prevalence,
            within_prev=self.within_prev,
            cross_prev=self.cross_prev,
            mask_mode=self.mask_mode,
            decay_rate=self.decay_rate,
        )


def label(p: Params) -> str:
    return f"{p.task}_n{p.num_clients}_s{p.seed}"


def feature_groups(partition_manifest: dict) -> dict:
    """Named slices within the eICU model inputs."""
    widths = {
        (
            client["metadata"]["n_flat_features"],
            client["metadata"]["n_diag_features"],
        )
        for client in partition_manifest["clients"]
    }
    if len(widths) != 1:
        raise ValueError("Client eICU feature widths are inconsistent.")
    n_flat, n_diag = widths.pop()
    if not n_diag:
        return {}
    return {
        "diagnoses": {
            "input": "static",
            "start": n_flat,
            "stop": n_flat + n_diag,
        }
    }


def build(p: Params) -> Iterator[ClientData]:
    _check(p)
    max_seq_len = TASKS[p.task]

    # Reuse a feature store with matching preprocessing parameters when available.
    processed, provenance = _stage1.ensure(
        Path(p.cache_dir), Path(p.source_dir), p.stage1())

    pids, ts, static, labels_df, n_flat, n_diag, stage_assignment = _load_all(
        p, max_seq_len, processed)
    if p.task.startswith("mortality"):
        available = _mortality_available_indices(labels_df, pids)
        pids, ts, static = _processed.take(available, pids, ts, static)
    summary, entries = _processed.make_labels(labels_df, pids, ts, p.task, p.paradigm)

    keep = _processed.observation_window_filter(max_seq_len, ts)
    pids, ts, static, summary, entries = _processed.take(
        keep, pids, ts, static, summary, entries)

    hospital_of, person_of = _identity_maps(Path(p.source_dir))
    hospitals = np.array([hospital_of.get(pid, -1) for pid in pids])

    mapped = np.flatnonzero(hospitals >= 0)
    if len(mapped) < len(pids):
        pids, ts, static, summary, entries = _processed.take(
            mapped, pids, ts, static, summary, entries)
        hospitals = hospitals[mapped]

    if p.task in EVENT_TASKS:
        pids, ts, static, summary, entries, hospitals = _apply_event_labels(
            p, max_seq_len, pids, ts, static, hospitals)

    persons = np.array([person_of.get(pid, pid) for pid in pids])
    inductive_store = _stage1.uses_inductive_split(processed)
    if inductive_store:
        global_train = np.array([stage_assignment[pid] for pid in pids], dtype=bool)
        global_test = ~global_train
    else:
        global_train, global_test = _cohort.global_person_split(
            summary, persons, p.train_ratio, p.seed)

    candidates = _cohort.qualifying(
        hospitals, summary, p.min_size, p.min_minority, p.min_prev)
    if not candidates:
        raise RuntimeError(
            f"No hospital passed the filters for task={p.task!r} "
            f"(min_size={p.min_size}, min_minority={p.min_minority}, "
            f"min_prev={p.min_prev})."
        )
    if p.num_clients > len(candidates):
        _report_client_shortfall(len(candidates), p.num_clients)
    selected = _cohort.rank(candidates, summary, p.sort_mode, p.num_clients)

    for hid, idx in selected:
        y = np.asarray(entries)[idx].astype(np.int64)
        groups = persons[idx]
        train_i = np.flatnonzero(global_train[idx])
        test_i = np.flatnonzero(global_test[idx])

        x_ts = np.stack([ts[i] for i in idx]).astype(np.float32)
        x_static = np.stack([static[i] for i in idx]).astype(np.float32)

        yield ClientData(
            client_id=str(hid),
            train=(Inputs(x_ts[train_i], x_static[train_i]), y[train_i], groups[train_i]),
            test=(Inputs(x_ts[test_i], x_static[test_i]), y[test_i], groups[test_i]),
            meta={
                "group_unit": "person (uniquepid)",
                "hospital_id": int(hid),
                "n_stays": int(len(idx)),
                "n_persons": int(len(np.unique(groups))),
                "label_counts": _cohort.label_counts(summary[idx]),
                "max_seq_len": max_seq_len,
                "n_flat_features": n_flat,
                "n_diag_features": n_diag,
                "diagnosis_hours": (
                    _diagnosis_hours(p) if p.include_diagnoses else 0
                ),
                # External stores do not expose verifiable preprocessing parameters.
                "stage1_provenance": provenance,
                "preprocessing_protocol": (
                    "inductive" if inductive_store else "external-unverified"
                ),
                **({"event_definition": _events.definition(
                    "arf" if p.task.startswith("arf") else "shock")}
                   if p.task in EVENT_TASKS else {}),
            },
        )


# ── internals ─────────────────────────────────────────────────────────────────

def _check(p: Params) -> None:
    if p.task not in TASKS:
        raise ValueError(f"Unknown task {p.task!r}. Choose from {sorted(TASKS)}.")
    if p.paradigm != "single_horizon":
        raise NotImplementedError(
            "Only paradigm='single_horizon' is implemented. A rolling paradigm "
            "requires an (N, T) label contract and mandatory person grouping."
        )
    if not p.cache_dir:
        raise ValueError(
            "cache_dir is required: it is where the feature store lives and is "
            "built on first use if absent.")
    if not p.source_dir:
        raise ValueError(
            "source_dir is required: hospital and person identity come from the "
            "raw patient table, and stage 1 reads the raw tables when a store "
            "must be built."
        )
    _diagnosis_hours(p)


def _diagnosis_hours(p: Params) -> int:
    """Validated diagnosis window, capped at the observable task horizon."""
    text = str(p.diag_window).strip().lower()
    if not text.endswith("h") or not text[:-1].isdigit():
        raise ValueError("diag_window must be an integer number of hours, such as '5h'")
    requested = int(text[:-1])
    if not 1 <= requested <= 5:
        raise ValueError("diag_window must be between '1h' and '5h'")
    return min(requested, TASKS[p.task])


def _report_client_shortfall(survived: int, requested: int) -> None:
    noun = "client" if survived == 1 else "clients"
    print(
        f"[BioSilo] Only {survived} {noun} survived the filters; "
        f"requested {requested}."
    )


def _mortality_available_indices(labels_df, pids) -> np.ndarray:
    """Indices whose in-hospital mortality outcome is actually known."""
    import pandas as pd

    values = labels_df["actualhospitalmortality"]
    numeric = pd.to_numeric(values, errors="coerce")
    textual = values.astype("string").str.upper()
    known = numeric.isin({0, 1}) | textual.isin({"ALIVE", "EXPIRED"})
    known_pids = set(int(pid) for pid in values.index[known])
    return np.array(
        [index for index, pid in enumerate(pids) if pid in known_pids],
        dtype=np.int64,
    )


def _load_all(p: Params, max_seq_len: int, processed: Path):
    """Load all stage shards while retaining their person assignment."""
    import pandas as pd

    pids, ts, static, frames = [], [], [], []
    stage_assignment = {}
    n_flat = n_diag = 0

    for split in ("train", "val", "test"):
        split_dir = processed / split
        if not split_dir.is_dir():
            continue
        s_pids, s_ts, s_static, s_labels, n_flat, n_diag = _processed.load_split(
            split_dir, max_seq_len, p.include_diagnoses,
            p.drop_hospital_vars)
        pids.extend(s_pids)
        ts.extend(s_ts)
        static.extend(s_static)
        frames.append(s_labels)
        for pid in s_pids:
            assignment = split != 'test'
            if pid in stage_assignment and stage_assignment[pid] != assignment:
                raise ValueError(f'eICU patient {pid} appears across stage-one splits')
            stage_assignment[pid] = assignment

    if not pids:
        raise FileNotFoundError(
            f"No preprocessed splits under {processed} "
            "(expected train/, val/ or test/ subdirectories)."
        )

    labels_df = pd.concat(frames)
    labels_df = labels_df[~labels_df.index.duplicated(keep="first")]
    return pids, ts, static, labels_df, n_flat, n_diag, stage_assignment


def _identity_maps(eicu_dir: Path):
    """``patientunitstayid`` → hospital, and → person.

    Person identity is what makes readmissions visible.
    """
    import pandas as pd

    path = _events.find_table(eicu_dir, "patient")
    if path is None:
        raise FileNotFoundError(f"patient.csv not found under {eicu_dir}")

    df = pd.read_csv(path, usecols=["patientunitstayid", "hospitalid", "uniquepid"])
    hospital = df.set_index("patientunitstayid")["hospitalid"].to_dict()
    person = df.set_index("patientunitstayid")["uniquepid"].to_dict()
    return hospital, person


def _apply_event_labels(p, max_seq_len, pids, ts, static, hospitals):
    """Label events, excluding stays whose onset is inside the input window."""
    outcome = "arf" if p.task.startswith("arf") else "shock"
    onsets = _events.onset_times(Path(p.source_dir), outcome)
    horizon_minutes = max_seq_len * 60.0

    keep, labels = [], []
    for i, pid in enumerate(pids):
        onset = onsets.get(pid)
        if onset is not None and onset <= horizon_minutes:
            continue                       # already happened; nothing to predict
        keep.append(i)
        labels.append(1 if onset is not None else 0)

    keep = np.array(keep, dtype=np.int64)
    pids, ts, static = _processed.take(keep, pids, ts, static)
    summary = np.array(labels, dtype=np.int64)
    return pids, ts, static, summary, summary, hospitals[keep]
