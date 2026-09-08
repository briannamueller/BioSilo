"""eICU downstream cohort, shape, grouping, and split tests.

Stage-one preprocessing is covered separately by optional demo/oracle tests.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import biosilo
import fixtures_eicu
from biosilo.datasets import eicu

PASSED, FAILED = [], []
TASK = "mortality_24h"
WINDOW = 24


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
    processed, raw = fixtures_eicu.build(root)
    kwargs = dict(
        task=TASK, num_clients=0, min_size=10, train_ratio=0.75, seed=1,
        cache_dir=str(processed), source_dir=str(raw),
    )
    kwargs.update(overrides)
    out = biosilo.generate("eICU", root=root / "data", **kwargs)
    return biosilo.load("eICU", root=root / "data", partition=out.name)


# ── cohort selection ──────────────────────────────────────────────────────────

@case
def only_qualifying_hospitals_become_clients(root):
    p = _generate(root)
    # hospital 3 is below min_size; hospital 4 has one class only
    assert sorted(p.client_ids) == ["1", "2"], p.client_ids


@case
def num_clients_caps_by_rank(root):
    p = _generate(root, num_clients=1)
    # hospital 1 has more stays than hospital 2, and sort_mode defaults to size
    assert p.client_ids == ["1"], p.client_ids


@case
def requested_client_shortfall_is_reported(root):
    output = StringIO()
    with redirect_stdout(output):
        p = _generate(root, num_clients=5)
    assert p.num_clients == 2, p.num_clients
    assert "Only 2 clients survived the filters; requested 5" in output.getvalue()


@case
def min_size_can_admit_the_small_hospital(root):
    p = _generate(root, min_size=2)
    assert "3" in p.client_ids, p.client_ids


# ── shapes ────────────────────────────────────────────────────────────────────

@case
def samples_are_two_inputs_of_fixed_length(root):
    p = _generate(root)
    assert p.is_multi_input
    names = [i["name"] for i in p.inputs]
    assert names == ["ts", "static"], names

    ts_spec, static_spec = p.inputs
    assert ts_spec["shape"][0] == WINDOW, ts_spec          # exactly the window
    assert ts_spec["shape"][1] == fixtures_eicu.N_TS_FEATURES * 2  # values + masks

    X, y, groups = p.client(0, "train")
    assert X.ts.shape[1:] == (WINDOW, fixtures_eicu.N_TS_FEATURES * 2), X.ts.shape
    assert len(X.ts) == len(X.static) == len(y) == len(groups)
    assert p.manifest["clients"][0]["metadata"]["diagnosis_hours"] == 5


@case
def hospital_level_flat_columns_are_dropped(root):
    with_vars = _generate(root, drop_hospital_vars=False)
    n_with = with_vars.inputs[1]["shape"][0]

    with tempfile.TemporaryDirectory() as other:
        without = _generate(Path(other), drop_hospital_vars=True)
        n_without = without.inputs[1]["shape"][0]

    assert n_without == n_with - 1, (n_without, n_with)


# ── the observation-window filter ─────────────────────────────────────────────

@case
def short_stays_are_dropped(root):
    p = _generate(root)
    # hospital 2 has 12 stays, two of which are only 10h long
    idx = p.client_ids.index("2")
    client = p.manifest["clients"][idx]
    n = sum(client[split] for split in ("train", "test"))
    assert n == 10, f"expected 10 stays after the window filter, got {n}"


# ── grouping ──────────────────────────────────────────────────────────────────

@case
def readmitted_person_does_not_straddle_the_split(root):
    p = _generate(root)
    assert p.has_groups
    assert "person" in (p.group_unit or ""), p.group_unit

    all_train, all_test = [], []
    for cid in range(p.num_clients):
        _, _, gtr = p.client(cid, "train")
        _, _, gte = p.client(cid, "test")
        all_train.extend(gtr)
        all_test.extend(gte)
        overlap = np.intersect1d(gtr, gte)
        assert overlap.size == 0, f"client {cid}: person on both sides: {overlap}"

    overlap = np.intersect1d(all_train, all_test)
    assert overlap.size == 0, f"person crosses hospitals and partitions: {overlap}"


@case
def readmission_is_actually_present_in_the_fixture(root):
    """Confirm that the fixture exercises repeated-person grouping."""
    p = _generate(root)
    idx = p.client_ids.index("1")
    meta = p.manifest["clients"][idx]["metadata"]
    assert meta["n_persons"] < meta["n_stays"], meta


# ── refusals ──────────────────────────────────────────────────────────────────

@case
def rolling_paradigm_is_refused(root):
    try:
        _generate(root, paradigm="rolling")
    except NotImplementedError as exc:
        assert "single_horizon" in str(exc)
    else:
        raise AssertionError("rolling was accepted")


@case
def unknown_task_is_refused(root):
    try:
        _generate(root, task="not_a_task")
    except ValueError as exc:
        assert "Unknown task" in str(exc)
    else:
        raise AssertionError("unknown task accepted")


@case
def missing_default_source_dir_is_refused(root):
    processed, _ = fixtures_eicu.build(root)
    data_root = root / "data"
    try:
        biosilo.generate(
            "eICU", root=data_root, task=TASK, cache_dir=str(processed),
        )
    except FileNotFoundError as exc:
        assert str(data_root / "_raw" / "eICU") in str(exc)
    else:
        raise AssertionError("missing default source directory accepted")


# ── identity ──────────────────────────────────────────────────────────────────

@case
def data_paths_do_not_change_the_partition_id(root):
    """The same cohort built from a different location must keep its identity."""
    p1 = _generate(root)
    with tempfile.TemporaryDirectory() as other:
        p2 = _generate(Path(other))
    assert p1.partition_id == p2.partition_id, (p1.partition_id, p2.partition_id)


@case
def partition_label_uses_task_client_count_and_seed(root):
    params = eicu.Params(
        task="shock_4h", num_clients=7, sort_mode="prevalence", seed=9,
        include_diagnoses=False, drop_hospital_vars=False,
    )
    assert eicu.label(params) == "shock_4h_n7_s9"


@case
def los_threshold_is_not_used_as_the_observation_window(root):
    """LOS is predicted from day one, before either outcome threshold."""
    from biosilo.datasets.eicu._processed import TASKS

    assert TASKS["los_3day"] == 24
    assert TASKS["los_7day"] == 24


@case
def raw_tables_resolve_regardless_of_filename_case(root):
    """Raw table lookup must handle case differences between distributions."""
    from biosilo.datasets.eicu._stage1.extract_tables import load_csv

    raw = root / "raw"
    raw.mkdir()
    (raw / "apachepatientresult.csv").write_text("patientunitstayid\n1\n")

    df = load_csv(raw, "apachePatientResult.csv")
    assert list(df.columns) == ["patientunitstayid"], list(df.columns)

    try:
        load_csv(raw, "notATable.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing table should raise")


@case
def static_extraction_keeps_stays_without_apache_rows(root):
    """Missing APACHE tables may remove fields, never the patient-table row."""
    import pandas as pd

    from biosilo.datasets.eicu._stage1.extract_tables import create_flat_features

    patient = pd.DataFrame([
        {
            "patientunitstayid": pid, "hospitalid": 10,
            "unitadmittime24": "08:00:00", "gender": "Female", "age": "60",
            "ethnicity": "Caucasian", "admissionheight": 165.0,
            "admissionweight": 70.0, "apacheadmissiondx": "diagnosis",
            "unittype": "MICU", "unitadmitsource": "ER",
            "unitvisitnumber": 1, "unitstaytype": "admit",
        }
        for pid in (1, 2)
    ])
    labels = pd.DataFrame({"patientunitstayid": [1, 2]})
    aps = pd.DataFrame({
        "patientunitstayid": [1], "intubated": [0], "vent": [0],
        "dialysis": [0], "eyes": [4], "motor": [6], "verbal": [5],
        "meds": [0],
    })
    apr = pd.DataFrame({
        "patientunitstayid": [1], "physicianspeciality": ["critical care"]})
    apv = pd.DataFrame({"patientunitstayid": [1], "bedcount": [20]})
    hospital = pd.DataFrame({
        "hospitalid": [10], "numbedscategory": [">= 50"],
        "teachingstatus": ["t"], "region": ["Midwest"],
    })

    flat = create_flat_features(patient, aps, apr, apv, hospital, labels)
    assert flat["patientunitstayid"].tolist() == [1, 2]
    missing_apache = flat.set_index("patientunitstayid").loc[2]
    assert missing_apache["gender"] == "Female"
    assert missing_apache["admissionheight"] == 165.0
    assert pd.isna(missing_apache["physicianspeciality"])


@case
def complete_stage1_store_can_seed_another_horizon(root):
    """Reusable raw artifacts are copied without sharing mutable file state."""
    import json

    from biosilo.datasets.eicu import _stage1

    source = root / "stage1_source"
    target = root / "target"
    source.mkdir()
    target.mkdir()
    for name in _stage1.RAW_ARTIFACTS:
        (source / name).write_text(f"content for {name}\n")
    (source / _stage1.MARKER).write_text(json.dumps({
        "schema_version": 3, "preprocessing_protocol": "inductive",
    }))

    assert _stage1._find_raw_seed(root) == source
    _stage1._copy_raw_artifacts(source, target)
    for name in _stage1.RAW_ARTIFACTS:
        assert (target / name).read_bytes() == (source / name).read_bytes()

    (target / "stays.txt").write_text("changed\n")
    assert (source / "stays.txt").read_text().startswith("content for")


@case
def unnamed_serialized_patient_indexes_are_accepted(root):
    """Stage-one CSVs may serialize a blank patient-index header."""
    from biosilo.datasets.eicu import _processed

    split = root / "split"
    split.mkdir()
    (split / "stays.txt").write_text("7\n")

    labels = pd.DataFrame({
        "actualhospitalmortality": [0], "unitdischargeoffset": [1440],
    }, index=pd.Index([7]))
    flat = pd.DataFrame({"age": [60.0]}, index=pd.Index([7]))
    diagnoses = pd.DataFrame({"dx": [1.0]}, index=pd.Index([7]))
    labels.to_csv(split / "labels.csv")
    flat.to_csv(split / "flat.csv")
    diagnoses.to_csv(split / "diagnoses.csv")
    pd.DataFrame({
        "time": [1], "value": [2.0], "value_mask": [1.0],
    }, index=pd.Index([7])).to_parquet(split / "timeseries.parquet")

    pids, ts, static, loaded_labels, n_flat, n_diag = _processed.load_split(
        split, max_seq_len=1, include_diagnoses=True,
        drop_hospital_vars=False,
    )
    assert pids == [7]
    assert ts[0].shape == (1, 2)
    assert static[0].tolist() == [60.0, 1.0]
    assert loaded_labels.index.tolist() == [7]
    assert (n_flat, n_diag) == (1, 1)


@case
def shock_onset_requires_a_positive_documented_infusion(root):
    """A medication order is not evidence that a vasopressor was administered."""
    from biosilo.datasets.eicu import _events

    raw = root / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"patientunitstayid": 1, "infusionoffset": 300,
         "drugname": "Norepinephrine", "drugrate": 0},
        {"patientunitstayid": 2, "infusionoffset": 360,
         "drugname": "Levophed", "drugrate": 2.5},
        {"patientunitstayid": 3, "infusionoffset": 420,
         "drugname": "saline", "drugrate": 10},
    ]).to_csv(raw / "infusionDrug.csv", index=False)
    pd.DataFrame([
        {"patientunitstayid": 4, "drugstartoffset": 120,
         "drugname": "norepinephrine"},
    ]).to_csv(raw / "medication.csv", index=False)

    assert _events.onset_times(raw, "shock") == {2: 360}


@case
def arf_peep_limit_must_be_positive(root):
    """Zero/default respiratory-care values must not create an onset."""
    from biosilo.datasets.eicu import _events

    raw = root / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"patientunitstayid": 1, "respcarestatusoffset": 100,
         "ventstartoffset": 0, "peeplimit": 0},
        {"patientunitstayid": 2, "respcarestatusoffset": 200,
         "ventstartoffset": 0, "peeplimit": 5},
        {"patientunitstayid": 3, "respcarestatusoffset": 300,
         "ventstartoffset": 250, "peeplimit": 0},
        {"patientunitstayid": 4, "respcarestatusoffset": 50,
         "ventstartoffset": -60, "peeplimit": 0},
    ]).to_csv(raw / "respiratoryCare.csv", index=False)

    assert _events.onset_times(raw, "arf") == {2: 200, 3: 250, 4: -60}


@case
def event_at_horizon_boundary_is_not_a_prediction_target(root):
    """Hour 4 includes minute 240, so an onset at 240 is already visible."""
    from biosilo.datasets.eicu import _events

    original = _events.onset_times
    _events.onset_times = lambda *_args: {1: 240.0, 2: 241.0}
    try:
        pids, _, _, summary, entries, _ = eicu._apply_event_labels(
            eicu.Params(task="shock_4h"), 4,
            [1, 2, 3],
            [np.zeros((4, 2), dtype=np.float32) for _ in range(3)],
            [np.zeros(1, dtype=np.float32) for _ in range(3)],
            np.array([10, 10, 10]),
        )
    finally:
        _events.onset_times = original

    assert pids == [2, 3], pids
    assert summary.tolist() == [1, 0], summary
    assert entries.tolist() == [1, 0], entries


@case
def diagnosis_window_never_extends_past_the_task_input(root):
    """A 4-hour task must not consume diagnoses entered during hour 5."""
    from biosilo.datasets.eicu._stage1.diagnoses import within_window

    params = eicu.Params(task="arf_4h", diag_window="5h")
    assert params.stage1().diagnosis_hours == 4

    diagnoses = pd.DataFrame({
        "patientunitstayid": [1, 1, 2],
        "diagnosisstring": ["before", "boundary", "after"],
        "diagnosis_event_offset": [239, 240, 299],
    })
    visible = within_window(diagnoses, 240)
    assert visible["diagnosisstring"].tolist() == ["before"]


@case
def diagnosis_sources_preserve_their_event_offsets(root):
    from biosilo.datasets.eicu._stage1.extract_tables import create_diagnoses

    labels = pd.DataFrame({"patientunitstayid": [1]})
    diagnoses = create_diagnoses(
        pd.DataFrame(),
        pd.DataFrame({
            "patientunitstayid": [1], "diagnosisoffset": [239],
            "diagnosisstring": ["current|diagnosis"],
        }),
        pd.DataFrame({
            "patientunitstayid": [1], "pasthistoryoffset": [240],
            "pasthistorypath": ["notes/Progress Notes/Past History/Organ Systems/x"],
        }),
        pd.DataFrame({
            "patientunitstayid": [1], "admitdxenteredoffset": [299],
            "admitdxpath": ["admission diagnosis|x"],
        }),
        labels,
    )
    assert sorted(diagnoses["diagnosis_event_offset"].tolist()) == [239, 240, 299]


@case
def unsupported_diagnosis_window_is_refused(root):
    try:
        eicu._check(eicu.Params(
            diag_window="six hours", cache_dir="x", source_dir="y"))
    except ValueError as exc:
        assert "diag_window" in str(exc)
    else:
        raise AssertionError("an ambiguous diagnosis window was accepted")


@case
def unknown_mortality_does_not_break_nonmortality_tasks(root):
    from biosilo.datasets.eicu import _processed

    labels = pd.DataFrame({
        "actualhospitalmortality": [np.nan, 1.0],
        "unitdischargeoffset": [2 * 1440, 8 * 1440],
    }, index=[1, 2])
    ts = [np.zeros((24, 1)), np.zeros((24, 1))]
    summary, entries = _processed.make_labels(
        labels, [1, 2], ts, "los_3day", "single_horizon")
    assert summary.tolist() == [0, 1]
    assert entries.tolist() == [0, 1]


@case
def mortality_availability_is_applied_only_when_needed(root):
    from biosilo.datasets.eicu import _processed

    labels = pd.DataFrame({
        "actualhospitalmortality": [0, np.nan, "EXPIRED", "unknown"],
        "unitdischargeoffset": [2000, 2000, 2000, 2000],
    }, index=[10, 20, 30, 40])
    keep = eicu._mortality_available_indices(labels, [10, 20, 30, 40])
    assert keep.tolist() == [0, 2]
    summary, _ = _processed.make_labels(
        labels, [10, 30], [np.zeros((24, 1))] * 2,
        "mortality_24h", "single_horizon")
    assert summary.tolist() == [0, 1]


CASES = [
    only_qualifying_hospitals_become_clients,
    num_clients_caps_by_rank,
    requested_client_shortfall_is_reported,
    min_size_can_admit_the_small_hospital,
    samples_are_two_inputs_of_fixed_length,
    hospital_level_flat_columns_are_dropped,
    short_stays_are_dropped,
    readmitted_person_does_not_straddle_the_split,
    readmission_is_actually_present_in_the_fixture,
    rolling_paradigm_is_refused,
    unknown_task_is_refused,
    missing_default_source_dir_is_refused,
    data_paths_do_not_change_the_partition_id,
    partition_label_uses_task_client_count_and_seed,
    los_threshold_is_not_used_as_the_observation_window,
    raw_tables_resolve_regardless_of_filename_case,
    static_extraction_keeps_stays_without_apache_rows,
    complete_stage1_store_can_seed_another_horizon,
    unnamed_serialized_patient_indexes_are_accepted,
    shock_onset_requires_a_positive_documented_infusion,
    arf_peep_limit_must_be_positive,
    event_at_horizon_boundary_is_not_a_prediction_target,
    diagnosis_window_never_extends_past_the_task_input,
    diagnosis_sources_preserve_their_event_offsets,
    unsupported_diagnosis_window_is_refused,
    unknown_mortality_does_not_break_nonmortality_tasks,
    mortality_availability_is_applied_only_when_needed,
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
