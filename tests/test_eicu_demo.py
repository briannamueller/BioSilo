"""Optional integration checks against the eICU demo tables.

The ODbL demo contains 2,520 stays from 186 hospitals and covers source column
names, filenames, and clinical distributions.

Point ``EICU_DEMO_DIR`` at the extracted demo. The suite skips without it, so it
still runs on a machine that has no eICU data at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosilo.datasets.eicu import _events
from biosilo.datasets.eicu import _identity_maps

def _env_dir(name):
    """The directory ``name`` points at, or None if the variable is unset.

    ``Path("")`` is ``Path(".")``, which always exists, so an unset variable
    would otherwise pass the skip guard and run the case against the repo root.
    """
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


DEMO = _env_dir("EICU_DEMO_DIR")

#: Optional: a processed store built by Fed-eICU's stage 1 over the demo tables.
#: Only the end-to-end case needs it; the rest read the raw tables directly.
PROCESSED = _env_dir("EICU_CACHE_DIR")

#: Tables the port reads by name. eICU's own distributions disagree on case,
#: the demo ships `infusiondrug.csv.gz`, the docs say `infusionDrug`.
TABLES = ("patient", "infusionDrug", "medication",
          "respiratoryCare", "respiratoryCharting")

PASSED, FAILED = [], []


def case(fn):
    def run():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            FAILED.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            PASSED.append(fn.__name__)
    return run


@case
def every_table_resolves_regardless_of_case():
    for table in TABLES:
        assert _events.find_table(DEMO, table) is not None, f"{table} not found"


@case
def table_lookup_is_case_insensitive():
    """Check lookup against the demo's lowercase infusion filename."""
    found = _events.find_table(DEMO, "infusionDrug")
    # The demo's actual file is lowercase; an exact-match lookup would only
    # "work" on a case-insensitive filesystem.
    assert found.name == "infusiondrug.csv.gz", found.name


@case
def identity_maps_read_real_columns():
    hospital_of, person_of = _identity_maps(DEMO)
    assert len(hospital_of) == 2520, len(hospital_of)
    assert len(set(hospital_of.values())) == 186, len(set(hospital_of.values()))
    assert len(set(person_of.values())) == 1841, len(set(person_of.values()))


@case
def readmissions_are_common_enough_to_matter():
    """Confirm repeat-stay frequency in the demo cohort."""
    _, person_of = _identity_maps(DEMO)
    counts = {}
    for person in person_of.values():
        counts[person] = counts.get(person, 0) + 1

    repeated = sum(1 for n in counts.values() if n > 1)
    stays_involved = sum(n for n in counts.values() if n > 1)

    assert repeated == 416, repeated
    assert stays_involved / len(person_of) > 0.20, stays_involved / len(person_of)


@case
def shock_onsets_are_plausible():
    onsets = _events.onset_times(DEMO, "shock")
    assert len(onsets) == 341, len(onsets)
    # Offsets are minutes from unit admission and may be negative: a patient
    # transferred in already on vasopressors started before this unit stay.
    assert min(onsets.values()) < 0
    positives_4h = sum(1 for t in onsets.values() if t >= 4 * 60)
    assert positives_4h == 123, positives_4h


@case
def arf_onsets_are_plausible():
    onsets = _events.onset_times(DEMO, "arf")
    assert len(onsets) == 469, len(onsets)
    positives_12h = sum(1 for t in onsets.values() if t >= 12 * 60)
    assert positives_12h == 87, positives_12h


@case
def end_to_end_on_real_preprocessed_data():
    """Run ``build()`` against the optional processed demo store."""
    import tempfile

    import biosilo

    with tempfile.TemporaryDirectory() as tmp:
        out = biosilo.generate(
            "eICU", root=tmp,
            task="mortality_24h", num_clients=0, min_size=4, min_minority=1,
            train_ratio=0.75, seed=1,
            cache_dir=str(PROCESSED), source_dir=str(DEMO),
        )
        p = biosilo.load("eICU", root=tmp, partition=out.name)

        assert p.num_clients > 50, p.num_clients
        assert p.num_classes == 2, p.num_classes
        assert p.has_groups and "person" in p.group_unit

        ts_spec, static_spec = p.inputs
        assert ts_spec["name"] == "ts" and ts_spec["shape"][0] == 24, ts_spec
        assert static_spec["name"] == "static", static_spec

        holds_a_readmission = False
        for cid in range(p.num_clients):
            X, y, gtr = p.client(cid, "train")
            _, yte, gte = p.client(cid, "test")

            assert X.ts.shape[1:] == tuple(ts_spec["shape"]), X.ts.shape
            assert len(X.ts) == len(X.static) == len(y) == len(gtr)
            assert np.intersect1d(gtr, gte).size == 0, f"client {cid} leaks a person"

            if len(np.unique(np.concatenate([gtr, gte]))) < len(gtr) + len(gte):
                holds_a_readmission = True

        assert holds_a_readmission, "no client had a repeated person, so grouping is untested"


CASES = [
    every_table_resolves_regardless_of_case,
    table_lookup_is_case_insensitive,
    identity_maps_read_real_columns,
    readmissions_are_common_enough_to_matter,
    shock_onsets_are_plausible,
    arf_onsets_are_plausible,
    end_to_end_on_real_preprocessed_data,
]


def main() -> int:
    if DEMO is None or not DEMO.is_dir():
        print("  skipped: set EICU_DEMO_DIR to the extracted eICU demo")
        return 0

    for fn in CASES:
        if fn is end_to_end_on_real_preprocessed_data and (PROCESSED is None or not PROCESSED.is_dir()):
            print("  skipped end_to_end: set EICU_CACHE_DIR")
            continue
        fn()

    for name in PASSED:
        print(f"  ok    {name}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
