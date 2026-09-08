"""Optional stage-one integration checks against real eICU inputs.

The test rebuilds features from the demo tables and checks the patient-hour
and person-split contracts. ``EICU_CACHE_DIR`` is used only
to exercise external-store reuse.

Needs ``EICU_DEMO_DIR`` and ``EICU_CACHE_DIR``; skips without them. Takes
about a minute, since it runs the whole ETL.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def _env_dir(name):
    """The directory ``name`` points at, or None if the variable is unset.

    ``Path("")`` is ``Path(".")``, which always exists, so an unset variable
    would otherwise pass the skip guard and run the case against the repo root.
    """
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


DEMO = _env_dir("EICU_DEMO_DIR")
ORACLE = _env_dir("EICU_CACHE_DIR")

#: Decay rate used to create the reference store. It differs from 4/3 in the
#: fifth decimal and must match for exact mask comparison.
ORACLE_DECAY_RATE = 1.3333

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
def rebuilt_store_satisfies_the_patient_hour_and_split_contract():
    import pandas as pd

    from biosilo.datasets.eicu import _stage1

    params = _stage1.Stage1Params(decay_rate=ORACLE_DECAY_RATE)

    with tempfile.TemporaryDirectory() as tmp:
        store, provenance = _stage1.ensure(Path(tmp), DEMO, params, force=True)
        assert provenance == "built", provenance

        people = {}
        for split in ("train", "test"):
            mine = store / split
            labels = pd.read_csv(mine / 'labels.csv')
            people[split] = set(labels['uniquepid'].astype(str))

            frame = pd.read_parquet(mine / "timeseries.parquet").reset_index()
            assert not frame.duplicated(['patient', 'time']).any(), split
            frame = frame.sort_values(['patient', 'time'])
            assert frame.groupby('patient')['time'].first().eq(1).all(), split
            assert frame.groupby('patient')['time'].diff().dropna().eq(1).all(), split

        assert people['train'].isdisjoint(people['test'])


@case
def stage1_params_change_the_store_identity():
    """Preprocessing parameters produce distinct store identities."""
    from biosilo.datasets.eicu import _stage1

    a = _stage1.Stage1Params()
    b = _stage1.Stage1Params(decay_rate=ORACLE_DECAY_RATE)
    assert a.identity() != b.identity(), (a.identity(), b.identity())

    root = Path("/tmp")
    assert _stage1.store_path(root, a) != _stage1.store_path(root, b)


@case
def an_existing_store_is_reused_not_rebuilt():
    """An external feature store should be reused."""
    from biosilo.datasets.eicu import _stage1

    store, provenance = _stage1.ensure(ORACLE, DEMO)
    assert provenance == "external", provenance
    assert store == ORACLE, store


@case
def a_partial_store_is_not_mistaken_for_a_finished_one():
    from biosilo.datasets.eicu import _stage1

    with tempfile.TemporaryDirectory() as tmp:
        params = _stage1.Stage1Params()
        half = _stage1.store_path(Path(tmp), params)
        (half / "train").mkdir(parents=True)
        # looks plausible, but no completion marker
        assert not _stage1.is_complete(half)


CASES = [
    rebuilt_store_satisfies_the_patient_hour_and_split_contract,
    stage1_params_change_the_store_identity,
    an_existing_store_is_reused_not_rebuilt,
    a_partial_store_is_not_mistaken_for_a_finished_one,
]


def main() -> int:
    if DEMO is None or ORACLE is None or not DEMO.is_dir() or not ORACLE.is_dir():
        print("  skipped: set EICU_DEMO_DIR and EICU_CACHE_DIR")
        return 0

    for fn in CASES:
        fn()

    for name in PASSED:
        print(f"  ok    {name}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
