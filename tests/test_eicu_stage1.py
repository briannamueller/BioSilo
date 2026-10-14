"""Optional stage-one integration checks against real eICU inputs.

The test rebuilds features from the demo tables and checks the patient-hour
and person-split contracts. ``EICU_CACHE_DIR`` is used only
to exercise external-store reuse.

The real-data checks need ``EICU_DEMO_DIR`` and ``EICU_CACHE_DIR`` and skip
without them. Rebuilding the demo takes about a minute.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def _env_dir(name):
    """Return the directory named by an environment variable, or None."""
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


DEMO = _env_dir("EICU_DEMO_DIR")
ORACLE = _env_dir("EICU_CACHE_DIR")

#: Decay rate used to create the reference store. It differs from 4/3 in the
#: fifth decimal and must match for exact mask comparison.
ORACLE_DECAY_RATE = 1.3333


@pytest.mark.skipif(
    DEMO is None or not DEMO.is_dir(),
    reason="set EICU_DEMO_DIR to the extracted eICU demo",
)
def test_rebuilt_store_satisfies_the_patient_hour_and_split_contract():
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


def test_stage1_params_change_the_store_identity():
    """Preprocessing parameters produce distinct store identities."""
    from biosilo.datasets.eicu import _stage1

    a = _stage1.Stage1Params()
    b = _stage1.Stage1Params(decay_rate=ORACLE_DECAY_RATE)
    assert a.identity() != b.identity(), (a.identity(), b.identity())

    root = Path("/tmp")
    assert _stage1.store_path(root, a) != _stage1.store_path(root, b)


@pytest.mark.skipif(
    DEMO is None or ORACLE is None or not DEMO.is_dir() or not ORACLE.is_dir(),
    reason="set EICU_DEMO_DIR and EICU_CACHE_DIR",
)
def test_an_existing_store_is_reused_not_rebuilt():
    """An external feature store should be reused."""
    from biosilo.datasets.eicu import _stage1

    store, provenance = _stage1.ensure(ORACLE, DEMO)
    assert provenance == "external", provenance
    assert store == ORACLE, store


def test_a_partial_store_is_not_mistaken_for_a_finished_one():
    from biosilo.datasets.eicu import _stage1

    with tempfile.TemporaryDirectory() as tmp:
        params = _stage1.Stage1Params()
        half = _stage1.store_path(Path(tmp), params)
        (half / "train").mkdir(parents=True)
        # looks plausible, but no completion marker
        assert not _stage1.is_complete(half)
