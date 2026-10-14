"""Always-on regression tests for the eICU patient-hour contract."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biosilo.datasets.eicu import _processed
from biosilo.datasets.eicu import _stage1
from biosilo.datasets.eicu._stage1 import apply_filters, flat_and_labels, timeseries


def test_unit_conversion_precedes_hourly_aggregation():
    raw = pd.DataFrame({
        'patientunitstayid': [42, 42],
        'offset': [5, 50],
        'feature': ['FiO2', 'FiO2'],
        'value': [0.4, 40.0],
    })
    raw = timeseries.normalize_long_feature(raw, 'feature', 'value')
    hourly = timeseries.reconfigure_timeseries_fast(
        raw, offset_column='offset', feature_column='feature')
    assert np.isclose(hourly.loc[(42, 1), 'FiO2'], 40.0), hourly

    temperature = timeseries.normalize_temperature_celsius(
        pd.Series([37.0, 98.6, 67.8]))
    assert np.allclose(temperature.iloc[:2], [37.0, 37.0])
    assert np.isnan(temperature.iloc[2])
    periodic = timeseries.normalize_temperature_celsius(
        pd.Series([37.0, 98.6]), allow_fahrenheit=False)
    assert np.isclose(periodic.iloc[0], 37.0)
    assert np.isnan(periodic.iloc[1])


def test_sources_join_to_one_dense_row_per_patient_hour():
    lab = pd.DataFrame(
        {'FiO2': [40.0, 50.0], 'sodium': [140.0, 141.0]},
        index=pd.MultiIndex.from_tuples(
            [(42, 1), (42, 3)], names=['patient', 'time']),
    )
    resp = pd.DataFrame(
        {'FiO2': [40.0], 'PEEP': [5.0]},
        index=pd.MultiIndex.from_tuples(
            [(42, 1)], names=['patient', 'time']),
    )
    sources, collisions = timeseries.namespace_overlapping_features(
        {'lab': lab, 'resp': resp})
    merged = timeseries.merge_patient_hours(sources, {42: 3}, [42])

    assert collisions == {'FiO2': ['lab', 'resp']}, collisions
    assert list(merged.index) == [(42, 1), (42, 2), (42, 3)]
    assert merged.index.is_unique
    assert {'lab__FiO2', 'resp__FiO2'} <= set(merged.columns)
    assert merged.loc[(42, 2)].isna().all()
    assert np.isclose(merged.loc[(42, 1), 'lab__FiO2'], 40.0)
    assert np.isclose(merged.loc[(42, 1), 'resp__FiO2'], 40.0)


def test_cross_source_name_collisions_are_qualified():
    index = pd.MultiIndex.from_tuples([(42, 1)], names=['patient', 'time'])
    lab_names = [
        'FiO2', 'LPM O2', 'PEEP', 'Pressure Support', 'Vent Rate',
        'Respiratory Rate', 'Temperature',
    ]
    resp_names = ['FiO2', 'LPM O2', 'PEEP', 'Pressure Support', 'Vent Rate']
    nurse_names = ['Respiratory Rate', 'Temperature']
    sources = {
        'lab': pd.DataFrame([[1.0] * len(lab_names)], index=index,
                            columns=lab_names),
        'resp': pd.DataFrame([[1.0] * len(resp_names)], index=index,
                             columns=resp_names),
        'nurse': pd.DataFrame([[1.0] * len(nurse_names)], index=index,
                              columns=nurse_names),
    }
    qualified, collisions = timeseries.namespace_overlapping_features(sources)
    assert set(collisions) == set(lab_names), collisions
    assert 'lab__Temperature' in qualified['lab']
    assert 'nurse__Temperature' in qualified['nurse']
    assert 'resp__PEEP' in qualified['resp']


def test_reciprocal_mask_resets_at_each_patient():
    index = pd.MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=['patient', 'time'])
    values = pd.DataFrame({'x': [10.0, np.nan, np.nan, np.nan]}, index=index)
    mask = apply_filters.compute_exponential_decay_mask(values, 4.0 / 3.0)
    assert np.allclose(mask['x'], [1.0, 0.75, 0.0, 0.0]), mask


def test_loader_rejects_duplicate_or_gapped_hours():
    duplicate = pd.DataFrame({'time': [1, 1], 'x': [1.0, 2.0]})
    with pytest.raises(ValueError, match='duplicate'):
        _processed._validated_patient_timeseries(duplicate, 42)

    gap = pd.DataFrame({'time': [1, 3], 'x': [1.0, 2.0]})
    with pytest.raises(ValueError, match='missing'):
        _processed._validated_patient_timeseries(gap, 42)


def test_failed_store_install_restores_the_prior_store():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = root / 'stage1'
        staged = root / '.stage1.building'
        store.mkdir()
        staged.mkdir()
        (store / 'identity.txt').write_text('old')
        (staged / 'identity.txt').write_text('new')

        calls = 0

        def fail_second_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('injected install failure')
            os.replace(source, destination)

        with pytest.raises(OSError, match='injected'):
            _stage1._install_staged_store(staged, store, replace=fail_second_replace)

        assert store.is_dir()
        assert (store / 'identity.txt').read_text() == 'old'


def test_rows_do_not_affect_fitted_timeseries_quantiles():
    base = pd.DataFrame(
        {'time': [1, 2], 'x': [0.0, 10.0]},
        index=pd.Index([1, 1], name='patient'),
    )
    with_test = pd.concat([
        base,
        pd.DataFrame({'time': [1], 'x': [10000.0]},
                     index=pd.Index([2], name='patient')),
    ])
    expected = apply_filters.normalize_and_clip(
        base.copy(), fit_patients=[1], fit_max_hour=2)
    actual = apply_filters.normalize_and_clip(
        with_test.copy(), fit_patients=[1], fit_max_hour=2)
    assert np.allclose(actual.loc[1, 'x'], expected.loc[1, 'x'])


def _flat_rows(test_height):
    rows = []
    for patient, height in [(1, 160.0), (2, 180.0), (3, test_height)]:
        rows.append({
            'patientunitstayid': patient,
            'apacheadmissiondx': 'x',
            'eyes': 1, 'motor': 1, 'verbal': 1, 'dialysis': 0,
            'vent': 0, 'meds': 0, 'intubated': 0, 'bedcount': 1,
            'gender': 'Female', 'teachingstatus': 'f',
            'ethnicity': 'Caucasian', 'unittype': 'MICU',
            'unitadmitsource': 'ER', 'unitvisitnumber': 1,
            'unitstaytype': 'admit', 'physicianspeciality': 'critical care',
            'numbedscategory': '>= 50', 'region': 'Midwest',
            'age': '60', 'admissionheight': height,
            'admissionweight': 80.0, 'hour': 12.0,
        })
    return pd.DataFrame(rows)


def test_rows_do_not_affect_fitted_static_scaling():
    ordinary = flat_and_labels.preprocess_flat(
        _flat_rows(200.0), fit_patients=[1, 2])
    extreme = flat_and_labels.preprocess_flat(
        _flat_rows(10000.0), fit_patients=[1, 2])
    assert np.allclose(
        ordinary.loc[[1, 2], 'admissionheight'],
        extreme.loc[[1, 2], 'admissionheight'],
    )
