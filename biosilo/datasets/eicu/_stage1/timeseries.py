"""Bin raw eICU time-series sources by patient and hour.

The cached ``binned_timeseries.parquet`` contains raw feature values with NaN
for unmeasured fields. Parameter-sensitive transformations run later.
"""
import argparse
import gc
import json
import os
from itertools import islice
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


LAST_INCLUDED_HOUR = 24 * 14 - 1
FEATURE_MANIFEST = 'timeseries_feature_manifest.json'


def normalize_fio2_percent(values):
    """Return clinically plausible FiO2 values on a percentage scale."""
    values = pd.to_numeric(values, errors='coerce').astype(float)
    out = values.copy()
    fraction = out.between(0.2, 1.0, inclusive='both')
    percent = out.between(20.0, 100.0, inclusive='both')
    out.loc[fraction] *= 100.0
    out.loc[~(fraction | percent)] = np.nan
    return out


def normalize_temperature_celsius(values, allow_fahrenheit=True):
    """Return plausible temperatures in Celsius, optionally converting F."""
    values = pd.to_numeric(values, errors='coerce').astype(float)
    out = values.copy()
    celsius = out.between(25.0, 45.0, inclusive='both')
    fahrenheit = out.between(77.0, 113.0, inclusive='both')
    if allow_fahrenheit:
        out.loc[fahrenheit] = (out.loc[fahrenheit] - 32.0) * 5.0 / 9.0
    out.loc[~(celsius | (fahrenheit & allow_fahrenheit))] = np.nan
    return out


def normalize_long_feature(df, feature_column, value_column):
    """Apply the few unit conversions supported by explicit eICU policies."""
    labels = df[feature_column].astype(str)
    fio2 = labels.str.casefold().eq('fio2')
    temperature = labels.str.casefold().eq('temperature')
    if fio2.any():
        df.loc[fio2, value_column] = normalize_fio2_percent(
            df.loc[fio2, value_column])
    if temperature.any():
        df.loc[temperature, value_column] = normalize_temperature_celsius(
            df.loc[temperature, value_column])
    return df


def namespace_overlapping_features(sources):
    """Qualify cross-source name collisions instead of silently combining them.

    Non-overlapping names retain their compact source spelling. Only names
    present in more than one source become ``source__feature``.
    """
    owners = {}
    for source, frame in sources.items():
        for column in frame.columns:
            owners.setdefault(str(column), []).append(source)

    collisions = {name: sorted(srcs) for name, srcs in owners.items()
                  if len(srcs) > 1}
    renamed = {}
    for source, frame in sources.items():
        mapping = {column: f'{source}__{column}' for column in frame.columns
                   if str(column) in collisions}
        renamed[source] = frame.rename(columns=mapping)
    return renamed, collisions


def load_stay_end_hours(eicu_path, patients):
    """Map extracted cohort stays to their last modeled positive hour."""
    labels = pd.read_csv(
        os.path.join(eicu_path, 'labels.csv'),
        usecols=['patientunitstayid', 'unitdischargeoffset'],
    )
    labels = labels[labels['patientunitstayid'].isin(patients)].copy()
    labels['end_hour'] = np.ceil(
        pd.to_numeric(labels['unitdischargeoffset'], errors='coerce') / 60.0
    )
    labels['end_hour'] = labels['end_hour'].clip(
        lower=0, upper=LAST_INCLUDED_HOUR)
    labels = labels.dropna(subset=['end_hour'])
    return labels.set_index('patientunitstayid')['end_hour'].astype(int).to_dict()


def dense_hour_index(stay_end_hours, patients):
    """Construct positive consecutive patient hours for one patient chunk."""
    patient_values = []
    hour_values = []
    for patient in patients:
        end = int(stay_end_hours.get(int(patient), 0))
        if end < 1:
            continue
        patient_values.extend([int(patient)] * end)
        hour_values.extend(range(1, end + 1))
    return pd.MultiIndex.from_arrays(
        [patient_values, hour_values], names=['patient', 'time'])


def merge_patient_hours(sources, stay_end_hours, patients):
    """Outer-join source features onto one dense row per patient-hour."""
    grid = pd.DataFrame(index=dense_hour_index(stay_end_hours, patients))
    patient_set = set(int(p) for p in patients)
    for frame in sources.values():
        available = frame.index.get_level_values('patient').isin(patient_set)
        if available.any():
            grid = grid.join(frame.loc[available], how='left', validate='one_to_one')
    grid.sort_index(inplace=True)
    if not grid.index.is_unique:
        raise AssertionError('binned eICU data contains duplicate patient-hours')
    return grid


def reconfigure_timeseries_fast(df, offset_column, feature_column=None, test=False):
    """Convert long observations to a wide ``(patient, hour)`` table."""
    if test:
        df = df.iloc[300000:5000000]

    df['hour'] = np.ceil(df[offset_column] / 60).astype(int)
    df.drop(columns=[offset_column], inplace=True)

    if feature_column is not None:
        df = df.groupby(['patientunitstayid', 'hour', feature_column]).mean().reset_index()
        df = df.pivot_table(
            index=['patientunitstayid', 'hour'],
            columns=feature_column,
            values=df.columns.difference(
                ['patientunitstayid', 'hour', feature_column]
            ).tolist(),
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df.columns.name = None
    else:
        df = df.groupby(['patientunitstayid', 'hour']).mean()

    df.index.names = ['patient', 'time']
    return df


def gen_patient_chunk(patients, size=1000):
    it = iter(patients)
    chunk = list(islice(it, size))
    while chunk:
        yield chunk
        chunk = list(islice(it, size))


def bin_sources(eICU_path, test=False):
    """Load each raw timeseries source, bin to (patient, hour), return dict."""
    print('==> Loading and binning timeseries sources...')
    read_kw = {'nrows': 500000} if test else {}

    print('==> Lab...')
    ts_lab = pd.read_csv(eICU_path + 'timeserieslab.csv', **read_kw)
    ts_lab = normalize_long_feature(ts_lab, 'labname', 'labresult')
    ts_lab = reconfigure_timeseries_fast(ts_lab,
                                         offset_column='labresultoffset',
                                         feature_column='labname',
                                         test=test)
    print(f'    Lab shape: {ts_lab.shape}')
    gc.collect()

    print('==> Respiratory...')
    ts_resp = pd.read_csv(eICU_path + 'timeseriesresp.csv', low_memory=False, **read_kw)
    ts_resp = ts_resp.replace('%', '', regex=True)
    ts_resp['respchartvalue'] = pd.to_numeric(ts_resp['respchartvalue'], errors='coerce')
    ts_resp = ts_resp.loc[ts_resp['respchartvalue'].notnull()]
    ts_resp = normalize_long_feature(
        ts_resp, 'respchartvaluelabel', 'respchartvalue')
    ts_resp = reconfigure_timeseries_fast(ts_resp,
                                          offset_column='respchartoffset',
                                          feature_column='respchartvaluelabel',
                                          test=test)
    print(f'    Resp shape: {ts_resp.shape}')
    gc.collect()

    print('==> Nurse...')
    ts_nurse = pd.read_csv(eICU_path + 'timeseriesnurse.csv', **read_kw)
    ts_nurse['nursingchartvalue'] = pd.to_numeric(ts_nurse['nursingchartvalue'], errors='coerce')
    ts_nurse = ts_nurse.loc[ts_nurse['nursingchartvalue'].notnull()]
    ts_nurse = normalize_long_feature(
        ts_nurse, 'nursingchartcelltypevallabel', 'nursingchartvalue')
    ts_nurse = reconfigure_timeseries_fast(ts_nurse,
                                           offset_column='nursingchartoffset',
                                           feature_column='nursingchartcelltypevallabel',
                                           test=test)
    print(f'    Nurse shape: {ts_nurse.shape}')
    gc.collect()

    print('==> Aperiodic...')
    ts_aperiodic = pd.read_csv(eICU_path + 'timeseriesaperiodic.csv', **read_kw)
    ts_aperiodic = reconfigure_timeseries_fast(ts_aperiodic,
                                               offset_column='observationoffset',
                                               test=test)
    print(f'    Aperiodic shape: {ts_aperiodic.shape}')
    gc.collect()

    print('==> Periodic...')
    ts_periodic = pd.read_csv(eICU_path + 'timeseriesperiodic.csv', **read_kw)
    if 'temperature' in ts_periodic:
        ts_periodic['temperature'] = normalize_temperature_celsius(
            ts_periodic['temperature'], allow_fahrenheit=False)
    ts_periodic = reconfigure_timeseries_fast(ts_periodic,
                                              offset_column='observationoffset',
                                              test=test)
    print(f'    Periodic shape: {ts_periodic.shape}')
    gc.collect()

    return {
        'lab': ts_lab,
        'resp': ts_resp,
        'nurse': ts_nurse,
        'periodic': ts_periodic,
        'aperiodic': ts_aperiodic,
    }


def write_binned(eICU_path, sources, test=False):
    """Write one dense, sorted row per patient-hour.

    Cross-source column collisions remain source-qualified. This deliberately
    avoids inventing a clinical precedence rule merely because labels match.
    """
    if not test:
        import pyarrow as pa
        import pyarrow.parquet as pq

    base = Path(eICU_path)
    out_path = base / 'binned_timeseries.parquet'
    stays_path = base / 'stays.txt'
    manifest_path = base / FEATURE_MANIFEST
    token = uuid4().hex
    tmp_out = base / f'.binned_timeseries.{token}.parquet.tmp'
    tmp_stays = base / f'.stays.{token}.txt.tmp'
    tmp_manifest = base / f'.{FEATURE_MANIFEST}.{token}.tmp'

    sources, collisions = namespace_overlapping_features(sources)
    feature_cols = sorted({str(c) for df in sources.values() for c in df.columns})
    patients = sorted({int(patient)
                       for frame in sources.values()
                       for patient in frame.index.get_level_values('patient').unique()})
    stay_end_hours = load_stay_end_hours(eICU_path, patients)

    size = 20000
    writer = None
    all_stays = []
    try:
        for i, chunk in enumerate(gen_patient_chunk(patients, size=size), start=1):
            merged = merge_patient_hours(sources, stay_end_hours, chunk)
            if merged.empty:
                continue
            merged.reset_index(level='time', inplace=True)

            # Normalize to a fixed schema: patient (int64), time (int32), then
            # all feature columns as float32 in sorted order.
            merged = merged.reindex(columns=['time'] + feature_cols)
            merged['time'] = merged['time'].astype(np.int32)
            merged[feature_cols] = merged[feature_cols].astype(np.float32)
            merged = merged.reset_index()  # 'patient' → column
            merged['patient'] = merged['patient'].astype(np.int64)

            all_stays.extend(merged['patient'].unique().tolist())

            if not test:
                table = pa.Table.from_pandas(merged, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_out, table.schema,
                                              compression='snappy')
                writer.write_table(table)

            print(f'==> Wrote chunk {i}: {len(chunk)} patients, running total '
                  f'{min(i * size, len(patients))}/{len(patients)}')
            del merged

        if not test and writer is None:
            raise RuntimeError('No eligible positive-hour eICU stays were binned')
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        for path in (tmp_out, tmp_stays, tmp_manifest):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if writer is not None:
            writer.close()

    stays = sorted(set(int(s) for s in all_stays))
    with tmp_stays.open('w') as f:
        for s in stays:
            f.write(f'{s}\n')

    manifest = {
        'schema_version': 1,
        'row_key': ['patient', 'time'],
        'hours': {'first': 1, 'last': LAST_INCLUDED_HOUR, 'dense': True},
        'collisions': collisions,
        'collision_policy': 'source-qualified; no implicit cross-source merge',
        'unit_policies': {
            'FiO2': 'percent; 0.2..1 converted x100; 20..100 retained',
            'Temperature': (
                'Celsius; lab/nurse 77..113 Fahrenheit converted; '
                'periodic documented-Celsius values only'
            ),
        },
        'features': feature_cols,
    }
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    if not test:
        os.replace(tmp_out, out_path)
    os.replace(tmp_stays, stays_path)
    os.replace(tmp_manifest, manifest_path)
    print(f'==> Wrote stays.txt with {len(stays)} patients')


def timeseries_main(eICU_path, test=False):
    """Produce cached ``binned_timeseries.parquet`` and ``stays.txt``."""
    parquet_path = eICU_path + 'binned_timeseries.parquet'
    if not test and os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
        print(f'[cached] {parquet_path} already present, skipping binning.')
        return
    if not test:
        try:
            os.remove(eICU_path + 'binned_timeseries.csv')
        except FileNotFoundError:
            pass
    sources = bin_sources(eICU_path, test=test)
    write_binned(eICU_path, sources, test=test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1b: bin eICU timeseries.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing intermediate CSVs from extract_tables.py.')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with limited data.')
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.endswith('/'):
        data_dir += '/'
    timeseries_main(data_dir, test=args.test)
