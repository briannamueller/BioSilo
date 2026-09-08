"""Filter, normalize, mask, and impute binned eICU time series.

Reads ``binned_timeseries.parquet`` and writes
``preprocessed_timeseries.csv``. Parameter-sensitive transformations are kept
separate from the cached binning step.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .._events import find_table


LENGTH_LIMIT_HOURS = 24 * 14  # 14-day cap on stay length


def validate_dense_hours(df: pd.DataFrame) -> None:
    """Require one sorted, consecutive positive-hour sequence per patient."""
    if df.duplicated(['patient', 'time']).any():
        duplicate_count = int(df.duplicated(['patient', 'time'], keep=False).sum())
        raise ValueError(
            f'binned timeseries contains {duplicate_count} rows with duplicate '
            '(patient, time) keys; rebuild it with the source join')

    ordered = df.sort_values(['patient', 'time'], kind='stable')
    first = ordered.groupby('patient', sort=False)['time'].first()
    if not first.eq(1).all():
        bad = first[first.ne(1)].index[:5].tolist()
        raise ValueError(f'binned timeseries must start at hour 1; bad patients: {bad}')
    gaps = ordered.groupby('patient', sort=False)['time'].diff().dropna()
    if not gaps.eq(1).all():
        raise ValueError('binned timeseries has missing or unsorted patient-hours')


# ─────────────────────────────────────────────────────────────────────
# Prevalence filter (within-hospital × cross-hospital)
# ─────────────────────────────────────────────────────────────────────

def load_hospital_map(eicu_dir: Path) -> dict:
    """Return the stay-to-hospital map from the raw patient table."""
    candidate = find_table(eicu_dir, 'patient')
    if candidate is None:
        raise FileNotFoundError(
            f'patient.csv not found under {eicu_dir}. Searched '
            f'case-insensitively, plain and gzipped.')
    df = pd.read_csv(candidate, usecols=['patientunitstayid', 'hospitalid'])
    return df.set_index('patientunitstayid')['hospitalid'].to_dict()


def apply_prevalence_filter(df: pd.DataFrame, hospital_map: dict,
                            within_threshold: float,
                            cross_threshold: float,
                            fit_patients=None,
                            fit_max_hour=None) -> pd.DataFrame:
    """Drop feature columns failing the double-threshold prevalence filter.

    A feature is kept if, in at least ``cross_threshold`` fraction of
    hospitals, the within-hospital prevalence (fraction of that hospital's
    patients with ≥1 non-null observation) is ≥ ``within_threshold``.
    """
    feature_cols = [c for c in df.columns if c != 'time']
    print(f'==> Prevalence filter ({len(feature_cols)} candidate features, '
          f'within≥{within_threshold}, cross≥{cross_threshold})')

    fit_df = df if fit_patients is None else df.loc[df.index.isin(fit_patients)]
    if fit_max_hour is not None:
        fit_df = fit_df.loc[fit_df['time'] <= fit_max_hour]
    if fit_df.empty:
        raise ValueError('prevalence fitting received no training patients')

    # Per-patient presence: True if the feature has any non-null value.
    presence = fit_df[feature_cols].notnull().groupby(level='patient').any()

    hospitals = presence.index.to_series().map(hospital_map)
    if hospitals.isnull().any():
        missing = int(hospitals.isnull().sum())
        print(f'  [warn] {missing} patients have no hospital mapping; dropping from filter computation')
        presence = presence[hospitals.notnull()]
        hospitals = hospitals.dropna()

    # Within-hospital prevalence per feature.
    within_prev = presence.groupby(hospitals.values).mean()
    # Fraction of hospitals where within-hospital prevalence ≥ threshold.
    hospitals_pass = (within_prev >= within_threshold).mean(axis=0)
    keep = hospitals_pass[hospitals_pass >= cross_threshold].index.tolist()

    dropped = len(feature_cols) - len(keep)
    print(f'  Kept {len(keep)} / {len(feature_cols)} features ({dropped} dropped)')
    return df[['time'] + keep]


# ─────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────

def normalize_and_clip(df: pd.DataFrame, fit_patients=None,
                       fit_max_hour=None) -> pd.DataFrame:
    """Fit 5/95 quantiles on training-visible rows, then transform all rows."""
    feature_cols = [c for c in df.columns if c != 'time']
    fit = df
    if fit_patients is not None:
        fit = fit.loc[fit.index.isin(fit_patients)]
    if fit_max_hour is not None:
        fit = fit.loc[fit['time'] <= fit_max_hour]
    if fit.empty:
        raise ValueError('normalization fitting received no training-visible rows')
    print(f'==> Computing training-only 5/95 quantiles on '
          f'{len(feature_cols)} features...')
    quantiles = fit[feature_cols].quantile([0.05, 0.95])
    mins = quantiles.loc[0.05]
    maxs = quantiles.loc[0.95]
    spread = (maxs - mins).replace(0, np.nan)

    df[feature_cols] = 2 * (df[feature_cols] - mins) / spread - 1
    df[feature_cols] = df[feature_cols].clip(lower=-4, upper=4)
    return df


# ─────────────────────────────────────────────────────────────────────
# Masks (vectorized exponential decay or binary)
# ─────────────────────────────────────────────────────────────────────

def compute_exponential_decay_mask(df: pd.DataFrame, decay_rate: float) -> pd.DataFrame:
    """Reciprocal measurement-freshness mask, reset per patient.

    Observed values receive 1, missing values before the first observation
    receive 0, and later missing runs receive
    ``1 / max(missing_hours * rate, 1)``.
    """
    observed = df.notnull()
    patient_ids = df.index.get_level_values('patient')
    result = pd.DataFrame(0.0, index=df.index, columns=df.columns)

    for column in df.columns:
        obs = observed[column]
        observations_so_far = obs.astype(np.int64).groupby(patient_ids).cumsum()
        missing = (~obs).astype(np.int64)
        missing_run = missing.groupby([patient_ids, observations_so_far]).cumsum()
        has_prior_observation = observations_so_far.gt(0)
        divisor = (missing_run.astype(float) * decay_rate).clip(lower=1.0)
        result[column] = np.where(
            obs, 1.0,
            np.where(has_prior_observation, 1.0 / divisor, 0.0),
        )
    return result


def compute_mask(df: pd.DataFrame, mode: str, decay_rate: float) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != 'time']
    features = df[feature_cols]
    if mode == 'binary':
        mask = features.notnull().astype(float)
    elif mode == 'exponential_decay':
        mask = compute_exponential_decay_mask(features, decay_rate)
    else:
        raise ValueError(f'Unknown mask mode: {mode!r}')
    mask.columns = [f'{c}_mask' for c in mask.columns]
    return mask


# ─────────────────────────────────────────────────────────────────────
# Forward-fill and length limit
# ─────────────────────────────────────────────────────────────────────

def forward_fill_and_limit(df: pd.DataFrame) -> pd.DataFrame:
    """Clip time > 0, time < LENGTH_LIMIT, forward-fill within patients, zero-fill."""
    df = df[(df['time'] > 0) & (df['time'] < LENGTH_LIMIT_HOURS)]
    patient_ids = df.index.get_level_values('patient').values
    feature_cols = [c for c in df.columns if c != 'time']
    df[feature_cols] = df[feature_cols].groupby(patient_ids).ffill()
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────
# Time-of-day feature
# ─────────────────────────────────────────────────────────────────────

def add_time_of_day(df: pd.DataFrame, flat_path: str) -> pd.DataFrame:
    """Attach a normalized time-of-day feature per row.

    ``flat_features.csv`` already stores the admit hour-of-day as ``hour``
    (computed in extract_tables). We add that to the ICU-relative hour
    offset, mod 24, and map onto a normalized [0, 1] position.
    """
    print('==> Adding time-of-day feature...')
    flat = pd.read_csv(flat_path, usecols=['patientunitstayid', 'hour'])
    hour_map = flat.set_index('patientunitstayid')['hour'].to_dict()
    hour_list = np.linspace(0, 1, 24)

    pid_col = df.index.get_level_values('patient')
    admit_hour = pid_col.to_series().map(hour_map).to_numpy(dtype=np.float64)
    combined = df['time'].to_numpy(dtype=np.float64) + admit_hour
    idx = np.where(np.isnan(combined), 0, combined.astype(np.int64) % 24 - 24)
    tod = np.where(np.isnan(combined), 0.0, hour_list[idx])
    df['hour'] = tod
    return df


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def apply_filters_main(data_dir: str, eicu_dir: str,
                       within_threshold: float, cross_threshold: float,
                       mask_mode: str, decay_rate: float,
                       fit_patients=None, fit_max_hour=None) -> None:
    if not data_dir.endswith('/'):
        data_dir += '/'
    eicu_path = Path(eicu_dir)

    print(f'==> Loading binned_timeseries.parquet...')
    df = pd.read_parquet(data_dir + 'binned_timeseries.parquet')
    validate_dense_hours(df)
    df = df.sort_values(['patient', 'time'], kind='stable').set_index('patient')
    print(f'  Loaded {len(df)} rows, {df.shape[1] - 1} feature columns')

    # A MultiIndex (patient, time) is needed for the groupby-by-level passes
    # below; time stays a column throughout for simple slicing.
    df.index.name = 'patient'

    hospital_map = load_hospital_map(eicu_path)
    df = apply_prevalence_filter(
        df, hospital_map, within_threshold, cross_threshold,
        fit_patients=fit_patients, fit_max_hour=fit_max_hour)
    df = normalize_and_clip(
        df, fit_patients=fit_patients, fit_max_hour=fit_max_hour)

    # Build a (patient, time) MultiIndex view for the mask/ffill passes.
    df = df.set_index('time', append=True)
    df.index.names = ['patient', 'time']

    mask = compute_mask(df, mode=mask_mode, decay_rate=decay_rate)
    df = pd.concat([df, mask], axis=1)
    df = df.reset_index(level='time')

    df = forward_fill_and_limit(df)
    df = add_time_of_day(df, data_dir + 'flat_features.csv')

    out_path = data_dir + 'preprocessed_timeseries.csv'
    print(f'==> Writing {out_path}...')
    df.to_csv(out_path)
    print(f'==> Done: {len(df)} rows, {df.shape[1]} columns')

    # Replace the pre-filter stay manifest with patients retained after clipping.
    surviving = sorted(int(p) for p in df.index.unique())
    stays_path = data_dir + 'stays.txt'
    print(f'==> Writing {stays_path} with {len(surviving)} surviving patients...')
    with open(stays_path, 'w') as f:
        for p in surviving:
            f.write(f'{p}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1c: apply filters, normalization and masks.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing binned_timeseries.parquet and flat_features.csv.')
    parser.add_argument('--eicu-dir', required=True,
                        help='Raw eICU directory (for hospital mapping).')
    parser.add_argument('--within-prev', type=float, default=0.25,
                        help='Within-hospital prevalence threshold (default 0.25).')
    parser.add_argument('--cross-prev', type=float, default=0.70,
                        help='Cross-hospital coverage threshold (default 0.70).')
    parser.add_argument('--mask-mode', choices=['binary', 'exponential_decay'],
                        default='exponential_decay',
                        help='Mask computation mode (default exponential_decay).')
    parser.add_argument('--decay-rate', type=float, default=4.0 / 3.0,
                        help='Decay rate for exponential_decay mask (default 4/3).')
    args = parser.parse_args()
    apply_filters_main(args.data_dir, args.eicu_dir,
                       args.within_prev, args.cross_prev,
                       args.mask_mode, args.decay_rate)
