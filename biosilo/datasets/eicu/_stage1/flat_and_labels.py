import argparse

import numpy as np
import pandas as pd

#: Categorical values appearing fewer than this many times are collapsed into
#: 'misc'. An absolute count, tuned for full eICU; see preprocess_flat.
RARE_VALUE_MIN_COUNT = 1000


def preprocess_flat(flat, fit_patients=None):

    # make naming consistent with the other tables
    flat.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat.set_index('patient', inplace=True)

    # admission diagnosis is dealt with in diagnoses.py not flat features
    flat.drop(columns=['apacheadmissiondx'], inplace=True)

    # drop apache variables as these aren't available until 24 hours into the stay
    flat.drop(columns=['eyes', 'motor', 'verbal', 'dialysis', 'vent', 'meds', 'intubated', 'bedcount'], inplace=True)

    fit_mask = (np.ones(len(flat), dtype=bool) if fit_patients is None
                else flat.index.isin(set(fit_patients)))
    if not fit_mask.any():
        raise ValueError('flat-feature fitting received no training patients')

    # Assignment through a column view is incompatible with pandas 3.
    flat['gender'] = flat['gender'].map(
        {'Male': 1.0, 'Female': 0.0, 'Other': 0.5, 'Unknown': 0.5})
    flat['teachingstatus'] = flat['teachingstatus'].map({'t': 1.0, 'f': 0.0})

    cat_features = ['ethnicity', 'unittype', 'unitadmitsource', 'unitvisitnumber', 'unitstaytype',
                                         'physicianspeciality', 'numbedscategory', 'region']
    # Collapse rare values into one bucket.
    #
    # RARE_VALUE_MIN_COUNT is an absolute count tuned for full eICU's ~200k
    # stays. On a small cohort almost every value falls below it and the
    # one-hot features come out degenerate, as in the 2,520-stay demo.
    # Override it if you preprocess a subset.
    for f in cat_features:
        counts = flat.loc[fit_mask, f].value_counts(dropna=False)
        known = counts[counts >= RARE_VALUE_MIN_COUNT].index.tolist()
        flat[f] = flat[f].astype(object)
        flat.loc[~flat[f].isin(known), f] = 'misc'
        categories = [value for value in known if pd.notna(value)]
        if 'misc' not in categories:
            categories.append('misc')
        flat[f] = pd.Categorical(flat[f], categories=categories)

    # convert the categorical features to one-hot
    flat = pd.get_dummies(flat, columns=cat_features)

    # eICU records ages over 89 as the string '> 89'. Convert to a number and
    # keep the fact in a separate indicator column.
    age_text = flat['age'].astype('string')
    flat['> 89'] = age_text.str.contains('> 89', na=False).astype(int)
    flat['age'] = pd.to_numeric(age_text.str.replace('> ', '', regex=False),
                                errors='coerce')
    age_mean = flat.loc[fit_mask, 'age'].mean()
    flat['age'] = flat['age'].fillna(age_mean)

    # Timeseries-derived features arrive already normalized. Standardize the
    # roughly normal flat features and min-max the rest.
    features_for_standardisation = ['admissionheight']
    means = flat.loc[fit_mask, features_for_standardisation].mean(axis=0)
    stds = flat.loc[fit_mask, features_for_standardisation].std(axis=0).replace(0, 1)
    flat[features_for_standardisation] = (flat[features_for_standardisation] - means) / stds

    features_for_min_max = ['admissionweight', 'age', 'hour']

    def scale_min_max(values):
        quantiles = values.loc[fit_mask].quantile([0.05, 0.95])
        maxs = quantiles.loc[0.95]
        mins = quantiles.loc[0.05]
        spread = (maxs - mins).replace(0, 1)
        return 2 * (values - mins) / spread - 1

    flat[features_for_min_max] = scale_min_max(flat[features_for_min_max])

    # Clip outliers. Values sit roughly in [-1, 1], so +-4 leaves three units of
    # headroom either side of the normal range.
    flat[features_for_standardisation] = flat[features_for_standardisation].clip(lower=-4, upper=4)
    flat[features_for_min_max] = flat[features_for_min_max].clip(lower=-4, upper=4)

    # Missing weight and height are zero-filled, with an indicator column each so
    # an imputed value is distinguishable from a measured one.
    flat['nullweight'] = flat['admissionweight'].isnull().astype(int)
    flat['nullheight'] = flat['admissionheight'].isnull().astype(int)
    flat['admissionweight'] = flat['admissionweight'].fillna(0)
    flat['admissionheight'] = flat['admissionheight'].fillna(0)
    # Unknown gender sits midway between the two encoded values.
    flat['gender'] = flat['gender'].fillna(0.5)

    return flat

def preprocess_labels(labels):

    # make naming consistent with the other tables
    labels.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    labels.set_index('patient', inplace=True)

    labels = pd.get_dummies(labels, columns=['unitdischargelocation', 'unitdischargestatus'])

    labels['actualhospitalmortality'] = labels['actualhospitalmortality'].replace(
        {'EXPIRED': 1, 'ALIVE': 0})

    return labels

def flat_and_labels_main(eICU_path, fit_patients=None):

    print('==> Loading data from labels and flat features files...')
    flat = pd.read_csv(eICU_path + 'flat_features.csv')
    flat = preprocess_flat(flat, fit_patients=fit_patients)
    flat.sort_index(inplace=True)
    labels = pd.read_csv(eICU_path + 'labels.csv')
    labels = preprocess_labels(labels)
    labels.sort_index(inplace=True)

    # filter out any patients that don't have timeseries
    try:
        with open(eICU_path + 'stays.txt', 'r') as f:
            ts_patients = [int(patient.rstrip()) for patient in f.readlines()]
    except FileNotFoundError:
        ts_patients = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
        ts_patients = [x for x in ts_patients.patient.unique()]
        with open(eICU_path + 'stays.txt', 'w') as f:
            for patient in ts_patients:
                f.write("%s\n" % patient)
    flat = flat.reindex(ts_patients).copy()
    labels = labels.reindex(ts_patients).copy()

    print('==> Saving finalised preprocessed labels and flat features...')
    flat.to_csv(eICU_path + 'preprocessed_flat.csv')
    labels.to_csv(eICU_path + 'preprocessed_labels.csv')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess eICU flat features and labels.')
    parser.add_argument('--data-dir', required=True,
                        help='Path to directory containing intermediate CSVs.')
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.endswith('/'):
        data_dir += '/'
    flat_and_labels_main(data_dir)
