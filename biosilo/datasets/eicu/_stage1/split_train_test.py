import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import os
import argparse


def create_person_partitions(path, train_ratio=0.75, seed=1):
    """Assign each person once before any fitted preprocessing runs."""
    labels = pd.read_csv(os.path.join(path, 'labels.csv'),
                         usecols=['patientunitstayid', 'uniquepid'])
    stays_path = os.path.join(path, 'stays.txt')
    if os.path.exists(stays_path):
        with open(stays_path) as handle:
            surviving = {int(line.strip()) for line in handle if line.strip()}
        labels = labels.loc[labels['patientunitstayid'].isin(surviving)]
    persons = np.asarray(sorted(labels['uniquepid'].astype(str).unique()))
    if len(persons) < 2:
        raise ValueError('eICU preprocessing requires at least two people')
    rng = np.random.default_rng(seed)
    rng.shuffle(persons)
    n_train = int(round(len(persons) * train_ratio))
    n_train = min(max(n_train, 1), len(persons) - 1)
    train_people = set(persons[:n_train])
    train = labels.loc[
        labels['uniquepid'].astype(str).isin(train_people),
        'patientunitstayid',
    ].astype(int).tolist()
    test = labels.loc[
        ~labels['uniquepid'].astype(str).isin(train_people),
        'patientunitstayid',
    ].astype(int).tolist()
    return {'train': train, 'test': test}


def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def shuffle_stays(stays, seed=9):
    return shuffle(stays, random_state=seed)

def process_table(table_name, table, stays, folder_path):
    table = table.loc[stays].copy()
    # Preserve the on-disk loader contract explicitly after index selection.
    table.index.name = 'patient'
    if table_name == 'timeseries':
        # Parquet for the bulk timeseries: much faster for downstream stages to
        # re-load than the CSV equivalent, and far smaller on disk.
        table.to_parquet('{}/{}.parquet'.format(folder_path, table_name))
    else:
        table.to_csv('{}/{}.csv'.format(folder_path, table_name))
    return

def split_train_test(path, is_test=True, seed=9, cleanup=True, MIMIC=False,
                     partitions=None):

    labels = pd.read_csv(path + 'preprocessed_labels.csv')
    labels.set_index('patient', inplace=True)
    # Split by unique patient identifier so no patient crosses into both the
    # train and the test sets.
    # np.asarray: pandas 3 backs string columns with pyarrow, and unique()
    # then returns an arrow-backed array that sklearn cannot fancy-index
    # ("only integer scalar arrays can be converted to a scalar index").
    if partitions is None:
        patients = np.asarray(labels.uniquepid.unique())
        train, test = train_test_split(patients, test_size=0.15, random_state=seed)
        train, val = train_test_split(train, test_size=0.15/0.85, random_state=seed)
        partition_stays = {
            name: labels.loc[labels['uniquepid'].isin(people)].index
            for name, people in zip(['train', 'val', 'test'], [train, val, test])
        }
    else:
        partition_stays = {
            name: pd.Index(stays).intersection(labels.index)
            for name, stays in partitions.items()
        }

    print('==> Loading data for splitting...')
    if is_test:
        timeseries = pd.read_csv(path + 'preprocessed_timeseries.csv', nrows=999999)
    else:
        timeseries = pd.read_csv(path + 'preprocessed_timeseries.csv')
    timeseries.set_index('patient', inplace=True)
    if not MIMIC:
        diagnoses = pd.read_csv(path + 'preprocessed_diagnoses.csv')
        diagnoses.set_index('patient', inplace=True)
    flat_features = pd.read_csv(path + 'preprocessed_flat.csv')
    flat_features.set_index('patient', inplace=True)

    # The source files are not needed once the splits are written.
    if is_test is False and cleanup:
        print('==> Removing the unsorted data...')
        os.remove(path + 'preprocessed_timeseries.csv')
        if not MIMIC:
            os.remove(path + 'preprocessed_diagnoses.csv')
        os.remove(path + 'preprocessed_labels.csv')
        os.remove(path + 'preprocessed_flat.csv')

    for partition_name, stays in partition_stays.items():
        print('==> Preparing {} data...'.format(partition_name))
        folder_path = create_folder(path, partition_name)
        with open(folder_path + '/stays.txt', 'w') as f:
            for stay in stays:
                f.write("%s\n" % stay)
        stays = shuffle_stays(stays, seed=9)
        if MIMIC:
            for table_name, table in zip(['labels', 'flat', 'timeseries'],
                                         [labels, flat_features, timeseries]):
                process_table(table_name, table, stays, folder_path)
        else:
            for table_name, table in zip(['labels', 'flat', 'diagnoses', 'timeseries'],
                                         [labels, flat_features, diagnoses, timeseries]):
                process_table(table_name, table, stays, folder_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split preprocessed eICU data into train/val/test.')
    parser.add_argument('--data-dir', required=True,
                        help='Path to directory containing preprocessed CSVs.')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove unsorted preprocessed files after splitting.')
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.endswith('/'):
        data_dir += '/'
    split_train_test(data_dir, is_test=False, cleanup=args.cleanup)
