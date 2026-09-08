import argparse
import pandas as pd
import numpy as np


def add_codes(splits, codes_dict, words_dict, count):
    codes = list()
    levels = len(splits)  # the max number of levels is 6
    if levels >= 1:
        try:
            codes.append(codes_dict[splits[0]][0])
            codes_dict[splits[0]][2] += 1
        except KeyError:
            codes_dict[splits[0]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0]
            count += 1
    if levels >= 2:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][0])
            codes_dict[splits[0]][1][splits[1]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1]
            count += 1
    if levels >= 3:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2]
            count += 1
    if levels >= 4:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3]
            count += 1
    if levels >= 5:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3] + '|' + splits[4]
            count += 1
    if levels == 6:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3] + '|' + splits[4] + '|' + splits[5]
            count += 1
    return codes, count


def get_mapping_dict(unique_diagnoses):

    # Note strings are largely repetitive, so coding stops at the organ-system level.
    main_diagnoses = [a for a in unique_diagnoses if not (a.startswith('notes') or a.startswith('admission'))]
    adm_diagnoses = [a for a in unique_diagnoses if a.startswith('admission diagnosis')]
    pasthistory_organsystems = [a for a in unique_diagnoses if a.startswith('notes/Progress Notes/Past History/Organ Systems/')]
    pasthistory_comments = [a for a in unique_diagnoses if a.startswith('notes/Progress Notes/Past History/Past History Obtain Options')]

    # sort into alphabetical order to keep the codes roughly together numerically.
    main_diagnoses.sort()
    adm_diagnoses.sort()
    pasthistory_organsystems.sort()
    pasthistory_comments.sort()

    mapping_dict = {}
    codes_dict = {}
    words_dict = {}
    count = 0

    for diagnosis in main_diagnoses:
        splits = diagnosis.split('|')
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    for diagnosis in adm_diagnoses:
        # Drop the prefix shared by every string in this group; it adds a level
        # that carries no information.
        shortened = diagnosis.replace('admission diagnosis|', '')
        shortened = shortened.replace('All Diagnosis|', '')
        shortened = shortened.replace('Additional APACHE  Information|', '')
        splits = shortened.split('|')
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_organsystems:
        # Drop the prefix shared by every string in this group; it adds a level
        # that carries no information.
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Organ Systems/', '')
        splits = shortened.split('/')  # note different split to main_diagnoses
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_comments:
        # Drop the prefix shared by every string in this group; it adds a level
        # that carries no information.
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Past History Obtain Options/', '')
        splits = shortened.split('/')  # note different split to main_diagnoses
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    return codes_dict, mapping_dict, count, words_dict

# Drop codes that parent exactly one child (index 2 is 1).
def find_pointless_codes(diag_dict):
    pointless_codes = []
    for key, value in diag_dict.items():
        # if there is only one child, then the branch is linear and can be condensed
        if value[2] == 1:
            pointless_codes.append(value[0])
        # Drop duplicates where parent and child carry the same title.
        for next_key, next_value in value[1].items():
            if key.lower() == next_key.lower():
                pointless_codes.append(next_value[0])
        pointless_codes += find_pointless_codes(value[1])
    return pointless_codes

# Drop codes below the prevalence cut-off.
def find_rare_codes(cut_off, sparse_df):
    prevalence = sparse_df.sum(axis=0)
    rare_codes = prevalence.loc[prevalence <= cut_off].index
    return list(rare_codes)

def within_window(diagnoses, max_offset_minutes):
    """Keep diagnoses available strictly before the configured cutoff."""
    if 'diagnosis_event_offset' not in diagnoses.columns:
        raise ValueError(
            'diagnoses.csv lacks diagnosis_event_offset; rebuild the stage-one store')
    offsets = pd.to_numeric(
        diagnoses['diagnosis_event_offset'], errors='coerce')
    return diagnoses.loc[offsets < max_offset_minutes].drop(
        columns=['diagnosis_event_offset'])


def diagnoses_main(
    eICU_path,
    cut_off_prevalence,
    fit_patients=None,
    max_offset_minutes=300,
):

    print('==> Loading data diagnoses.csv...')
    diagnoses = pd.read_csv(eICU_path + 'diagnoses.csv')
    diagnoses = within_window(diagnoses, max_offset_minutes)
    diagnoses.set_index('patientunitstayid', inplace=True)

    fit_set = (set(diagnoses.index.unique()) if fit_patients is None
               else set(int(p) for p in fit_patients))
    fit_diagnoses = diagnoses.loc[diagnoses.index.isin(fit_set)]
    if fit_diagnoses.empty:
        raise ValueError('diagnosis fitting received no training diagnoses')
    unique_diagnoses = fit_diagnoses.diagnosisstring.unique()
    codes_dict, mapping_dict, count, words_dict = get_mapping_dict(unique_diagnoses)

    patients = diagnoses.index.unique()
    index_to_patients = dict(enumerate(patients))
    patients_to_index = {v: k for k, v in index_to_patients.items()}

    # reconfiguring the diagnosis data into a dictionary
    diagnoses = diagnoses.groupby('patientunitstayid').apply(lambda diag: diag.to_dict(orient='list')['diagnosisstring']).to_dict()
    diagnoses = {
        patient: [code for diag in list_diag for code in mapping_dict.get(diag, [])]
        for (patient, list_diag) in diagnoses.items()
    }

    num_patients = len(patients)
    sparse_diagnoses = np.zeros(shape=(num_patients, count))
    for patient, codes in diagnoses.items():
        sparse_diagnoses[patients_to_index[patient], codes] = 1  # repeats in `codes` are harmless

    pointless_codes = find_pointless_codes(codes_dict)

    sparse_df = pd.DataFrame(sparse_diagnoses, index=patients, columns=range(count))
    fit_sparse = sparse_df.reindex(sorted(fit_set), fill_value=0)
    cut_off = round(cut_off_prevalence * len(fit_set))
    rare_codes = find_rare_codes(cut_off, fit_sparse)
    drop_codes = sorted(set(rare_codes + pointless_codes) & set(sparse_df.columns))
    sparse_df.drop(columns=drop_codes, inplace=True)
    sparse_df.rename(columns=words_dict, inplace=True)
    print('==> Keeping ' + str(sparse_df.shape[1]) + ' diagnoses which have a prevalence of more than ' + str(cut_off_prevalence*100) + '%...')

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
    sparse_df = sparse_df.reindex(ts_patients)

    # make naming consistent with the other tables
    sparse_df.rename_axis('patient', inplace=True)
    sparse_df.sort_index(inplace=True)
    sparse_df.fillna(0, inplace=True)  # make sure all values are filled in

    print('==> Saving finalised preprocessed diagnoses...')
    sparse_df.to_csv(eICU_path + 'preprocessed_diagnoses.csv')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess eICU diagnoses data.')
    parser.add_argument('--data-dir', required=True,
                        help='Path to directory containing intermediate CSVs.')
    parser.add_argument('--cut-off-prevalence', type=float, default=0.01,
                        help='Minimum prevalence to keep a diagnosis code (default: 0.01 = 1%%).')
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.endswith('/'):
        data_dir += '/'
    diagnoses_main(data_dir, args.cut_off_prevalence)
