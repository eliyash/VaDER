import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DAYS_ORDERED = Path(r"../data/complied_days_ordered")

FILE_PATH = r"../data/stard_all.csv"
# SUBJECT_ID_KEY = 'src_subject_id'
# SUBJECT_ID_KEY = 'promoted_subjectkey'
SUBJECT_ID_KEY = 'subjectkey'
QSTOTS = 'qstots'
DAY_KEY = 'days_baseline'
WEEK_KEY = 'week'


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def save_premade(output_path, w_train, x_train, names):
    with open(str(output_path / 'names.txt'), 'w') as f:
        f.write(','.join(names))
    np.save(str(output_path / 'w_train.npy'), w_train)
    np.save(str(output_path / 'x_train.npy'), x_train)


def read_premade(output_path):
    with open(str(output_path / 'names.txt')) as f:
        names = f.read().split(',')
    w_train = np.load(str(output_path / 'w_train.npy'))
    x_train = np.load(str(output_path / 'x_train.npy'))
    return w_train, x_train, names


def read_data():
    with open(FILE_PATH) as file:
        file_iterator = csv.reader(file)
        names = list(next(file_iterator))
        samples = list(file_iterator)

    patient_value_index = names.index(SUBJECT_ID_KEY)
    day_value_index = names.index(DAY_KEY)

    # vss = [name for name in names if name.startswith('vs')]

    samples_with_date_and_id = [sample for sample in samples if sample[patient_value_index] and sample[day_value_index]]
    all_by_ind = [[sample[ind] for sample in samples_with_date_and_id] for ind in range(len(names))]

    # too_meany_by_ind = [len(set(var)) for var in all_by_ind]
    # to_ignore_indices_too_meany_values = {ind for ind, count in enumerate(too_meany_by_ind) if count > 100}

    empty_count_by_ind = [var.count('') for var in all_by_ind]
    to_ignore_indices_missing = {ind for ind, count in enumerate(empty_count_by_ind) if count > len(samples_with_date_and_id) * 0.6}

    # to_ignore_indices = to_ignore_indices_too_meany_values.union(to_ignore_indices_missing)
    to_ignore_indices = to_ignore_indices_missing

    non_int_values = [[var for var in set(var) if var != '' and not isfloat(var)] for var in all_by_ind]
    non_int_values_indices = {ind for ind, var in enumerate(non_int_values) if len(var) > 0}

    non_int_values_indices = non_int_values_indices.difference(to_ignore_indices)

    dict_name_to_value = {index: list(set(all_by_ind[index])) for index in non_int_values_indices}
    samples_with_numeric_values = [
        [dict_name_to_value[ind].index(var) if (ind in dict_name_to_value and var != '') else var for ind, var in enumerate(sample)]
        for sample in samples_with_date_and_id
    ]

    samples_with_relevant_values = [[var for ind, var in enumerate(sample) if ind not in to_ignore_indices] for sample in samples_with_numeric_values]
    names_reduced = [var for ind, var in enumerate(names) if ind not in to_ignore_indices]
    patient_value_index = names_reduced.index(SUBJECT_ID_KEY)
    day_value_index = names_reduced.index(DAY_KEY)

    patient_samples_dict = defaultdict(list)
    for sample in samples_with_relevant_values:
        patient_samples_dict[sample[patient_value_index]].append(sample)

    patient_samples_ordered_dict = {k: sorted(v, key=lambda d: float(sample[day_value_index])) for k, v in patient_samples_dict.items()}
    a = [sorted(v, key=lambda d: float(sample[day_value_index])) for k, v in patient_samples_dict.items()]
    b = [[int(float(sample[day_value_index])) for sample in p_samples] for p_samples in a]
    res_per_offset = []
    for offset in range(7):
        c = sorted([{(day + offset)//7 for day in patient} for patient in b], key=lambda x: -len(x))
        res_per_offset.append(c)
    res_per_offset_lens = [[len(sample) for sample in patient_samples] for patient_samples in res_per_offset]
    res_per_offset_num_of_min = [list(map(lambda x: min(x, 7), patient_samples)).count(7) for patient_samples in res_per_offset_lens]
    number_of_times = 6
    samples = [patient_samples[:number_of_times] for patient_samples in patient_samples_ordered_dict.values() if len(patient_samples) >= number_of_times]

    # for patient_samples in patient_samples_ordered_dict.values():
    # max_days_of_sample = max(map(len, patient_samples_ordered_dict.values()))
    #     patient_samples.extend([[''] * len(names_reduced)] * (max_days_of_sample-len(patient_samples)))
    w_train = np.array([[[0 if var == '' else 1 for var in day] for day in sample] for sample in samples], dtype='float32')
    x_train = np.array([[[float(var) if var != '' else 0 for var in day] for day in sample] for sample in samples], dtype='float32')

    filter_permanent_values(names_reduced, w_train, x_train)

    return w_train, x_train, names_reduced


def filter_permanent_values(names_reduced, w_train, x_train):
    all_time_series_index = set()
    for sample in np.stack([x_train.swapaxes(1, 2), w_train.swapaxes(1, 2)], axis=3):
        for i, days_of_feature in enumerate(sample):
            feature_values = {feature for feature, valid in days_of_feature if valid}
            if len(feature_values) > 1:
                all_time_series_index.add(i)
    names_times = [names_reduced[ind] for ind in all_time_series_index]


def main():
    w_train, x_train, names = read_data()
    save_premade(DAYS_ORDERED, w_train, x_train, names)


if __name__ == '__main__':
    main()
