import pandas as pd
import os

from sklearn.model_selection import KFold

random_state = 0
working_directory = os.getcwd()

wbcd_dataset = pd.read_csv(working_directory + './dataset/wbcd.data', header=None)
wdbc_dataset = pd.read_csv(working_directory + './dataset/wdbc.data', header=None)

# wbcd_dataset
wbcd_dataset = wbcd_dataset.drop(0, axis=1)  # drop the id column
# if record contains ? value for any column (feature incomplete), delete the record
incomplete_records = []
for index, row in wbcd_dataset.iterrows():
    if '?' in row.values:
        incomplete_records.append(index)
wbcd_dataset = wbcd_dataset.drop(incomplete_records, axis=0)
print(f'removed {len(incomplete_records)} incomplete records: {incomplete_records}')

# wbcd partitioning
# 50-50
train_50 = wbcd_dataset.sample(frac=0.5, random_state=random_state)
test_50 = wbcd_dataset.drop(train_50.index)
# train_50 = wbcd_dataset.iloc[:len(wbcd_dataset) // 2]
# test_50 = wbcd_dataset.iloc[len(wbcd_dataset) // 2:]
# 60-40
train_60 = wbcd_dataset.sample(frac=0.6, random_state=random_state)
test_60 = wbcd_dataset.drop(train_60.index)
# 10-CV
kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
for train_index, test_index in kf.split(wbcd_dataset):
    train_10cv = wbcd_dataset.iloc[train_index]
    test_10cv = wbcd_dataset.iloc[test_index]


wbcd_partitioned = {
    '50-50': {
        'train': train_50,
        'test': test_50
    },
    '60-40': {
        'train': train_60,
        'test': test_60
    },
    '10-CV': wbcd_dataset.copy()
}


def show_wbcd_statistic_data(dataset) -> None:
    print(f'number of records: {len(dataset)}')
    print(f'B: {len(dataset[dataset[10] == 2])}')
    print(f'M: {len(dataset[dataset[10] == 4])}')


for key, val in wbcd_partitioned.items():
    if key == '10-CV':
        print(f'10-CV')
    else:
        print(key)
        print('Train set')
        show_wbcd_statistic_data(val['train'])
        print('Test set')
        show_wbcd_statistic_data(val['test'])

# -------------------------------------------------------
# WDBC dataset
wdbc_dataset = wdbc_dataset.drop(0, axis=1)  # drop the id column
wdbc_dataset.iloc[:, 0] = wdbc_dataset.iloc[:, 0].replace('M', 2)
wdbc_dataset.iloc[:, 0] = wdbc_dataset.iloc[:, 0].replace('B', 4)

# Moving the 1st column (ground truth label) to the last
cols = list(wdbc_dataset.columns)
cols = cols[1:] + cols[:1]
wdbc_dataset = wdbc_dataset[cols]
wdbc_dataset[cols[-1]] = wdbc_dataset[cols[-1]].astype(int)

# wdbc partitioning
# 50-50
train_50 = wdbc_dataset.sample(frac=0.5, random_state=random_state)
test_50 = wdbc_dataset.drop(train_50.index)
# 60-40
train_60 = wdbc_dataset.sample(frac=0.6, random_state=random_state)
test_60 = wdbc_dataset.drop(train_60.index)
# 10-CV
kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
for train_index, test_index in kf.split(wdbc_dataset):
    train_10cv = wdbc_dataset.iloc[train_index]
    test_10cv = wdbc_dataset.iloc[test_index]
wdbc_partitioned = {
    '50-50': {
        'train': train_50,
        'test': test_50
    },
    '60-40': {
        'train': train_60,
        'test': test_60
    },
    '10-CV': wdbc_dataset.copy()
}
