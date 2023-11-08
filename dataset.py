import pandas as pd
random_state = 0


wbcd_dataset = pd.read_csv('./dataset/wbcd.data', header=None)
wdbc_dataset = pd.read_csv('./dataset/wdbc.data', header=None)

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
train_10cv = wbcd_dataset.copy()
test_10cv = []
for i in range(10):
    test_10cv.append(train_10cv.sample(frac=0.1, random_state=(random_state + i)))

wbcd_partitioned = {
    '50-50': {
        'train': train_50,
        'test': test_50
    },
    '60-40': {
        'train': train_60,
        'test': test_60
    },
    '10-CV': {
        'train': train_10cv,
        'test': test_10cv
    }
}


def show_wbcd_statistic_data(dataset) -> None:
    print(f'number of records: {len(dataset)}')
    print(f'B: {len(dataset[dataset[10] == 2])}')
    print(f'M: {len(dataset[dataset[10] == 4])}')


for key, val in wbcd_partitioned.items():
    if key == '10-CV':
        print(f'10-CV')
        for i in range(10):
            print(f'fold {i + 1}')
            show_wbcd_statistic_data(val['test'][i])
    else:
        print(key)
        print('Train set')
        show_wbcd_statistic_data(val['train'])
        print('Test set')
        show_wbcd_statistic_data(val['test'])