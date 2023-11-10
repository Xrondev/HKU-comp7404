import numpy as np
import pandas as pd

wbcd_dataset = pd.read_csv('./dataset/wbcd.data', header=None)
random_state = 0

#print(wbcd_dataset.head(),wdbc_dataset.head())

# wbcd_dataset
wbcd_dataset = wbcd_dataset.drop(0, axis=1)  # drop the id column
# if record contains ? value for any column (feature incomplete), delete the record
incomplete_records = []
for index, row in wbcd_dataset.iterrows():
    if '?' in row.values:
        incomplete_records.append(index)
wbcd_dataset = wbcd_dataset.drop(incomplete_records, axis=0)
#print(f'removed {len(incomplete_records)} incomplete records: {incomplete_records}')

# wbcd partitioning
# 50-50
train_50 = wbcd_dataset.iloc[:len(wbcd_dataset) // 2]
test_50 = wbcd_dataset.iloc[len(wbcd_dataset) // 2:]
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


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def basic_svm_fit(partition: dict, c=None, sigma=None) -> tuple[float, float, float, float]:
    train = partition['train']
    test = partition['test']
    train_x = train.drop(10, axis=1)
    train_y = train[10]
    test_x = test.drop(10, axis=1)
    test_y = test[10]
    if sigma is not None:
        gamma = 1 / (sigma ** 2)
        svm = SVC(C=c, kernel='rbf', random_state=random_state, gamma=gamma)
    else:
        svm = SVC(kernel='rbf', random_state=random_state)
    svm.fit(train_x, train_y)
    pred_y = svm.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    tp, fp, fn, tn = confusion_matrix(test_y, pred_y).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(test_y, pred_y)
    return acc, sensitivity, specificity, auc

#def basic_svm_fit_10cv(partition: dict) -> tuple[float, float, float, float]:
#print('50-50')
#c = np.random.randint(1, 1000)
#sigma = np.random.randint(1, 100)
#acc, sensitivity, specificity, auc = basic_svm_fit(wbcd_partitioned['50-50'], sigma, c)
#print(f'accuracy: {acc}, sensitivity: {sensitivity}, specificity: {specificity}, auc: {auc}')


# Initialize the parameters
# a. Foraging of prey
n = 100  # The population size of whales
num_classes = 2


def initialize_population(population_size: int, c_constraint=(1, 1000), sigma_constraint=(1, 100)) -> np.ndarray:
    """
    Initialize the whales population, each whale is represented by a vector of two elements (c, sigma). Both c and sigma are the hyperparameters of SVM.
    :param population_size: number of whales
    :param c_constraint: (C_min, C_max), default (1, 1000)
    :param sigma_constraint: (sigma_min, sigma_max), default (1, 100)
    :return: ndarray of shape (population_size, 2)
    """
    population = np.zeros((population_size, 2))
    for i in range(population_size):
        population[i][0] = np.random.randint(c_constraint[0], c_constraint[1])
        population[i][1] = np.random.randint(sigma_constraint[0], sigma_constraint[1])
    return population


def apply_constraint(whale, c_constraint=(1, 1000), sigma_constraint=(1, 100)) -> np.ndarray:
    """
    Apply the constraints to the whale
    :param whale: ndarray of shape (2,)
    :param c_constraint: (C_min, C_max), default (1, 1000)
    :param sigma_constraint: (sigma_min, sigma_max), default (1, 100)
    :return: ndarray of shape (2,)
    """
    whale[0] = np.clip(whale[0], c_constraint[0], c_constraint[1])
    whale[1] = np.clip(whale[1], sigma_constraint[0], sigma_constraint[1])
    return whale

rng = np.random.default_rng(seed=random_state)  # random number generator


def whale_optimization_algorithm(partition, population, max_iteration=50, a=2, b=0.5):
    """
    Whale Optimization Algorithm
    :param partition: dataset partition
    :param population: whales population
    :param max_iteration: maximum number of iterations
    :param a: variable used in Eq.2: linearly decreases from 2 (by default =2) to 0
    :param b: *Not mentioned default value in the paper* constant to determine the spiral shape.
    :return:
    """
    a_step = a / max_iteration
    current_iteration = 0
    fitness = np.array([basic_svm_fit(partition, c, sigma)[0] for c, sigma in population])
    print('fitness',fitness)
    best_whale = population[np.argmax(fitness)]
    print('best_whale',best_whale)
    print(np.around(basic_svm_fit(partition, best_whale[0], best_whale[1]),8))


    while current_iteration < max_iteration:
        # using whales C and sigma to train SVMs and calculate the CA as fitness
        for idx, whale in enumerate(population):
            t = rng.random()
            if t < 0.5:
                A = 2 * a * rng.random() - a
                C = 2 * rng.random()
                if np.linalg.norm(A) < 1:
                    # update the position by Eq.1
                    population[idx] = best_whale - A * (C * best_whale - whale)
                else:
                    # update the position by Eq.5
                    population[idx] = population[np.random.randint(0, len(population))] - A * (C * population[np.random.randint(0, len(population))] - whale)
                    pass
            else:
                # update the position by Eq.4, t>=1
                l = 2 * rng.random() - 1
                population[idx] = np.linalg.norm(best_whale - whale) * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
            # check if any search agent goes beyond the search space and amend it.
            population[idx] = apply_constraint(population[idx])

        a -= a_step

        # update the fitness
        fitness = np.array([basic_svm_fit(partition, c, sigma)[0] for c, sigma in population])
        best_whale = population[np.argmax(fitness)]
        current_iteration += 1

    return best_whale

population = initialize_population(n)
best_whale = whale_optimization_algorithm(wbcd_partitioned['50-50'], population)
print(best_whale)
print(basic_svm_fit(wbcd_partitioned['50-50'], best_whale[0], best_whale[1]))
