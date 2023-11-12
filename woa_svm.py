import numpy as np
from matplotlib import pyplot as plt

from algo_util import apply_constraint, initialize_population
from config import random_state
from dataset import wbcd_partitioned, wdbc_partitioned
from svm import basic_svm_fit

rng = np.random.default_rng(random_state)


def whale_optimization_algorithm(partition, population, max_iteration=50, a=2, b=0.5):
    """
    Whale Optimization Algorithm
    :param partition: dataset partition
    :param population: whales population
    :param max_iteration: maximum number of iterations
    :param a: variable used in Eq.2: linearly decreases from 2 (by default=2) to 0
    :param b: *Not mentioned default value in the paper* constant to determine the spiral shape.
    :return:
    """
    a_step = a / max_iteration
    current_iteration = 0
    fitness = np.array([basic_svm_fit(partition, c, sigma)[0] for c, sigma in population])
    best_whale = population[np.argmax(fitness)]

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
                    random_whale = population[np.random.randint(0, len(population))]
                    population[idx] = random_whale - A * (C * random_whale - whale)
            else:
                # update the position by Eq.4, t>=1
                l = 2 * rng.random() - 1
                population[idx] = np.linalg.norm(best_whale - whale) * np.exp(b * l) * np.cos(
                    2 * np.pi * l) + best_whale
            # check if any search agent goes beyond the search space and amend it.
            population[idx] = apply_constraint(population[idx])

        a -= a_step

        # update the fitness
        fitness = np.array([basic_svm_fit(partition, c, sigma)[0] for c, sigma in population])
        best_whale = population[np.argmax(fitness)]
        current_iteration += 1

    return best_whale


def main():
    n = 50  # The population size of whales
    num_classes = 2
    rng = np.random.default_rng()  # Create a random number generator instance

    normal_svm_c = rng.random() * 1000
    normal_svm_sigma = rng.random() * 100
    print(f'c: {normal_svm_c}, sigma: {normal_svm_sigma}')

    result_svm = {}
    result_woa = {}

    for name, d in {'wbcd': wbcd_partitioned, 'wdbc': wdbc_partitioned}.items():
        for p in ('50-50', '60-40', '10-CV'):
            normal_svm = basic_svm_fit(d[p], normal_svm_c, normal_svm_sigma)
            result_svm[f'{name}{p}'] = normal_svm
            print(f'svm {name} {p}: {normal_svm}')

            population = initialize_population(n, num_classes)
            best_whale = whale_optimization_algorithm(d[p], population, max_iteration=50)
            result_woa[f'{name}{p}'] = basic_svm_fit(d[p], *best_whale)
            print(f'{name} {p} best whale: {best_whale}')
            print(basic_svm_fit(d[p], *best_whale))

    # Plotting results
    metrics = ['Accuracy', 'Specificity', 'Sensitivity', 'AUC']
    for metric_index, metric_name in enumerate(metrics):
        plot_results(result_svm, result_woa, metric_name, metric_index)


def plot_results(result_svm, result_woa, metric_name, metric_index):
    plt.figure(figsize=(16, 10))
    plt.title(f'{metric_name} comparison between WOA and SVM')
    plt.xlabel('Dataset')
    plt.ylabel(metric_name)

    for idx, (name, d) in enumerate({'SVM': result_svm, 'WOA': result_woa}.items()):
        plt.bar(np.arange(len(d)) + idx * 0.2, [v[metric_index] for v in d.values()], width=0.2, label=name)

    plt.xticks(np.arange(len(d)) + 0.1, d.keys())
    plt.yticks(np.arange(0.80, 1.00, 0.01))
    plt.ylim(0.80, 1.00)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
