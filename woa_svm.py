import numpy as np

from algo_util import apply_constraint, initialize_population
from config import random_state
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

    from dataset import wbcd_partitioned, wdbc_partitioned

    for name, d in {'wbcd': wbcd_partitioned, 'wdbc': wdbc_partitioned}.items():
        for p in ('50-50', '60-40', '10-CV'):
            population = initialize_population(n, num_classes)
            best_whale = whale_optimization_algorithm(d[p], population, max_iteration=50)
            print(f'{name} {p} best whale: {best_whale}')
            print(basic_svm_fit(d[p], best_whale[0], best_whale[1]))


if __name__ == '__main__':
    main()
