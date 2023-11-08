import numpy as np

from config import random_state
from svm import basic_svm_fit

rng = np.random.default_rng(random_state)


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
    print('fitness', fitness)
    best_whale = population[np.argmax(fitness)]
    print('best_whale', best_whale)
    print(np.around(basic_svm_fit(partition, best_whale[0], best_whale[1]), 8))

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
                    population[idx] = population[np.random.randint(0, len(population))] - A * (
                            C * population[np.random.randint(0, len(population))] - whale)
                    pass
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


def initialize_population(population_size: int, num_classes: int, c_constraint=(1, 1000),
                          sigma_constraint=(1, 100)) -> np.ndarray:
    """
    Initialize the whales population, each whale is represented by a vector of two elements (c, sigma).
    Both c and sigma are the hyperparameters of SVM.
    :param num_classes: number of classes in each whale
    :param population_size: number of whales
    :param c_constraint: (C_min, C_max), default (1, 1000)
    :param sigma_constraint: (sigma_min, sigma_max), default (1, 100)
    :return: ndarray of shape (population_size, 2)
    """
    population = np.zeros((population_size, num_classes))
    for i in range(population_size):
        population[i][0] = np.random.randint(c_constraint[0], c_constraint[1])
        population[i][1] = np.random.randint(sigma_constraint[0], sigma_constraint[1])
    return population


def main():
    n = 50  # The population size of whales
    num_classes = 2

    from dataset import wbcd_partitioned
    population = initialize_population(n, num_classes)
    best_whale = whale_optimization_algorithm(wbcd_partitioned['50-50'], population)
    print(best_whale)
    print(basic_svm_fit(wbcd_partitioned['50-50'], best_whale[0], best_whale[1]))


if __name__ == '__main__':
    main()
