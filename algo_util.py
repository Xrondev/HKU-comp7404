import numpy as np
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


def apply_constraint(individual, c_constraint=(1, 1000), sigma_constraint=(1, 100)) -> np.ndarray:
    """
    Apply the constraints to the whale
    :param individual: ndarray of shape (2,)
    :param c_constraint: (C_min, C_max), default (1, 1000)
    :param sigma_constraint: (sigma_min, sigma_max), default (1, 100)
    :return: ndarray of shape (2,)
    """
    individual[0] = np.clip(individual[0], c_constraint[0], c_constraint[1])
    individual[1] = np.clip(individual[1], sigma_constraint[0], sigma_constraint[1])
    return individual
