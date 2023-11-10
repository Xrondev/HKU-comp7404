# Dragonfly Algorithm
import numpy as np

from algo_util import initialize_population
from config import random_state

rng = np.random.default_rng(random_state)


def update_step_vector():
    return NotImplementedError


def dragonfly_algorithm():
    return NotImplementedError

def separation_alignment_cohesion(population, radius):
    separations = np.zeros((len(population), 2))
    alignments = np.zeros((len(population), 2))
    cohesions = np.zeros((len(population), 2))

def initialize_dragonflies(n_swarm, c_constraint, sigma_constraint):
    dragonflies = []
    for _ in range(n_swarm):
        dragonfly = {
            'position': np.array([np.random.uniform(*c_constraint), np.random.uniform(*sigma_constraint)]),
            'velocity': np.zeros(2),
            'fitness': None
        }
        dragonflies.append(dragonfly)
    return dragonflies
def main():
    population = initialize_population(50, 2)
    return NotImplementedError


if __name__ == '__main__':
    main()
