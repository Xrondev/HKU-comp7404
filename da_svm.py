import numpy as np
from scipy.special import gamma
from dataset import wbcd_partitioned
from svm import basic_svm_fit
from config import random_state

rng = np.random.default_rng(random_state)


def objective_function(dataset, position):
    # We try minimizing the objective function: in this case, minimize the Error rate
    return 1 - basic_svm_fit(dataset, position[0], position[1])[0]


class Dragonfly:
    def __init__(self, dim, bounds):
        self.position = rng.uniform(bounds[:, 0], bounds[:, 1], dim)
        self.step = np.zeros(dim)
        self.velocity = np.zeros(dim)
        self.bounds = bounds

    def update_position(self, new_position):
        # Correcting the new positions based on the boundaries of variables
        self.position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def levy_flight(self, beta):
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = rng.normal(0, sigma, len(self.position))
        v = rng.normal(0, 1, len(self.position))
        step = 0.01 * u / (np.abs(v) ** (1 / beta))
        return step


class DragonflySwarm:
    def __init__(self, pop_size, bounds, max_iter, dataset):
        self.dragonflies = [Dragonfly(len(bounds), bounds) for _ in range(pop_size)]
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.iter = 0
        self.radius = np.mean(bounds[:, 1] - bounds[:, 0]) / 4
        self.beta = 1.5  # Levy flight parameter
        self.dataset = dataset

        # Initialize global best and worst with infinity and negative infinity
        self.food_fitness = np.inf
        self.enemy_fitness = -np.inf
        self.food_position = np.zeros(len(bounds))
        self.enemy_position = np.zeros(len(bounds))

    def update_parameters(self):
        # Here you could update w, s, a, c, f, and e based on iteration or other logic
        # For demonstration purposes, we'll use fixed values for these weights
        self.w = 0.5  # Inertia weight
        self.s = 0.2  # Separation weight
        self.a = 0.2  # Alignment weight
        self.c = 0.7  # Cohesion weight
        self.f = 1  # Food factor
        self.e = 1  # Enemy factor

        # Update the neighbouring radius as per the given definition
        count = self.iter
        self.radius = (self.bounds[:, 1] - self.bounds[:, 0]) / 4 + \
                      ((self.bounds[:, 1] - self.bounds[:, 0]) * (count / self.max_iter) * 2)

    def calculate_distance(self, d1_position, d2_position):
        # Calculate the distance between two dragonflies in each dimension
        return np.abs(d1_position - d2_position)

    def get_neighbours(self, dragonfly):
        # Find neighbours within the radius for each dimension
        neighbours = []
        for other in self.dragonflies:
            if other == dragonfly:  # Skip the same dragonfly
                continue
            distance = self.calculate_distance(dragonfly.position, other.position)
            if np.all(distance < self.radius):  # Check if within radius for all dimensions
                neighbours.append(other)
        return neighbours

    def optimize(self):
        while self.iter < self.max_iter:
            # Calculate the fitness of all dragonflies
            for dragonfly in self.dragonflies:
                fitness = objective_function(self.dataset, dragonfly.position)
                if fitness < self.food_fitness:  # Update food source
                    self.food_fitness = fitness
                    self.food_position = dragonfly.position.copy()
                if fitness > self.enemy_fitness:  # Update enemy
                    self.enemy_fitness = fitness
                    self.enemy_position = dragonfly.position.copy()

            # Update parameters
            self.update_parameters()

            # Calculate behaviors and update velocities and positions
            for dragonfly in self.dragonflies:
                neighbours = self.get_neighbours(dragonfly)
                if neighbours:  # If there are neighbours
                    # Calculate the behavior using equations provided
                    separation = -np.sum([dragonfly.position - neighbour.position for neighbour in neighbours], axis=0)
                    alignment = np.sum([neighbour.velocity for neighbour in neighbours], axis=0) / len(neighbours)
                    cohesion = np.sum([neighbour.position for neighbour in neighbours], axis=0) / len(
                        neighbours) - dragonfly.position
                    food_attraction = self.food_position - dragonfly.position
                    enemy_distraction = self.enemy_position + dragonfly.position

                    # Update the step vector using Eq. (3.6)
                    dragonfly.step = (self.s * separation + self.a * alignment + self.c * cohesion +
                                      self.f * food_attraction + self.e * enemy_distraction) + self.w * dragonfly.step

                    # Update the position using Eq. (3.7)
                    new_position = dragonfly.position + dragonfly.step
                    dragonfly.update_position(new_position)
                else:  # If no neighbours, use levy flight
                    # Update position using Levy flight equation (Eq. 3.8)
                    new_position = dragonfly.position + dragonfly.levy_flight(self.beta) * dragonfly.position
                    dragonfly.update_position(new_position)

            # This is the end of an iteration
            self.iter += 1
            # print(
            #     f'Dragonfly population at iteration {self.iter}:{[dragonfly.position for dragonfly in self.dragonflies]}')

        # At the end of the optimization, you can return the best solution found
        return self.food_position, self.food_fitness


# Define min and max values for each dimension
bounds = np.array([[1, 1000], [1, 100]])  # Example with different ranges for two dimensions

# Parameters
pop_size = 50  # Population size
max_iter = 50  # Maximum number of iterations

def main():
    from dataset import wbcd_partitioned, wdbc_partitioned

    normal_svm_c = rng.random() * 1000
    normal_svm_sigma = rng.random() * 100
    print(f'c: {normal_svm_c}, sigma: {normal_svm_sigma}')
    for name, d in {'wbcd': wbcd_partitioned, 'wdbc': wdbc_partitioned}.items():
        for p in ('50-50', '60-40', '10-CV'):
            normal_svm = basic_svm_fit(d[p], normal_svm_c, normal_svm_sigma)
            print(f'svm {name} {p}: {normal_svm}')

    for name, d in {'wbcd': wbcd_partitioned, 'wdbc': wdbc_partitioned}.items():
        for p in ('50-50', '60-40', '10-CV'):
            swarm = DragonflySwarm(pop_size, bounds, max_iter, d[p])
            best_position, best_fitness = swarm.optimize()
            print(f'{name} {p} best position: {best_position}:{1 - best_fitness}')
            print(basic_svm_fit(d[p], *best_position))


if __name__ == '__main__':
    main()