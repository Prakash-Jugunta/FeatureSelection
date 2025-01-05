import random

import numpy as np
from model import init_pso


class BinaryPSO:
    def __init__(self, num_particles, num_dimensions, max_iter, rate):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iter = max_iter
        self.particles = self.init_p()
        self.pbest = np.copy(self.particles)
        self.gbest = np.copy(self.particles[np.argmin(self.particles.sum(axis=1))])
        self.rate = rate
        self.P = np.zeros((self.num_particles, self.num_dimensions))

    def init_p(self):
        lst = []
        for i in range(self.num_particles):
            l = init_pso(self.rate)
            lst.append(l)
        return np.array(lst)

    def fitness(self, x, acc):
        # 这里定义你的适应度函数，例如一个简单的和函数
        return 0.1 * len(x) / self.num_dimensions + (1 - acc) * 0.9

    def caculate_P(self):
        rd = np.random.rand(self.num_particles, self.num_dimensions) * (1 - self.rate)
        cognitive = self.pbest - self.particles
        social = self.gbest - self.particles
        self.P = rd + np.abs(cognitive) + np.abs(social)

    def update_position(self):
        rd = random.random()
        self.particles[self.particles > rd] = 1 - self.particles

    def update_pbest(self):
        fitness = self.fitness(self.particles)
        better_mask = fitness < self.fitness(self.pbest)
        self.pbest[better_mask] = self.particles[better_mask]

    def update_gbest(self):
        global_fitness = self.fitness(self.particles)
        best_idx = np.argmin(global_fitness)
        if global_fitness[best_idx] < self.fitness(self.gbest):
            self.gbest = self.particles[best_idx]

    def run(self):
        for _ in range(self.max_iter):
            self.caculate_P()
            self.update_position()
            self.update_pbest()
            self.update_gbest()

        best_solution = self.gbest
        best_fitness = self.fitness(best_solution)
        return best_solution, best_fitness
