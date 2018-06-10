#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:05:21 2018

@author: daniel
"""

import matplotlib.pyplot as plt
import math
from random import random, randint

class GA:

    CROSSOVER_ONECUT = 1
    #CROSSOVER_MULCUT = 2
    #CROSSOVER_UNIFORM = 2

    def __init__(self, size_chrom, p_m):
        self.size_chrom = size_chrom
        self.p_m = p_m
        pass

    def Crossover(self, population, method=1):
        size_p = len(population)
        offspring = population[:] # Initial the offspring array
        # Doing crossover for size_p/2 times
        for idx_p in range(size_p//2):
            # Randomly find two individuals to crossover
            idx_fa = -1
            idx_mo = -1
            while True:
                idx_fa = randint(0, size_p-1)
                idx_mo = randint(0, size_p-1)
                if idx_fa == idx_mo:
                    p1 = population[idx_fa]
                    p2 = population[idx_mo]
                    break
            # Do crossover for selected individuals
            c1, c2 = self.crossover(p1, p2, method)
            offspring[idx_fa] = c1
            offspring[idx_mo] = c2
        return offspring

    def Mutation(self, population):
        size_p = len(population)
        mutation = population[:]
        for idx_p in range(size_p):
            mut = population[idx_p]
            # Try to mutate every bit
            for idx_g in range(self.size_chrom-1):
                r = random()
                if r <= self.p_m: # Do the mutation
                    mut = mut[:idx_g]  \
                        + self.flip(mut[idx_g]) \
                        + mut[idx_g+1:]
            mutation[idx_p] = mut
        return mutation

    def Selection(self, population):
        size_p = len(population)
        selection = population[:]
        fitness = self.calc_fitness(population, size_p)

        # Calculate the total fitness
        F = 0
        for idx_p in range(size_p):
            F = F + fitness[idx_p]

        # Calculate the selection probability
        p_k = []
        for idx_p in range(size_p):
            p_k.append(fitness[idx_p] / F)

        # Calculate the cumulative probability
        q_k = []
        q_k.append(p_k[0])
        for idx_p in range(1, size_p):
            q_k.append(q_k[idx_p-1] + p_k[idx_p])

        # Rotate the roulette wheel for size_p times
        for idx_p in range(size_p):
            r = random()
            for idx_q in range(size_p):
                if q_k[idx_q] >= r:
                    selection[idx_p] = population[idx_q]
                    break

        return selection

    def evaluation(self, chromosome):
        pass

    def crossover(self, p1, p2, method):
        if(method == self.CROSSOVER_ONECUT):
            return self.crossover_onecut(p1, p2)
        else:
            raise Exception("No such method.")

    def crossover_onecut(self, p1, p2):
        cut_point = randint(0, self.size_chrom-1)
        offspring1 = p1[:cut_point] + p2[cut_point:]
        offspring2 = p2[:cut_point] + p1[cut_point:]
        return offspring1, offspring2

    def flip(self, bit_char):
        if bit_char == "1":
            return "0"
        else:
            return "1"

    def calc_fitness(self, population, size_p=-1):
        if size_p == -1:
            size_p = len(population)

        fitness = []
        for idx_p in range(size_p):
            fitness.append(self.evaluation(population[idx_p]))

        return fitness


class numGA(GA):

    a1 = -3.0
    b1 = 12.1
    a2 = 4.1
    b2 = 5.8
    preci_factor = 10000
    x1_bits = 18
    x2_bits = 15
    size_chrom = x1_bits + x2_bits

    def __init__(self, size_p, p_m, max_gen):
        self.local_max = []
        self.global_max = []
        self.global_max_curr = 0
        self.initPopulation(size_p)
        self.p_m = p_m
        self.max_gen = max_gen
        super().__init__(self.size_chrom, self.p_m)

    def run(self, prnt=False):
        for i in range(self.max_gen):
            selection = self.Selection(self.population)
            offspring = self.Crossover(selection)
            mutation = self.Mutation(offspring)

            self.population = mutation
            self.output(prnt)

    def initPopulation(self, size_p):
        population = []
        for idx_p in range(size_p):
            x1 = self.a1 + random()*(self.b1 - self.a1)
            x2 = self.a2 + random()*(self.b2 - self.a2)
            population.append(self.encode(x1, x2))
        self.population = population

    def encode(self, x1, x2):
        x1_offset = x1 - self.a1
        x2_offset = x2 - self.a2

        x1_encode = "{0:018b}".format(int(x1_offset * self.preci_factor))
        x2_encode = "{0:015b}".format(int(x2_offset * self.preci_factor))

        return x1_encode + x2_encode

    def decode(self, chromosome):
        x1_code = chromosome[:18]
        x2_code = chromosome[18:]

        x1 = self.a1 + int(x1_code, 2) * (self.b1 - self.a1) / (2**self.x1_bits - 1)
        x2 = self.a2 + int(x2_code, 2) * (self.b2 - self.a2) / (2**self.x2_bits - 1)

        return x1, x2

    def evaluation(self, chromosome):
        x1, x2 = self.decode(chromosome)
        return 21.5 + \
            x1 * math.sin(4 * math.pi * x1) + \
            x2 * math.sin(20 * math.pi * x2)

    def output(self, prnt):
        lm_index, lm_value = max(enumerate(self.calc_fitness(self.population)),
                                 key=lambda p: p[1])

        if lm_value > self.global_max_curr:
            self.global_max_curr = lm_value
            self.solution = self.decode(self.population[lm_index])

        self.local_max.append(lm_value)
        self.global_max.append(self.global_max_curr)

        if(prnt):
            print(self.global_max_curr)

if __name__ == "__main__":
    ga = numGA(100, 0.02, 1000) # population_size, probability_mutation, max_generation
    ga.run(prnt=True)

    print("The best solution of this run is {}, at {}".format(ga.global_max_curr, ga.solution))

    plt.plot(ga.local_max, 'b', ga.global_max, 'r')
    plt.show()
