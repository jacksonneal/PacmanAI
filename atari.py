import numpy as np
from numpy.core.numeric import Inf
import gym

from genes import *
from geneticOptimizer import *
from atari_env import *
from atari_help import *

import multiprocessing as mp

class Game:
    def __init__(self):
        self.num_threads = int(mp.cpu_count() - 1)

    def battle(self, ind):
        environment = AtariEnv(game="boxing")
        fitness = run_game(environment, ind, False)
        return fitness

    def calculateFitness(self, population, _):
        all = []
        for list in population:
            for ind in list["individuals"]:
                all.append(ind)
        pool = mp.Pool(self.num_threads)
        res = pool.map(self.battle, all)
        for ind, fitness in zip(all, res):
            ind.setFitness(fitness)
        pool.close()

if __name__ == "__main__":
    inputs = 128
    outputs = 4
    base = Genes(inputs, outputs, Genes.Metaparameters(perturbation_chance=0.5, perturbation_stdev=0.5, new_link_weight_stdev=4, c1=2, c2=2, c3=0.4))
    population = [base.clone() for i in range(150)]
    for ind in population:
        for output_node_index in range(3):
            for input_node_index in range(inputs):
                ind.add_connection(ind.input_node_index(input_node_index), ind.output_node_index(output_node_index))
            ind.add_connection(Genes.BIAS_INDEX, ind.output_node_index(output_node_index))
    game = Game()
    optimizer = GeneticOptimizer(population, game, 50)
    optimizer.initialize()
    optimizer.evolve()
    best = optimizer.getBestIndividual()
    population = optimizer.getPopulation()

    environment = AtariEnv(game="boxing")
    fitness = run_game(environment, best, True, 99999999999999999999999)
    environment.close()
    
    f = open("best_atari.json", "w")
    best.save(f)
    f.close()

    
