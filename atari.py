import numpy as np
from numpy.core.numeric import Inf
import gym

from genes import *
from geneticOptimizer import *
from atari_env import *
from atari_help import *

import multiprocessing as mp

import json

import sys

class Game:
    def __init__(self):
        self.environment = AtariEnv(game="boxing")

    def battle(self, ind):
        fitness = run_game(self.environment, ind, False)
        return fitness + 20

    def calculateFitness(self, population, _):
        all = []
        for list in population:
            for ind in list["individuals"]:
                all.append(ind)
        num_threads = int(mp.cpu_count() - 3)
        pool = mp.Pool(num_threads)
        res = pool.map(self.battle, all)
        for ind, fitness in zip(all, res):
            ind.setFitness(fitness)
        pool.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fg = open("atari_best.json", "r")
        fm = open("atari_meta.json", "r")
        meta = Genes.Metaparameters.load(fm)
        fm.close()
        base = Genes.load(fg, meta)
        fg.close()
        game = Game()
        run_game(game.environment, base, True, 999999999999999)
        game.environment.close()

    else:
        inputs = 128
        outputs = 4
        base = Genes(inputs, outputs, Genes.Metaparameters(perturbation_chance=0.5, perturbation_stdev=0.5, new_link_weight_stdev=4, c1=8, c2=8, c3=0.8))
        population = [base.clone() for i in range(150)]
        for ind in population:
            for output_node_index in range(3):
                for input_node_index in range(inputs):
                    ind.add_connection(ind.input_node_index(input_node_index), ind.output_node_index(output_node_index))
                ind.add_connection(Genes.BIAS_INDEX, ind.output_node_index(output_node_index))
        game = Game()
        optimizer = GeneticOptimizer(population, game, 100)
        optimizer.initialize()
        optimizer.evolve()
        best = optimizer.getBestIndividual()
        population = optimizer.getPopulation()

        f = open("atari_best.json", "w")
        best.save(f)
        f.close()
        f = open("atari_population.json", "w")
        all_inds = []
        for species in optimizer.getPopulation():
            for individual in species["individuals"]:
                all_inds.append(individual)
        f.write(json.dumps([ind.as_json() for ind in all_inds]))
        f.close()
        f = open("atari_meta.json", "w")
        best._metaparameters.save(f)
        f.close()

        environment = AtariEnv(game="boxing")
        fitness = run_game(environment, best, True, 99999999999999999999999)
        environment.close()
