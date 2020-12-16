import numpy as np
from numpy.core.fromnumeric import mean
from numpy.core.numeric import Inf
import gym

from genes import *
from geneticOptimizer import *
from atari_env import *

import multiprocessing as mp
import math
import json

import sys

class Game:
    def __init__(self):
        self.environment = AtariEnv(game="ms_pacman")
        self.inited = False

    def battle(self, ind, render=False):
        env = self.environment
        fitness = 0
        observation = env.reset()
        neurons = None
        # feed_time = 0
        # game_time = 0
        while True:
            if render:
                env.render()
            # t0 = time.time()
            neurons = ind.feed_sensor_values(observation, neurons)
            result = ind.extract_output_values(neurons)
            up = result[0] > 0.5 and result[0] > result[3]
            right = result[1] > 0.5 and result[1] > result[2]
            left = result[2] > 0.5
            down = result[3] > 0.5
            if up:
                if right:
                    action = 5
                elif left:
                    action = 6
                else:
                    action = 1
            elif down:
                if right:
                    action = 7
                elif left:
                    action = 8
                else:
                    action = 4
            elif right:
                action = 2
            elif left:
                action = 3
            else:
                action = 0
            t1 = time.time()
            observation, reward, done, info = env.step(action)
            t2 = time.time()
            # feed_time += (t1 - t0)
            # game_time += (t2 - t1)
            fitness += reward % 200
            if done:
                break
        # print(f"{feed_time}, {game_time}")
        print(fitness, file=sys.stderr)
        return fitness

    def calculateFitness(self, population, _):
        if not self.inited:
            self.inited = True
            return
        sys.stdout.flush()
        all = []
        networks = []
        for list in population:
            for ind in list["individuals"]:
                all.append(ind)
                networks.append(ind.network)
        num_threads = int(mp.cpu_count() - 1)
        pool = mp.Pool(num_threads)
        res = pool.map(self.battle, networks)
        for ind, fitness in zip(all, res):
            ind.setFitness(fitness)
        fitnesses = [(ind.getFitness(), f) for ind, f in zip(all, res)]
        average = mean(res)
        print(f"average fitness {average}")
        pool.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fg = open("sample_gene_446.json", "r")
        fm = open("metaparameters_446.json", "r")
        meta = Genes.Metaparameters.load(fm)
        fm.close()
        base = Genes.load(fg, meta)
        fg.close()
        game = Game()
        game.battle(base, True)
        game.environment.close()

    else:
        inputs = 128
        outputs = 4
        if len(sys.argv) < 2:
            base = Genes(inputs, outputs, Genes.Metaparameters(
                perturbation_chance=0.5, 
                perturbation_stdev=0.5,
                reset_weight_chance=0.05,
                new_link_weight_stdev=4,
                mutate_loop=6,
                new_node_chance=0.5,
                new_link_chance=0.5,
                c1=2.2, c2=2.2, c3=1.2,
                allow_recurrent=False
                ))
            population = [base.clone() for i in range(150)]
            for ind in population:
                for _ in range(50):
                    ind.mutate()
        else:
            population = []
            f = open("metaparameters_652.json", "r")
            metaparameters = load_metaparameters(f)
            f.close()
            f = open("sample_population_652.json", "r")
            obj = json.load(f)
            f.close()
            population = []
            for entry in obj:
                population.append(Genes.load_from_json(entry, metaparameters))
        game = Game()
        optimizer = GeneticOptimizer(population, game, 1000)
        optimizer.initialize()
        optimizer.evolve()
        best = optimizer.getBestIndividual()
        population = optimizer.getPopulation()

        f = open("atari_pacman_best.json", "w")
        best.save(f)
        f.close()
        f = open("atari_pacman_population.json", "w")
        all_inds = []
        for species in optimizer.getPopulation():
            for individual in species["individuals"]:
                all_inds.append(individual)
        f.write(json.dumps([ind.as_json() for ind in all_inds]))
        f.close()
        f = open("atari_pacman_meta.json", "w")
        best._metaparameters.save(f)
        f.close()

        game.battle(best, True)
