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
        self.skip_first = False

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
            # neurons = ind.feed_sensor_values(observation, neurons)
            # result = ind.extract_output_values(neurons)
            result = [np.random.normal() for _ in range(5)]
            up = result[0] > 0.5 and result[0] > result[3]
            right = result[1] > 0.5 and result[1] > result[2]
            left = result[2] > 0.5
            down = result[3] > 0.5
            if up:
                if right:
                    action = 6
                elif left:
                    action = 7
                else:
                    action = 2
            elif down:
                if right:
                    action = 8
                elif left:
                    action = 9
                else:
                    action = 5
            elif right:
                action = 3
            elif left:
                action = 4
            else:
                action = 0
            fire = result[4] > 0.5
            if fire:
                if action == 0:
                    action = 1
                else:
                    action += 8
            observation, reward, done, info = env.step(action)
            fitness += reward
            if done or fitness <= -10:
                break
        # print(f"{feed_time}, {game_time}")
        print(fitness, file=sys.stderr)
        return fitness + 20
    
    def calculateFitnessHelp(self, all):
        networks = [ind.network for ind in all]
        num_threads = int(mp.cpu_count() - 1)
        pool = mp.Pool(num_threads)
        res = pool.map(self.battle, networks)
        for ind, fitness in zip(all, res):
            ind.setFitness(fitness)
        pool.close()
        #print(res, file=sys.stderr)
        print(np.mean(res))

    def calculateFitness(self, population, _):
        if self.skip_first:
            self.skip_first = False
            return
        all = []
        for list in population:
            for ind in list["individuals"]:
                all.append(ind)
        self.calculateFitnessHelp(all)
        


if __name__ == "__main__":
    game = Game()
    game.battle(None, True)
    game.environment.close()
    pass
    if len(sys.argv) > 1:
        fm = open("boxing2/metaparameters_300.json", "r")
        meta = load_metaparameters(fm)
        fm.close()
        fg = open("boxing2/sample_gene_300.json", "r")
        best = Genes.load(fg, meta)
        fg.close()
        game = Game()
        # game.calculateFitnessHelp(population)
        # best = max(population, key=lambda ind: ind.getFitness())
        # fout = open("atari_best_194.json", "w")
        # best.save(fout)
        # fout.close()
        game.battle(best.network, True)
        game.environment.close()

    else:
        inputs = 128
        outputs = 5
        base = Genes(inputs, outputs, Genes.Metaparameters(
            perturbation_chance=0.5, 
            perturbation_stdev=0.5, 
            new_link_weight_stdev=4, 
            new_node_chance=0.15, 
            c1=1.7, c2=1.7, c3=1.2, 
            allow_recurrent=False))
        population = [base.clone() for i in range(150)]
        for ind in population:
            for _ in range(50):
                ind.mutate()
            #for output_node_index in range(outputs):
            #    for input_node_index in range(inputs):
            #        ind.add_connection(ind.input_node_index(input_node_index), ind.output_node_index(output_node_index))
            #    ind.add_connection(Genes.BIAS_INDEX, ind.output_node_index(output_node_index))
        game = Game()
        optimizer = GeneticOptimizer(population, game, 300)
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
