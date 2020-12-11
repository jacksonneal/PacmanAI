import numpy as np
from numpy.core.numeric import Inf
import gym

from genes import *
from geneticOptimizer import *

import multiprocessing as mp

min_pos = -1.2
max_pos = 0.6
diff = max_pos - min_pos


def run_mountain_car(env, ind, render, time_limit=200):
    fitness = 0
    observation = env.reset()
    action_space = env.action_space
    neurons = None
    max_reached = observation[0]
    j = ind.as_json()
    for t in range(time_limit):
        if render:
            env.render()
        neurons = ind.feed_sensor_values(observation, neurons)
        result = ind.extract_output_values(neurons)
        action = np.argmax(result)
        observation, reward, done, info = env.step(action)
        max_reached = max(max_reached, observation[0])
        fitness += reward
        if done:
            break
    return fitness + (max_reached - min_pos) / diff + time_limit


class MountainCarFitness:
    def __init__(self):
        self.environment = gym.make('MountainCar-v0')
        self.environment._max_episode_steps = float("inf")

    def battle(self, ind):
        fitness = run_mountain_car(self.environment, ind, False)
        return fitness

    def calculateFitness(self, population, _):
        networks = []
        all = []
        for list in population:
            for ind in list["individuals"]:
                all.append(ind)
                networks.append(ind.network)
        num_threads = int(mp.cpu_count() - 1)
        pool = mp.Pool(num_threads)
        res = pool.map(self.battle, networks)
        for ind, fitness in zip(all, res):
            ind.setFitness(fitness)
        print(f"average fitness = {sum(res) / len(res)}")
        pool.close()


if __name__ == "__main__":
    base = Genes(2, 3, Genes.Metaparameters(perturbation_chance=0.5, perturbation_stdev=0.5, new_link_weight_stdev=4, c1=2, c2=2, c3=0.8))
    population = [base.clone() for i in range(150)]
    # for ind in population:
    #    for output_node_index in range(3):
    #        ind.add_connection(ind.input_node_index(0), ind.output_node_index(output_node_index))
    #        ind.add_connection(ind.input_node_index(1), ind.output_node_index(output_node_index))
    #        ind.add_connection(Genes.BIAS_INDEX, ind.output_node_index(output_node_index))
    fitness = MountainCarFitness()
    optimizer = GeneticOptimizer(population, fitness, 100, 100)
    optimizer.initialize()
    optimizer.evolve()
    best = optimizer.getBestIndividual()
    best_json = best.as_json()
    population = optimizer.getPopulation()
    fitness = run_mountain_car(fitness.environment, best, True, 99999999999999999999999)
