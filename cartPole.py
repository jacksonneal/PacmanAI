import numpy as np
from numpy.core.numeric import Inf
import gym

from genes import *
from geneticOptimizer import *


env = gym.make('CartPole-v0')
env._max_episode_steps = float("inf")

def run_cart_pole(ind, render, time_limit=500):
    observation = env.reset()
    action_space = env.action_space
    neurons = None
    t = 0
    total_deviation = 0.0
    while t < time_limit:
        if render:
            env.render()
        neurons = ind.feed_sensor_values(observation, neurons)
        result = ind.extract_output_values(neurons)
        action = 0 if result[0] > result[1] else 1
        observation, reward, done, info = env.step(action)
        total_deviation += abs(observation[0]) + abs(observation[2])
        t += 1
        if done:
            break
    return -total_deviation / t + t

class CartPoleFitness:
    def calculateFitness(self, population, _):
        count = 0
        total_fitness = 0
        for list in population:
            for ind in list["individuals"]:
                fitness = run_cart_pole(ind, False)
                ind.setFitness(fitness)
                total_fitness += fitness
                count += 1
        print(f'average fitness = {total_fitness / count}')

base = Genes(4, 2, Genes.Metaparameters(perturbation_chance=0.5, perturbation_stdev=0.5, new_link_weight_stdev=4, c1=2, c2=2, c3=0.4))
population = [base.clone() for i in range(150)]
optimizer = GeneticOptimizer(population, CartPoleFitness(), 20)
optimizer.initialize()
optimizer.evolve()
best = optimizer.getBestIndividual()
population = optimizer.getPopulation()
fitness = run_cart_pole(best, True, 99999999999999999999999)

env.close()