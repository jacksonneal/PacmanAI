import numpy as np
from numpy.core.numeric import Inf
import gym

from genes import *
from geneticOptimizer import *


env = gym.make('MountainCar-v0')
env._max_episode_steps = float("inf")
min_pos = -1.2
max_pos = 0.6
diff = max_pos - min_pos

def run_cart_pole(ind, render, time_limit=200):
    fitness = 0
    observation = env.reset()
    action_space = env.action_space
    neurons = None
    max_reached = observation[0]
    for t in range(time_limit):
        if render:
            env.render()
        neurons = ind.feed_sensor_values(observation, neurons)
        result = ind.extract_output_values(neurons)
        action = np.argmax(result)
        observation, reward, done, info = env.step(action)
        position = observation[0]
        max_reached = max(max_reached, observation[0])
        fitness += reward
        if done:
            break
    return fitness + (max_reached - min_pos) / diff


class MountainCarFitness:
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


base = Genes(2, 3, Genes.Metaparameters(perturbation_chance=0.5, perturbation_stdev=0.5, new_link_weight_stdev=4, c1=2, c2=2, c3=0.4))
population = [base.clone() for i in range(150)]
for ind in population:
    for output_node_index in range(3):
        ind.add_connection(ind.input_node_index(0), ind.output_node_index(output_node_index))
        ind.add_connection(ind.input_node_index(1), ind.output_node_index(output_node_index))
        ind.add_connection(Genes.BIAS_INDEX, ind.output_node_index(output_node_index))
optimizer = GeneticOptimizer(population, MountainCarFitness(), 8)
optimizer.initialize()
optimizer.evolve()
best = optimizer.getBestIndividual()
population = optimizer.getPopulation()
fitness = run_cart_pole(best, True, 99999999999999999999999)

env.close()
