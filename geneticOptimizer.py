import random
from random import randint


class GeneticOptimizer:

    def __init__(self, population, fitnessCalculator):
        """ Initialize the optimizer
        :param population - set of starting genes, all of same initial species
        :param fitnessCalculator - capable of scoring a population of individuals
        """
        self.population = [population]
        self.fitnessCalculator = fitnessCalculator
        self.stagnations = {}
        self.fitness = {}
        self.generationCount = 0
        self.maxGenerations = 10

    def initialize(self, population):
        """ Prepare for evolution. """
        for species in population:
            self.stagnations[species] = 0
            self.fitness[species] = -1
        self._calculateFitness()

    def evolve(self):
        """ Run optimization until a termination condition is met. """
        while not self.isTerminated:
            self._evolveSingle()
            self.generationCount += 1

    def _evolveSingle(self):
        """ Execute a single evolution of genetic optimization. """
        representatives = map(
            lambda x: (x[randint(0, len(x) - 1)], x), self.population)
        for rep in representatives:
            rep[1].clear()

        populationFitnessSum = sum(map(lambda species: sum(
            map(lambda ind: ind.getFitness(), species)), self.population))

        for species in population:
            speciesFitnessSum = sum(map(lambda ind: ind.getFitness(), species))
            # Constant number of offspring proportional to species fitness within larger population
            # TODO: ensure population size stays consistent
            speciesNumOffspring = round(
                speciesFitnessSum / populationFitnessSum)
            species.sort(key=compare)

            speciesOffspring = []
            # Eliminate worst individual
            candidates = species[1:]
            # Autocopy best individual for large species
            if len(species > 5):
                speciesOffspring.append(species[len(species) - 1])

            # Only non-empty, non-stagnating species may evolve
            if len(candidates > 1) and self.stagnations[species] < 15:
                while len(speciesOffspring) < speciesNumOffspring:
                    # TODO: what is selection strategy within species?
                    selected = candidates[randint(0, len(candidates) - 1)]
                    child = selected
                    crossRand = random.uniform(0, 1)
                    connMutateRand = random.uniform(0, 1)
                    nodeMutateRand = random.uniform(0, 1)
                    if crossRand < .75:
                        if crossRand < .001:
                            # TODO: perform interspecies crossover. what is selection strategy for mate?
                            pass
                        else:
                            # TODO: what is selection strategy for mate?
                            mate = candidates[randint(0, len(candidates) - 1)]
                            child = selected.clone.breed(mate.clone)
                    if connMutateRand < 80:
                        connTypeRand = random.uniform(0, 1)
                        if connTypeRand < .9:
                            child = selected.mutateConnUniformPertubation()
                        else:
                            child = selected.mutateConnRandValue()
                speciesOffspring.append(child)

            for offspring in speciesOffspring:
                compatible = False
                for rep in representatives:
                    # TODO: where is compatibility configuration set? Probably should be in optimizer and passed to function here
                    if offspring.compatible(rep[0]):
                        compatible = True
                        rep[1].append(offspring)
                        break
                if not compatible:
                    newRep = (offspring, [offspring])
                    representatives.append(offspring, newRep)
                    stagnations[newRep[1]] = 0

        nextGenPopulation = []
        # Representative species that are non-empty have received an offspring and may continue to exist
        for rep in representatives:
            if len(rep[1]):
                nextGenPopulation.append(rep[1])

        self._calculateFitness(nextGenPopulation)
        for species in nextGenPopulation:
            maxFitness = max(map(lambda ind: ind.getFitness(), species))
            if self.fitess.has_key(species) and maxFitness <= self.fitness[species]:
                self.stagnations[species] += 1
            else:
                self.fitness[species] = maxFitness
                self.stagnations = 0
        self.population = nextGenPopulation

    def _calculateFitness(self, population):
        """ Calculate and cache the fitness of each individual in the population. """
        fitnessCalculator.calculateFitness(population)

    def getPopulation(self):
        """ Access all individuals by species. """
        return self.population

    def isTerminated(self):
        """ Access whether optimization has reached any termination condition. """
        return self.generationCount < self.maxGenerations


class FitnessCalculator:

    def calculateFitness(self, population):
        """ Calculate and cache fitness of each individual in the population.  
        Run competitive simulation to determine fitness scores. 
        Utilize fitness sharing within each species. 
        """
        # TODO: calculate species sharing fitness
        pass


def compare(ind0, ind1):
    """ Comparator used to sort individuals by fitness values. """
    return ind0.getFitness() - ind1.getFitness()
