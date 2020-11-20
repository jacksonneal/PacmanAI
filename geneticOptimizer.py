import random
from random import randint
from captureAgents import GenesAgent
from capture import CaptureRules
from genes import Genes
from baselineTeam import OffensiveReflexAgent


class GeneticOptimizer:

    def __init__(self, population, fitnessCalculator, maxGenerations):
        """ Initialize the optimizer
        :param population - set of starting genes, all of same initial species
        :param fitnessCalculator - capable of scoring a population of individuals
        """
        self.population = [{
            "id": 0,
            "individuals": population,
            "fitness": -1,
            "stagnation": 0
        }]
        self.speciesCount = 1
        self.fitnessCalculator = fitnessCalculator
        self.generationCount = 0
        self.maxGenerations = maxGenerations
        self.best = population[randint(0, len(population) - 1)]

    def initialize(self):
        """ Prepare for evolution. """
        self._calculateFitness(self.population, self.best)

    def evolve(self):
        """ Run optimization until a termination condition is met. """
        while not self.isTerminated:
            self._evolveSingle()
            self.generationCount += 1

    def _evolveSingle(self):
        """ Execute a single evolution of genetic optimization. """
        representatives = map(
            lambda species: (species["individuals"][randint(0, len(species["individuals"]) - 1)], species), self.population)

        populationFitnessSum = sum(map(lambda species: sum(
            map(lambda ind: ind.getFitness(), species["individuals"])), self.population))

        nextGenPopulation = []
        for species in population:
            nextGenPopulation.append({
                "id": species["id"],
                "individuals": []
            })

        for species in self.population:
            speciesFitnessSum = sum(
                map(lambda ind: ind.getFitness(), species["individuals"]))
            # Constant number of offspring proportional to species fitness within larger population
            # TODO: ensure total population size stays consistent
            speciesNumOffspring = round(
                speciesFitnessSum / populationFitnessSum)
            species["individuals"].sort(key=compare)

            speciesOffspring = []
            # Eliminate worst individual
            candidates = species["individuals"][1:]
            # Autocopy best individual for large species
            if len(species["individuals"] > 5):
                speciesOffspring.append(
                    species["individuals"][len(species["individuals"]) - 1])

            # Only non-empty, non-stagnating species may evolve
            if len(candidates > 1) and species["stagnation"] < 15:
                while len(speciesOffspring) < speciesNumOffspring:
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
                            mate = candidates[randint(0, len(candidates) - 1)]
                            child = selected.clone.breed(mate.clone)
                    # TODO: properly call into mutate functions
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
                        for species in nextGenPopulation:
                            if species["id"] == rep[1]["id"]:
                                species["individuals"].append(offspring)
                        break
                if not compatible:
                    newSpecies = {
                        "id": self.speciesCount,
                        "individuals": [offspring],
                        "stagnation": 0,
                        "fitness": -1
                    }
                    self.speciesCount += 1
                    nextGenPopulation.append(newSpecies)
                    newRep = (offspring, [offspring])
                    representatives.append(offspring, newRep)

        # Filter out empty species
        nextGenPopulation = filter(lambda species: len(
            species["individuals"]), nextGenPopulation)

        # Calculate fitness in each new species
        self._calculateFitness(nextGenPopulation, self.best)

        # Update fitness and stagnation values
        for species in nextGenPopulation:
            maxFitness = max(
                map(lambda ind: ind.getFitness(), species["individuals"]))
            if maxFitness <= species["fitness"]:
                species["stagnation"]
            else:
                species["fitness"] = maxFitness
                species["stagnation"] = 0
        self.population = nextGenPopulation
        self.best = self.getBestIndividual()

    def _calculateFitness(self, population, bestInd):
        """ Calculate and cache the fitness of each individual in the population. """
        self.fitnessCalculator.calculateFitness(
            population, bestInd)

    def getPopulation(self):
        """ Access all individuals by species. """
        return self.population

    def getBestIndividual(self):
        """ Access best individual by fitness across entire population. """
        bestInd = None
        for species in self.population:
            for ind in species:
                if bestInd is None or ind.getFitness() > bestInd.getFitness():
                    bestInd = ind
        return bestInd

    def isTerminated(self):
        """ Access whether optimization has reached any termination condition. """
        return self.generationCount < self.maxGenerations


class FitnessCalculator:

    def __init__(self, layout, gameDisplay, length, muteAgents, catchExceptions):
        self.layout = layout
        self.gameDisplay = gameDisplay
        self.length = length
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self.rules = CaptureRules()

    def calculateFitness(self, population, prevBest):
        """ Calculate and cache fitness of each individual in the population.  
        Run competitive simulation to determine fitness scores. 
        Utilize fitness sharing within each species. 
        """
        # Run a game for each member of the population against the previous best member of the population
        # Cache score as fitness on individual
        for species in population:
            for individual in species["individuals"]:
                agents = [GenesAgent(0), GenesAgent(1), OffensiveReflexAgent(2), OffensiveReflexAgent(3)]
                g = self.rules.newGame(self.layout, agents, self.gameDisplay,
                                       self.length, self.muteAgents, self.catchExceptions)
                g.run()
                # TODO: terminate if running too long
                # TODO: extract score and save


class Runner:

    def __init__(self, layout, gameDisplay, length, muteAgents, catchExceptions):
        self.fitnessCalculator = FitnessCalculator(
            layout, gameDisplay, length, muteAgents, catchExceptions)
        base = Genes(16 * 32 + 8, 5, Genes.Metaparameters())
        self.optimizer = GeneticOptimizer([base], self.fitnessCalculator, 2)

    def run(self):
        self.optimizer.initialize()
        self.optimizer.evolve()
        exit(0)


def compare(ind0, ind1):
    """ Comparator used to sort individuals by fitness values. """
    return ind0.getFitness() - ind1.getFitness()
