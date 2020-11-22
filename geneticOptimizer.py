import random
from random import randint
from captureAgents import GenesAgent
from capture import CaptureRules
from genes import Genes
from baselineTeam import OffensiveReflexAgent, DefensiveReflexAgent
import math as m
import multiprocessing as mp


class GeneticOptimizer:

    def __init__(self, population, fitnessCalculator, maxGenerations, populationSize):
        """ Initialize the optimizer
        :param population - set of starting genes, all of same initial species
        :param fitnessCalculator - capable of scoring a population of individuals
        """
        self.population = [{
            "id": 0,
            "individuals": population,
            "fitness": 0,
            "stagnation": 0
        }]
        self.speciesCount = 1
        self.fitnessCalculator = fitnessCalculator
        self.generationCount = 0
        self.populationSize = populationSize
        self.maxGenerations = maxGenerations
        self.best = population[randint(0, len(population) - 1)]
        self.stagnated = False

    def initialize(self):
        """ Prepare for evolution. """
        self._calculateFitness(self.population, self.best)

    def evolve(self):
        """ Run optimization until a termination condition is met. """

        while not self.isTerminated():
            self._evolveSingle()
            self._endOfEpoch()
            self.generationCount += 1

    def _evolveSingle(self):
        """ Execute a single evolution of genetic optimization. """
        representatives = list(map(
            lambda species: (species["individuals"][randint(0, len(species["individuals"]) - 1)], species), self.population))

        populationFitnessSum = sum(map(lambda species: sum(
            map(lambda ind: ind.getFitness(), species["individuals"])), self.population))

        nextGenPopulation = []
        for species in self.population:
            nextGenPopulation.append({
                "id": species["id"],
                "individuals": [],
                "fitness": species["fitness"],
                "stagnation": species["stagnation"]
            })

        for species in self.population:
            speciesFitnessSum = sum(
                map(lambda ind: ind.getFitness(), species["individuals"]))
            # Constant number of offspring proportional to species fitness within larger population
            speciesNumOffspring = 0
            if populationFitnessSum == 0:
                speciesNumOffspring = len(species["individuals"])
            else:
                speciesNumOffspring = m.ceil(
                    speciesFitnessSum / populationFitnessSum) * self.populationSize
            species["individuals"].sort(key=lambda ind: ind.getFitness())

            speciesOffspring = []
            # Eliminate worst individual
            candidates = species["individuals"][1:]
            # Autocopy best individual for large species
            if len(species["individuals"]) > 5:
                speciesOffspring.append(
                    species["individuals"][len(species["individuals"]) - 1])

            # Only non-empty, non-stagnating species may evolve
            if len(candidates) >= 1 and species["stagnation"] < 15:
                while len(speciesOffspring) < speciesNumOffspring:
                    selected = candidates[randint(0, len(candidates) - 1)]
                    child = selected
                    crossRand = random.uniform(0, 1)
                    connMutateRand = random.uniform(0, 1)
                    if crossRand < .75:
                        if crossRand < .001:
                            randSpecies = self.population[randint(0, len(self.population) - 1)]
                            randMate = randSpecies["individuals"][randint(0, len(randSpecies["individuals"]) - 1)]
                            child = selected.clone().breed(randMate.clone(), (selected.getFitness() > randMate.getFitness()))
                        else:
                            mate = candidates[randint(0, len(candidates) - 1)]
                            child = selected.clone().breed(mate.clone(), (selected.getFitness() > mate.getFitness()))
                    if connMutateRand < .80:
                        child = selected.clone().mutate()
                    speciesOffspring.append(child)

            for offspring in speciesOffspring:
                compatible = False
                for rep in representatives:
                    if offspring.distance(rep[0]) < 3.0:
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
                        "fitness": 0
                    }
                    self.speciesCount += 1
                    nextGenPopulation.append(newSpecies)
                    newRep = (offspring, [offspring])
                    representatives.append((offspring, newRep))

        # Filter out empty species
        nextGenPopulation = list(filter(lambda species: len(
            species["individuals"]), nextGenPopulation))

        # Calculate fitness in each new species
        self._calculateFitness(nextGenPopulation, self.best)

        # Update fitness and stagnation values
        for species in nextGenPopulation:
            maxFitness = max(
                map(lambda ind: ind.getFitness(), species["individuals"]))
            if maxFitness <= species["fitness"]:
                species["stagnation"] += 1
            else:
                species["fitness"] = maxFitness
                species["stagnation"] = 0
        if len(nextGenPopulation):
            self.population = nextGenPopulation
            self.best = self.getBestIndividual()
        else: 
            self.stagnated = True

    def _calculateFitness(self, population, bestInd):
        """ Calculate and cache the fitness of each individual in the population. """
        self.fitnessCalculator.calculateFitness(
            population, bestInd)

    def _endOfEpoch(self):
        print("BEST_FITNESS: ", self.best.getFitness(), " SPECIES_SIZE: ", 
            [len(species["individuals"]) for species in self.population], " POP_SIZE: ", len(self.population))
        self.best._metaparameters.reset_tracking()

    def getPopulation(self):
        """ Access all individuals by species. """
        return self.population

    def getBestIndividual(self):
        """ Access best individual by fitness across entire population. """
        bestInd = None
        for species in self.population:
            for ind in species["individuals"]:
                if bestInd is None or ind.getFitness() > bestInd.getFitness():
                    bestInd = ind
        return bestInd

    def isTerminated(self):
        """ Access whether optimization has reached any termination condition. """
        return self.generationCount >= self.maxGenerations or self.stagnated


class FitnessCalculator:

    def __init__(self, layout, gameDisplay, length, muteAgents, catchExceptions):
        self.layout = layout
        self.gameDisplay = gameDisplay
        self.length = length
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self.rules = CaptureRules()
        self.prevBest = None
        self.isRunParallel = True

    def calculateFitness(self, population, prevBest):
        """ Calculate and cache fitness of each individual in the population.  
        Run competitive simulation to determine fitness scores. 
        Utilize fitness sharing within each species. 
        """
        # Run a game for each member of the population against the previous best member of the population
        # Cache score as fitness on individual
        self.prevBest = prevBest
        all_inds = []
        for species in population:
            for individual in species["individuals"]:
                all_inds.append(individual)
        if self.isRunParallel:
            pool = mp.Pool(int(mp.cpu_count() / 2))
            res = pool.map(self.battle, all_inds)
            pool.close()
            i = 0
            while i < len(res):
                all_inds[i].setFitness(res[i])
                i += 1
        else:
            for ind in all_inds:
                ind.setFitness(self.battle(ind))
    
    def battle(self, individual):
        agents = [GenesAgent(0, individual), GenesAgent(1, self.prevBest), DefensiveReflexAgent(2), DefensiveReflexAgent(3)]
        g = self.rules.newGame(self.layout, agents, self.gameDisplay,
                                self.length, self.muteAgents, self.catchExceptions)
        g.run()
        score = g.state.getScore()
        if score == 0:
            score = agents[0].maxPathDist / 10000
        else:
            score += 1


class Runner:

    def __init__(self, layout, gameDisplay, length, muteAgents, catchExceptions):
        self.load = True
        self.save = True
        self.fitnessCalculator = FitnessCalculator(
            layout, gameDisplay, length, muteAgents, catchExceptions)
        base = []
        self.baseUnit = Genes(16 * 32 + 8, 5, Genes.Metaparameters())
        if self.load:
            self.baseUnit = self.baseUnit.load(open("sample_gene.json", "r"), self.baseUnit._metaparameters)
        maxGen = 10
        populationSize = 100
        for i in range(populationSize):
            base.append(self.baseUnit.clone())
        self.optimizer = GeneticOptimizer(base, self.fitnessCalculator, maxGen, populationSize)

    def run(self):
        self.optimizer.initialize()
        self.optimizer.evolve()
        if self.save:
            self.optimizer.getBestIndividual().save(open("sample_gene.json", "w"))
        exit(0)
