import random
from random import randint
from captureAgents import GenesAgent
from capture import CaptureRules
from genes import Genes
from baselineTeam import OffensiveReflexAgent, DefensiveReflexAgent
import math as m
import multiprocessing as mp
import json


class GeneticOptimizer:

    def __init__(self, population, fitnessCalculator, maxGenerations):
        """ Initialize the optimizer
        :param population - set of starting genes, all of same initial species
        :param fitnessCalculator - capable of scoring a population of individuals
        """
        self.population = [{
            "id": 0,
            "individuals": population,
            "fitness": None,
            "stagnation": 0
        }]
        self.speciesCount = 1
        self.fitnessCalculator = fitnessCalculator
        self.generationCount = 0
        self.populationSize = len(population)
        self.maxGenerations = maxGenerations
        self.best = population[randint(0, len(population) - 1)]
        self.stagnated = False

    def initialize(self):
        """ Prepare for evolution. """
        self._calculateFitness(self.population, self.best)
        self.population[0]["fitness"] = max(list(map(lambda ind: ind.getFitness(), self.population[0]["individuals"])))

    def evolve(self):
        """ Run optimization until a termination condition is met. """

        while not self.isTerminated():
            self._evolveSingle()
            self._endOfEpoch()
            self.generationCount += 1

    def _evolveSingle(self):
        """ Execute a single evolution of genetic optimization. """
        representatives = list(map(
            lambda species: (species["individuals"][randint(0, len(species["individuals"]) - 1)], species["id"]), self.population))


        populationFitnessSum = sum(list(map(lambda species: sum(
            list(map(lambda ind: ind.getFitness(), species["individuals"]))), self.population)))

        nextGenPopulation = []
        for species in self.population:
            nextGenPopulation.append({
                "id": species["id"],
                "individuals": [],
                "fitness": species["fitness"],
                "stagnation": species["stagnation"]
            })

        allOffspring = []
        for species in self.population:
            speciesOffspring = []
            speciesFitnessSum = sum(
                list(map(lambda ind: ind.getFitness(), species["individuals"])))
            # Constant number of offspring proportional to species fitness within larger population
            speciesNumOffspring = 0
            if populationFitnessSum == 0:
                speciesNumOffspring = len(species["individuals"])
            else:
                speciesNumOffspring = m.ceil(
                    speciesFitnessSum / populationFitnessSum * self.populationSize)
            species["individuals"].sort(key=lambda ind: ind.getFitness())

            # Eliminate worst individual
            # TODO: eliminate worst individual from population, not from each species
            # candidates = species["individuals"][1:]
            candidates = species["individuals"]
            # Autocopy best individual for large species
            if len(species["individuals"]) > 5:
                speciesOffspring.append(
                    species["individuals"][len(species["individuals"]) - 1].clone())

            # Only non-empty, non-stagnating species may evolve
            if len(candidates) >= 1 and species["stagnation"] < 50:
                while len(speciesOffspring) < speciesNumOffspring:
                    selected = candidates[randint(0, len(candidates) - 1)]
                    child = selected.clone()
                    crossRand = random.uniform(0, 1)
                    connMutateRand = random.uniform(0, 1)
                    if crossRand < .75:
                        if crossRand < .001:
                            randSpecies = self.population[randint(0, len(self.population) - 1)]
                            randMate = randSpecies["individuals"][randint(0, len(randSpecies["individuals"]) - 1)]
                            child = child.breed(randMate.clone(), (child.getFitness() > randMate.getFitness()))
                        else:
                            mate = candidates[randint(0, len(candidates) - 1)]
                            child = child.breed(mate.clone(), (child.getFitness() > mate.getFitness()))
                    if connMutateRand < .80:
                        child = child.mutate()
                    speciesOffspring.append(child)
            allOffspring.extend(speciesOffspring)

        # If a species stagnates, it won't reproduce.  Fill its space with random sample across all species.
        while len(allOffspring) < self.populationSize:
            randSpecies = self.population[randint(0, len(self.population) - 1)]
            allOffspring.append(randSpecies["individuals"][randint(0, len(randSpecies["individuals"]) - 1)])

        for offspring in allOffspring:
            compatible = False
            for rep in representatives:
                if compatible == False and offspring.distance(rep[0]) < 3.0:
                    compatible = True
                    for species in nextGenPopulation:
                        if species["id"] == rep[1]:
                            species["individuals"].append(offspring)
            if compatible == False:
                newSpecies = {
                    "id": self.speciesCount,
                    "individuals": [offspring],
                    "stagnation": 0,
                    "fitness": None
                }
                self.speciesCount += 1
                nextGenPopulation.append(newSpecies)
                newRep = (offspring, newSpecies["id"])
                representatives.append(newRep)

        # Filter out empty species
        nextGenPopulation = list(filter(lambda species: len(
            species["individuals"]), nextGenPopulation))

        # Calculate fitness in each new species
        self._calculateFitness(nextGenPopulation, self.best)            

        # Update fitness and stagnation values
        for species in nextGenPopulation:
            maxFitness = max(
                list(map(lambda ind: ind.getFitness(), species["individuals"])))
            if species["fitness"] is not None and maxFitness <= species["fitness"]:
                species["stagnation"] += 1
            else:
                species["fitness"] = maxFitness
                species["stagnation"] = 0
        if sum(list(map(lambda species: len(species["individuals"]), nextGenPopulation))) >= self.populationSize:
            self.population = nextGenPopulation
            self.best = self.getBestIndividual()
        else:
            self.stagnated = True

    def _calculateFitness(self, population, bestInd):
        """ Calculate and cache the fitness of each individual in the population. """
        self.fitnessCalculator.calculateFitness(
            population, bestInd)

    def _endOfEpoch(self):
        print("BEST_FITNESS: ", self.best.getFitness(), " GEN_COUNT: ", self.generationCount, " SPECIES_SIZE: ", 
            [len(species["individuals"]) for species in self.population], " POP_SIZE: ", 
            sum(list(map(lambda species: len(species["individuals"]), self.population))))
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
        return score


class Runner:

    def __init__(self, layout, gameDisplay, length, muteAgents, catchExceptions):
        maxGen = 1000
        populationSize = 150
        self.load = True
        self.save = True
        self.fitnessCalculator = FitnessCalculator(
            layout, gameDisplay, length, muteAgents, catchExceptions)
        base = []
        self.baseUnit = Genes(16 * 32 + 8, 5, Genes.Metaparameters())
        if self.load:
            f = open("sample_population.json", "r")
            asJson = json.load(f)
            for ind in asJson:
                if len(base) < populationSize:
                    base.append(Genes.load_from_json(ind, self.baseUnit._metaparameters))
            # self.baseUnit = Genes.load(open("sample_gene.json", "r"), self.baseUnit._metaparameters)
        while len(base) < populationSize:
            base.append(self.baseUnit.clone())
        self.optimizer = GeneticOptimizer(base, self.fitnessCalculator, maxGen)

    def run(self):
        self.optimizer.initialize()
        self.optimizer.evolve()
        if self.save:
            all_inds = []
            for species in self.optimizer.getPopulation():
                for individual in species["individuals"]:
                    all_inds.append(individual)
            f = open("sample_population.json", "w")
            f.write(json.dumps(list(map(lambda ind: ind.as_json(), all_inds))))
            f.flush()
            self.optimizer.getBestIndividual().save(open("sample_gene.json", "w"))
        exit(0)
