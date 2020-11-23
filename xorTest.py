from genes import *
from geneticOptimizer import *

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]

class XorFitness:
    def calculateFitness(self, population, _):
        for list in population:
            for ind in list["individuals"]:
                error = 0
                for input, output in zip(inputs, outputs):
                    neurons = ind.feed_sensor_values(input)
                    result = ind.extract_output_values(neurons)[0]
                    diff = result - output
                    error += diff * diff
                ind.setFitness(-error)            

base = Genes(2, 1, Genes.Metaparameters(perturbation_chance=0.5, perturbation_stdev=0.5, new_link_weight_stdev=4, c1=2, c2=2, c3=0.4))
population = [base.clone() for i in range(150)]
for ind in population:
    ind.add_connection(ind.input_node_index(0), ind.output_node_index(0))
    ind.add_connection(ind.input_node_index(1), ind.output_node_index(0))
    ind.add_connection(Genes.BIAS_INDEX, ind.output_node_index(0))
optimizer = GeneticOptimizer(population, XorFitness(), 200)
optimizer.initialize()
optimizer.evolve()
best = optimizer.getBestIndividual()
population = optimizer.getPopulation()
for input, output in zip(inputs, outputs):
    neurons = best.feed_sensor_values(input)
    result = best.extract_output_values(neurons)[0]
    print(f"{input[0]} xor {input[1]} = {result}")
pass
