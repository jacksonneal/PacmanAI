from scipy.special import expit as sigmoid
import numpy as np
import util
import copy
import json

class RandomlyTrue:
    def __bool__(self):
        return util.flipCoin(0.5)
    instance = None


RandomlyTrue.instance = RandomlyTrue()


class Genes:

    class Metaparameters:
        def __init__(self,
                     c1=1, c2=1, c3=1,
                     perturbation_chance=0.1,
                     perturbation_stdev=0.1,
                     new_link_chance=0.1,
                     bias_link_chance=0.1,
                     new_link_weight_stdev=1,
                     new_node_chance=0.1,
                     disable_mutation_chance=0.1,
                     enable_mutation_chance=0.1):
            self.innovation_number = 0
            self.c1 = c1
            self.c2 = c2
            self.c3 = c3
            self.new_link_chance = new_link_chance
            self.bias_link_chance = bias_link_chance
            self.new_link_weight_stdev = new_link_weight_stdev
            self.new_node_chance = new_node_chance
            self.perturbation_chance = perturbation_chance
            self.perturbation_stdev = perturbation_stdev
            self.disable_mutation_chance = disable_mutation_chance
            self.enable_mutation_chance = enable_mutation_chance

        def increment_innovation(self):
            self.innovation_number += 1
            return self.innovation_number

    _IN_NODE = 0
    _OUT_NODE = 1
    _WEIGHT = 2
    _ENABLED = 3
    _INNOV_NUMBER = 4

    def __init__(self, num_sensors_or_copy, num_outputs=None, metaparameters=None):
        if isinstance(num_sensors_or_copy, Genes):
            to_copy = num_sensors_or_copy
            self._num_sensors = to_copy._num_sensors
            self._num_outputs = to_copy._num_outputs
            self._dynamic_nodes = copy.deepcopy(to_copy._dynamic_nodes)
            self._connections = copy.deepcopy(to_copy._connections)
            self._metaparameters = to_copy._metaparameters
        else:
            self._num_sensors = num_sensors_or_copy
            self._num_outputs = num_outputs
            self._dynamic_nodes = []
            for _ in range(num_outputs):
                self._dynamic_nodes.append([])
            self._connections = []
            self._metaparameters = metaparameters
        self.fitness = 0

    def feed_sensor_values(self, values, neurons=None):
        """ Run the network with the given input through the given neurons (creates them if not given), returns neuron values """
        if neurons is None:
            neurons = [0] * (len(self._dynamic_nodes) + self._num_sensors + 1)
        assert len(values) == self._num_sensors, "invalid number of inputs"
        neurons[0] = 1  # BIAS node
        for i in range(self._num_sensors):
            neurons[i + 1] = values[i]
        for node_index, node in enumerate(self._dynamic_nodes):
            neuron_index = node_index + self._num_sensors + 1
            has_connections = False
            sum = 0

            for connection_index in node:
                in_node, out_node, weight, enabled, innov = self._connections[connection_index]
                # assert out_node = neuron_index
                if enabled:
                    has_connections = True
                    sum += neurons[in_node] * weight
            if has_connections:
                neurons[neuron_index] = sigmoid(sum)
        return neurons

    def extract_output_values(self, neuron_values):
        """ Extracts the output values from the result of feed_sensor_values """
        return neuron_values[self._num_sensors + 1:self._num_sensors + self._num_outputs + 1]

    def _node_by_index(self, index):
        return self._dynamic_nodes[index - self._num_sensors - 1]

    def _total_nodes(self):
        return 1 + self._num_sensors + len(self._dynamic_nodes)

    def _add_connection(self):
        total_nodes = self._total_nodes()
        input_index = 0 if util.flipCoin(self._metaparameters.bias_link_chance) else util.random.randint(1, total_nodes - 1)
        output_index = util.random.randint(self._num_sensors, total_nodes - 1)

        incoming = self._node_by_index(output_index)
        for connection_index in incoming:
            connection = self._connections[connection_index]
            if connection[Genes._IN_NODE] == input_index:
                return
        innovation_number = self._metaparameters.increment_innovation()
        connection = [input_index, output_index, np.random.normal(0, self._metaparameters.new_link_weight_stdev), 1, innovation_number]
        incoming.append(len(self._connections))
        self._connections.append(connection)
        pass

    def _add_node(self):
        if len(self._connections) == 0:
            return
        connection = util.random.choice(self._connections)
        connection[Genes._ENABLED] = False
        in_node, out_node, _a, _b, _c  = connection
        new_node = []
        self._dynamic_nodes.append(new_node)
        leading = [in_node, self._total_nodes() - 1, 1, True, self._metaparameters.increment_innovation()]
        trailing = [self._total_nodes() - 1, out_node, connection[Genes._WEIGHT], True, self._metaparameters.increment_innovation()]
        self._connections.append(leading)
        self._connections.append(trailing)
        new_node.append(len(self._connections) - 2)
        self._node_by_index(out_node).append(len(self._connections) - 1)

    def _perturb(self):
        for connection in self._connections:
            connection[Genes._WEIGHT] += np.random.normal(0, self._metaparameters.perturbation_stdev)

    def _enable_mutation(self, enable):
        if len(self._connections) == 0:
            return
        connection = util.random.choice(self._connections)
        connection[Genes._ENABLED] = enable

    def mutate(self):
        """ Mutate the genes in this genome, returns self """
        if util.flipCoin(self._metaparameters.new_link_chance):
            self._add_connection()
        if util.flipCoin(self._metaparameters.new_node_chance):
            self._add_node()
        if util.flipCoin(self._metaparameters.perturbation_chance):
            self._perturb()
        if util.flipCoin(self._metaparameters.disable_mutation_chance):
            self._enable_mutation(False)
        if util.flipCoin(self._metaparameters.enable_mutation_chance):
            self._enable_mutation(True)
        return self

    def breed(self, other, self_more_fit=RandomlyTrue.instance):
        """ Creates a child from the result of breeding self and other genes, returns new child """
        ret = Genes(self._num_sensors, self._num_outputs, self._metaparameters)
        slen = len(self._connections)
        olen = len(other._connections)
        i = 0
        j = 0
        while i < slen and j < olen:
            sc = self._connections[i]
            so = other._connections[j]
            sci = sc[Genes._INNOV_NUMBER]
            soi = so[Genes._INNOV_NUMBER]
            if sci == soi:
                ret._connections.append(copy.deepcopy(sc if util.flipCoin(0.5) else so))
                i += 1
                j += 1
            elif sci < soi:
                i += i
                if self_more_fit:
                    ret._connections.append(copy.deepcopy(sc))
            else:
                j += 1
                if not self_more_fit:
                    ret._connections.append(copy.deepcopy(soi))
        while i < slen:
            if self_more_fit:
                ret._connections.append(self._connections[i])
            i += 1
        while j < olen:
            if not self_more_fit:
                ret._connections.append(other._connections[j])
            j += 1
        max_node = 0
        for connection in ret._connections:
            max_node = max(max_node, connection[Genes._IN_NODE], connection[Genes._OUT_NODE])
        i = ret._num_sensors + ret._num_outputs + 1
        while i < max_node:
            ret._dynamic_nodes.append([])
        for index, connection in enumerate(ret._connections):
            ret._node_by_index(connection[Genes._OUT_NODE]).append(index)
        return ret

    def distance(self, other):
        assert self._metaparameters is other._metaparameters
        c1 = self._metaparameters.c1
        c2 = self._metaparameters.c2
        c3 = self._metaparameters.c3
        slen = len(self._connections)
        olen = len(other._connections)
        n = max(olen, slen)
        if n == 0:
            return 0
        disjoint = 0
        weight_difference = 0
        shared = 0
        i = 0
        j = 0
        while i < slen and j < olen:
            sc = self._connections[i]
            so = other._connections[j]
            sci = sc[Genes._INNOV_NUMBER]
            soi = so[Genes._INNOV_NUMBER]
            if sci == soi:
                weight_difference += abs(sc[Genes._WEIGHT] - so[Genes._WEIGHT])
                shared += 1
                i += 1
                j += 1
            else:
                disjoint += 1
                if sci < soi:
                    i += 1
                else:
                    j += 1
        excess = olen - j + slen - i
        return (c1 * excess / n) + (c2 * disjoint / n) + (0 if shared == 0 else c3 * weight_difference / shared)

    def clone(self):
        return Genes(self)

    def save(self, out_stream):
        asJson = {"nodeCount": self._total_nodes(), "inputCount": self._num_sensors, "outputCount": self._num_outputs, "connections": self._connections}
        out_stream.write(json.dumps(asJson))
        out_stream.flush()

    def load(in_stream, metaparameters):
        asJson = json.load(in_stream)
        ret = Genes(asJson["inputCount"], asJson["outputCount"], metaparameters)
        toAdd = asJson["nodeCount"] - ret._total_nodes()
        for _ in range(toAdd):
            ret._dynamic_nodes.append([])
        connections = asJson["connections"]
        count = 0
        for in_node, out_node, weight, enabled, innov in connections:
            ret._node_by_index(out_node).append(count)
            count += 1
        return ret

    def setFitness(self, fitness):
        self.fitness = fitness

    def getFitness(self):
        return self.fitness