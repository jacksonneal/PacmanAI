import math
from numpy.core.fromnumeric import sort
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

def random_uniform0(half_range):
    return np.random.normal(0, half_range)
    # return np.random.uniform(-half_range, half_range)

class Genes:

    class Metaparameters:
        def __init__(self,
                     c1=1, c2=1, c3=1,
                     perturbation_chance=0.8,
                     perturbation_stdev=0.1,
                     reset_weight_chance=0.1,
                     new_link_chance=0.3,
                     bias_link_chance=0.01,
                     new_link_weight_stdev=1,
                     new_node_chance=0.03,
                     disable_mutation_chance=0.1,
                     enable_mutation_chance=0.25):
            self.innovation_number = 0

            def none_or(value, default_value):
                return default_value if value is None else value
            self.c1 = none_or(c1, 0.1)
            self.c2 = none_or(c2, 0.1)
            self.c3 = none_or(c3, 0.1)
            self.new_link_chance = none_or(new_link_chance, 0.1)
            self.bias_link_chance = none_or(bias_link_chance, 0.1)
            self.new_link_weight_stdev = none_or(new_link_weight_stdev, 1)
            self.new_node_chance = none_or(new_node_chance, 0.1)
            self.perturbation_chance = none_or(perturbation_chance, 0.1)
            self.reset_weight_chance = none_or(reset_weight_chance, 0.5)
            self.perturbation_stdev = none_or(perturbation_stdev, 0.1)
            self.disable_mutation_chance = none_or(disable_mutation_chance, 0.1)
            self.enable_mutation_chance = none_or(enable_mutation_chance, 0.1)
            self._connections = {}
            self._node_splits = {}

        def _increment_innovation(self):
            self.innovation_number += 1
            return self.innovation_number

        def reset_tracking(self):
            self._connections = {}
            self._node_splits = {}

        def register_connection(self, in_node, out_node):
            pair = (in_node, out_node)
            innovation_number = self._connections.get(pair, None)
            if innovation_number is None:
                innovation_number = self._increment_innovation()
                self._connections[pair] = innovation_number
            return innovation_number

        def register_node_split(self, in_node, out_node, between_node):
            tuple = (in_node, out_node, between_node)
            innovation_numbers = self._node_splits.get(tuple, None)
            if innovation_numbers is None:
                leading = self._increment_innovation()
                trailing = self._increment_innovation()
                innovation_numbers = (leading, trailing)
                self._node_splits[tuple] = innovation_numbers
            return innovation_numbers

        def load(in_stream, decoder=json):
            as_json = decoder.load(in_stream)
            ret = Genes.Metaparameters(
                c1=as_json.get("c1"),
                c2=as_json.get("c2"),
                c3=as_json.get("c3"),
                new_link_weight_stdev=as_json.get("new_link_weight_stdev"),
                new_node_chance=as_json.get("new_node_chance"),
                perturbation_chance=as_json.get("perturbation_chance"),
                perturbation_stdev=as_json.get("perturbation_stdev"),
                reset_weight_chance=as_json.get("reset_weight_chance"),
                disable_mutation_chance=as_json.get("disable_mutation_chance"),
                enable_mutation_chance=as_json.get("enable_mutation_chance")
            )
            if "innovation_number" in as_json:
                ret.innovation_number = as_json["innovation_number"]
            return ret

        def save(self, out_stream, encoder=json):
            out = {
                "innovation_number": self.innovation_number,
                "c1": self.c1,
                "c2": self.c2,
                "c3": self.c3,
                "new_link_chance": self.new_link_chance,
                "bias_link_chance": self.bias_link_chance,
                "new_link_weight_stdev": self.new_link_weight_stdev,
                "new_node_chance": self.new_node_chance,
                "perturbation_chance": self.perturbation_chance,
                "perturbation_stdev": self.perturbation_stdev,
                "reset_weight_chance": self.reset_weight_chance,
                "disable_mutation_chance": self.disable_mutation_chance,
                "enable_mutation_chance": self.enable_mutation_chance,
            }
            out_stream.write(encoder.dumps(out))
            out_stream.flush()

    _IN_NODE = 0
    _OUT_NODE = 1
    _WEIGHT = 2
    _ENABLED = 3
    _INNOV_NUMBER = 4
    BIAS_INDEX = 0

    def __init__(self, num_sensors_or_copy, num_outputs=None, metaparameters=None):
        if isinstance(num_sensors_or_copy, Genes):
            to_copy = num_sensors_or_copy
            self._num_sensors = to_copy._num_sensors
            self._num_outputs = to_copy._num_outputs
            self._dynamic_nodes = copy.deepcopy(to_copy._dynamic_nodes)
            self._connections = copy.deepcopy(to_copy._connections)
            self._metaparameters = to_copy._metaparameters
            self._connections_sorted = to_copy._connections_sorted
        else:
            self._num_sensors = num_sensors_or_copy
            self._num_outputs = num_outputs
            self._dynamic_nodes = []
            for _ in range(num_outputs):
                self._dynamic_nodes.append([])
            self._connections = []
            self._metaparameters = metaparameters
            self._connections_sorted = True
        self.fitness = 0

    def feed_sensor_values(self, values, neurons=None):
        """ Run the network with the given input through the given neurons (creates them if not given), returns neuron values """
        if neurons is None:
            neurons = [0] * (len(self._dynamic_nodes) + self._num_sensors + 1)
        assert len(values) == self._num_sensors, "invalid number of inputs"
        neurons[0] = 1  # BIAS node
        for i in range(self._num_sensors):
            neurons[i + 1] = values[i]

        def feed(node_index):
            node = self._dynamic_nodes[node_index]
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

        for hidden_node_index in range(self._num_outputs, len(self._dynamic_nodes)):
            feed(hidden_node_index)
        for output_node_index in range(self._num_outputs):
            feed(output_node_index)

        return neurons

    def extract_output_values(self, neuron_values):
        """ Extracts the output values from the result of feed_sensor_values """
        return neuron_values[self._num_sensors + 1:self._num_sensors + self._num_outputs + 1]

    def _node_by_index(self, index):
        return self._dynamic_nodes[index - self._num_sensors - 1]

    def total_nodes(self):
        return 1 + self._num_sensors + len(self._dynamic_nodes)

    def input_node_index(self, input_index):
        return 1 + input_index

    def output_node_index(self, index):
        return 1 + self._num_sensors + index    

    def add_connection(self, input_index, output_index):
        incoming = self._node_by_index(output_index)
        for connection_index in incoming:
            connection = self._connections[connection_index]
            if connection[Genes._IN_NODE] == input_index:
                return self

        innovation_number = self._metaparameters.register_connection(input_index, output_index)
        connection = [input_index, output_index, random_uniform0(self._metaparameters.new_link_weight_stdev), True, innovation_number]
        incoming.append(len(self._connections))
        if len(self._connections) > 0 and innovation_number < self._connections[-1][Genes._INNOV_NUMBER]:
            self._connections_sorted = False
        self._connections.append(connection)
        return self

    def _add_connection(self):
        total_nodes = self.total_nodes()
        input_index = 0 if util.flipCoin(self._metaparameters.bias_link_chance) else util.random.randint(0, total_nodes - 1)
        output_index = util.random.randint(self._num_sensors, total_nodes - 1)
        self.add_connection(input_index, output_index)

    def _add_node(self):
        if len(self._connections) == 0:
            return
        connection = util.random.choice(self._connections)
        connection[Genes._ENABLED] = False
        in_node, out_node, _a, _b, _c = connection
        new_node = []
        self._dynamic_nodes.append(new_node)
        leading_innov, trailing_innov = self._metaparameters.register_node_split(in_node, out_node, self.total_nodes() - 1)
        leading = [in_node, self.total_nodes() - 1, 1, True, leading_innov]
        trailing = [self.total_nodes() - 1, out_node, connection[Genes._WEIGHT], True, trailing_innov]
        if len(self._connections) > 0 and leading_innov < self._connections[-1][Genes._INNOV_NUMBER]:
            self._connections_sorted = False
        self._connections.append(leading)
        self._connections.append(trailing)
        new_node.append(len(self._connections) - 2)
        self._node_by_index(out_node).append(len(self._connections) - 1)

    def _perturb(self):
        reset_chance = self._metaparameters.reset_weight_chance
        for connection in self._connections:
            if util.flipCoin(reset_chance):
                connection[Genes._WEIGHT] = random_uniform0(self._metaparameters.new_link_weight_stdev)
            else:
                connection[Genes._WEIGHT] += random_uniform0(self._metaparameters.perturbation_stdev)

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
        # if util.flipCoin(self._metaparameters.disable_mutation_chance):
        #     self._enable_mutation(False)
        # if util.flipCoin(self._metaparameters.enable_mutation_chance):
        #     self._enable_mutation(True)
        return self

    def _sorted_connections(self):
        if self._connections_sorted:
            return self._connections
        return sorted(self._connections, key=lambda c: c[Genes._INNOV_NUMBER])

    def breed(self, other, self_more_fit=RandomlyTrue.instance):
        """ Creates a child from the result of breeding self and other genes, returns new child """
        ret = Genes(self._num_sensors, self._num_outputs, self._metaparameters)

        sconnections = self._sorted_connections()
        oconnections = other._sorted_connections()
        slen = len(sconnections)
        olen = len(oconnections)
        i = 0
        j = 0

        def turn_on_maybe(connection):
            if not connection[Genes._ENABLED] and util.flipCoin(self._metaparameters.enable_mutation_chance):
                connection[Genes._ENABLED] = True

        while i < slen and j < olen:
            sc = sconnections[i]
            so = oconnections[j]
            sci = sc[Genes._INNOV_NUMBER]
            soi = so[Genes._INNOV_NUMBER]
            if sci == soi:
                ret._connections.append(copy.copy(sc if util.flipCoin(0.5) else so))
                turn_on_maybe(ret._connections[-1])
                i += 1
                j += 1
            elif sci < soi:
                i += 1
                if self_more_fit:
                    ret._connections.append(copy.copy(sc))
                    turn_on_maybe(ret._connections[-1])
            else:
                j += 1
                if not self_more_fit:
                    ret._connections.append(copy.copy(so))
                    turn_on_maybe(ret._connections[-1])
        while i < slen:
            if self_more_fit:
                ret._connections.append(copy.copy(sconnections[i]))
                turn_on_maybe(ret._connections[-1])
            i += 1
        while j < olen:
            if not self_more_fit:
                ret._connections.append(copy.copy(oconnections[j]))
                turn_on_maybe(ret._connections[-1])
            j += 1
        max_node = 0
        for connection in ret._connections:
            max_node = max(max_node, connection[Genes._IN_NODE], connection[Genes._OUT_NODE])
        i = ret.total_nodes()
        while i <= max_node:
            ret._dynamic_nodes.append([])
            i += 1
        for index, connection in enumerate(ret._connections):
            ret._node_by_index(connection[Genes._OUT_NODE]).append(index)
        return ret

    def distance(self, other):
        assert self._metaparameters is other._metaparameters
        c1 = self._metaparameters.c1
        c2 = self._metaparameters.c2
        c3 = self._metaparameters.c3
        sconnections = self._sorted_connections()
        oconnections = other._sorted_connections()
        slen = len(sconnections)
        olen = len(oconnections)
        n = max(olen, slen)
        if n == 0:
            return 0
        disjoint = 0
        weight_difference = 0
        shared = 0
        i = 0
        j = 0
        while i < slen and j < olen:
            sc = sconnections[i]
            so = oconnections[j]
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

    def as_json(self):
        """ returns self as a dict """
        return {"nodeCount": self.total_nodes(), "inputCount": self._num_sensors, "outputCount": self._num_outputs, "connections": self._connections}

    def save(self, out_stream, encoder=json):
        """ save to the stream using the given encoder, encoder must define dumps function that takes in a JSON-like object"""
        as_json = self.as_json()
        out_stream.write(encoder.dumps(as_json))
        out_stream.flush()

    def load_from_json(json_object, metaparameters):
        """ loads from a dict-like object """
        ret = Genes(json_object["inputCount"], json_object["outputCount"], metaparameters)
        to_add = json_object["nodeCount"] - ret.total_nodes()
        for _ in range(to_add):
            ret._dynamic_nodes.append([])
        connections = json_object["connections"]
        count = 0
        ret._connections = connections
        connections.sort(key=lambda c: c[Genes._INNOV_NUMBER])
        for in_node, out_node, weight, enabled, innov in connections:
            ret._node_by_index(out_node).append(count)
            count += 1
        return ret

    def load(in_stream, metaparameters, decoder=json):
        """ load from stream using given decoder, decoder must define load function that takes in a stream and returns a dict-like object"""
        as_json = decoder.load(in_stream)
        return Genes.load_from_json(as_json, metaparameters)

    def setFitness(self, fitness):
        self.fitness = fitness

    def getFitness(self):
        return self.fitness
