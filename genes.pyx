from numpy.lib.histograms import _ravel_and_check_weights
from scipy.special import expit as sigmoid
import numpy as np
import util
import copy
import json
import math
import array


class RandomlyTrue:
    def __bool__(self):
        return util.flipCoin(0.5)
    instance = None

cdef class Connection:
    cdef unsigned int in_node, out_node
    cdef float weight
    cdef int enabled
    cdef unsigned int innov_number

    def __init__(self, unsigned int in_node, unsigned int out_node, float weight, int enabled, unsigned int innov):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innov_number = innov

    def to_json(self):
        return [self.in_node, self.out_node, self.weight, self.enabled, self.innov_number]
    
    def get_innov(self):
        return self.innov_number

RandomlyTrue.instance = RandomlyTrue()

def random_uniform0(double half_range):
    # return np.random.normal(0, half_range)
    return np.random.uniform(-half_range, half_range)

class Genes:

    class Metaparameters:
        def __init__(self,
                     c1=1, c2=1, c3=3,
                     perturbation_chance=0.8,
                     perturbation_stdev=0.1,
                     reset_weight_chance=0.1,
                     new_link_chance=0.3,
                     bias_link_chance=0.01,
                     new_link_weight_stdev=1,
                     new_node_chance=0.03,
                     disable_mutation_chance=0.1,
                     enable_mutation_chance=0.25,
                     allow_recurrent=True,
                     mutate_loop=1):
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
            self.allow_recurrent = none_or(allow_recurrent, True)
            self.mutate_loop = none_or(mutate_loop, 1)
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

        def load_from_json(as_json):
            ret = Genes.Metaparameters(
                c1=as_json.get("c1"),
                c2=as_json.get("c2"),
                c3=as_json.get("c3"),
                perturbation_chance=as_json.get("perturbation_chance"),
                perturbation_stdev=as_json.get("perturbation_stdev"),
                reset_weight_chance=as_json.get("reset_weight_chance"),
                new_link_chance=as_json.get("new_link_chance"),
                bias_link_chance=as_json.get("bias_link_chance"),
                new_link_weight_stdev=as_json.get("new_link_weight_stdev"),
                new_node_chance=as_json.get("new_node_chance"),
                disable_mutation_chance=as_json.get("disable_mutation_chance"),
                enable_mutation_chance=as_json.get("enable_mutation_chance"),
                allow_recurrent=as_json.get("allow_recurrent"),
                mutate_loop=as_json.get("mutate_loop")
            )
            if "innovation_number" in as_json:
                ret.innovation_number = as_json["innovation_number"]
            return ret

        def load(in_stream, decoder=json):
            as_json = decoder.load(in_stream)
            return Genes.Metaparameters.load_from_json(as_json)

        def as_json(self):
            return {
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
                "allow_recurrent": self.allow_recurrent,
                "mutate_loop": self.mutate_loop
            }

        def save(self, out_stream, encoder=json):
            out = self.as_json()
            out_stream.write(encoder.dumps(out))
            out_stream.flush()

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
            self.fitness = copy.deepcopy(to_copy.fitness)
        else:
            self._num_sensors = num_sensors_or_copy
            self._num_outputs = num_outputs
            self._dynamic_nodes = []
            for _ in range(num_outputs):
                self._dynamic_nodes.append(array.array("I"))
            self._connections = []
            self._metaparameters = metaparameters
            self._connections_sorted = True
            self.fitness = 0

    def feed_sensor_values(self, values, neurons=None):
        """ Run the network with the given input through the given neurons (creates them if not given), returns neuron values """
        if neurons is None:
            neurons = [0 for _ in range(self.total_nodes() + 1)]
        assert len(values) == self._num_sensors, "invalid number of inputs"
        neurons[0] = 1.0  # BIAS node
        for i in range(self._num_sensors):
            neurons[i + 1] = values[i]

        def feed(unsigned long long node_index):
            node = self._dynamic_nodes[node_index]
            neuron_index = node_index + self._num_sensors + 1
            has_connections = False
            cdef double sum = 0
            cdef Connection connection = None
            for connection_index in node:
                connection = <Connection>self._connections[connection_index]
                # assert out_node = neuron_index
                if connection.enabled:
                    has_connections = True
                    sum += neurons[connection.in_node] * connection.weight
            if has_connections:
                neurons[neuron_index] = math.tanh(sum)
        
        cdef unsigned long long num_outputs = self._num_outputs
        cdef unsigned long long total_dynamic = len(self._dynamic_nodes)

        for hidden_node_index in range(num_outputs, total_dynamic):
            feed(hidden_node_index)
        for output_node_index in range(num_outputs):
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

    def _is_hidden_node_index(self, index):
        return index >= 1 + self._num_sensors + self._num_outputs

    def _is_output_node_index(self, index):
        return index > self._num_sensors and index < 1 + self._num_sensors + self._num_outputs

    def add_connection(self, input_index, output_index):
        if not self._metaparameters.allow_recurrent:
            def swap():
                return output_index, input_index
            if input_index == output_index:
                return
            if self._is_output_node_index(input_index):
                if output_index < input_index or self._is_hidden_node_index(output_index):
                    input_index, output_index = swap()
            elif self._is_hidden_node_index(output_index) and output_index < input_index: # both hidden and output is earlier
                input_index, output_index = swap()
        incoming = self._node_by_index(output_index)

        cdef Connection connection
        for connection_index in incoming:
            connection = <Connection>self._connections[connection_index]
            if connection.in_node == input_index:
                return self

        innovation_number = self._metaparameters.register_connection(input_index, output_index)
        connection = Connection(input_index, output_index, random_uniform0(self._metaparameters.new_link_weight_stdev), True, innovation_number)
        incoming.append(len(self._connections))
        cdef Connection last_connection
        if len(self._connections) > 0:
            last_connection = <Connection>self._connections[-1]
            if innovation_number < last_connection.innov_number:
                self._connections_sorted = False
        self._connections.append(connection)
        return self

    def _add_connection(self):
        total_nodes = self.total_nodes()
        input_index = 0 if util.flipCoin(self._metaparameters.bias_link_chance) else util.random.randint(0, total_nodes - 1)
        output_index = util.random.randint(self._num_sensors + 1, total_nodes - 1)
        self.add_connection(input_index, output_index)

    def _add_node(self):
        if len(self._connections) == 0:
            return
        cdef Connection connection = <Connection>util.random.choice(self._connections)
        connection.enabled = False
        new_node = array.array("I")
        self._dynamic_nodes.append(new_node)
        leading_innov, trailing_innov = self._metaparameters.register_node_split(connection.in_node, connection.out_node, self.total_nodes() - 1)
        leading = Connection(connection.in_node, self.total_nodes() - 1, 1, True, leading_innov)
        trailing = Connection(self.total_nodes() - 1, connection.out_node, connection.weight, True, trailing_innov)
        cdef Connection last_connection
        if len(self._connections) > 0:
            last_connection = <Connection>(self._connections[-1])
            if leading_innov < last_connection.innov_number:
                self._connections_sorted = False
        self._connections.append(leading)
        self._connections.append(trailing)
        new_node.append(len(self._connections) - 2)
        self._node_by_index(connection.out_node).append(len(self._connections) - 1)

    def perturb(self):
        cdef double reset_chance = self._metaparameters.reset_weight_chance
        cdef double new_link_weight_stdev = self._metaparameters.new_link_weight_stdev
        cdef double perturbation_stdev = self._metaparameters.perturbation_stdev
        cdef Connection connection
        for _connection in self._connections:
            connection = <Connection>_connection
            if util.flipCoin(reset_chance):
                connection.weight = random_uniform0(new_link_weight_stdev)
            else:
                connection.weight += random_uniform0(perturbation_stdev)
        return self

    def _enable_mutation(self, enable):
        if len(self._connections) == 0:
            return
        cdef Connection connection = <Connection>util.random.choice(self._connections)
        connection.enabled = enable

    def mutate(self):
        """ Mutate the genes in this genome, returns self """
        for _ in range(self._metaparameters.mutate_loop):
            if util.flipCoin(self._metaparameters.new_link_chance):
                self._add_connection()
            if util.flipCoin(self._metaparameters.new_node_chance):
                self._add_node()
            if util.flipCoin(self._metaparameters.perturbation_chance):
                self.perturb()
        # if util.flipCoin(self._metaparameters.disable_mutation_chance):
        #     self._enable_mutation(False)
        # if util.flipCoin(self._metaparameters.enable_mutation_chance):
        #     self._enable_mutation(True)
        return self

    def _sorted_connections(self):
        if self._connections_sorted:
            return self._connections
        return sorted(self._connections, key=Connection.get_innov)

    def breed(self, other, self_more_fit=RandomlyTrue.instance):
        """ Creates a child from the result of breeding self and other genes, returns new child """
        ret = Genes(self._num_sensors, self._num_outputs, self._metaparameters)

        sconnections = self._sorted_connections()
        oconnections = other._sorted_connections()
        slen = len(sconnections)
        olen = len(oconnections)
        i = 0
        j = 0

        def turn_on_maybe(Connection connection):
            if not connection.enabled and util.flipCoin(self._metaparameters.enable_mutation_chance):
                connection.enabled = True
        cdef Connection sc = None
        cdef Connection so = None
        while i < slen and j < olen:
            sc = <Connection>sconnections[i]
            so = <Connection>oconnections[j]
            sci = sc.innov_number
            soi = so.innov_number
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
        cdef Connection connection
        for _connection in ret._connections:
            connection = <Connection>_connection
            max_node = max(max_node, connection.in_node, connection.out_node)
        i = ret.total_nodes()
        while i <= max_node:
            ret._dynamic_nodes.append([])
            i += 1
        for index, _connection in enumerate(ret._connections):
            connection = <Connection>_connection
            ret._node_by_index(connection.out_node).append(index)
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
        cdef Connection sc, so
        while i < slen and j < olen:
            sc = <Connection>sconnections[i]
            so = <Connection>oconnections[j]
            sci = sc.innov_number
            soi = so.innov_number
            if sci == soi:
                weight_difference += abs(sc.weight - so.weight)
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
        return {"nodeCount": self.total_nodes(), "inputCount": self._num_sensors, "outputCount": self._num_outputs, "connections": [(<Connection>c).to_json() for c in self._connections], "fitness": self.fitness}

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
        connections = [Connection(c[0], c[1], c[2], c[3], c[4]) for c in json_object["connections"]]
        count = 0
        ret._connections = connections
        connections.sort(key=Connection.get_innov)
        cdef Connection connection
        for _connection in connections:
            connection = <Connection>_connection
            ret._node_by_index(connection.out_node).append(count)
            count += 1
        ret.fitness = json_object.get("fitness", 0)
        return ret

    def load(in_stream, metaparameters, decoder=json):
        """ load from stream using given decoder, decoder must define load function that takes in a stream and returns a dict-like object"""
        as_json = decoder.load(in_stream)
        return Genes.load_from_json(as_json, metaparameters)

    def setFitness(self, fitness):
        self.fitness = fitness

    def getFitness(self):
        return self.fitness
