# distutils: language = c++
from Genes_ cimport _Metaparameters
from Genes_ cimport _Network
from Genes_ cimport connection_t
from Genes_ cimport breed
from Genes_ cimport distance
from cpython cimport array
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.vector cimport vector
import array
import json

cdef class Metaparameters:

    cdef _Metaparameters *data

    def __cinit__(self):
        self.data = new _Metaparameters()

    def __dealloc__(self):
        del self.data

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
        def none_or(value, default_value):
            return default_value if value is None else value
        self.data.c1 = none_or(c1, 0.1)
        self.data.c2 = none_or(c2, 0.1)
        self.data.c3 = none_or(c3, 0.1)
        self.data.new_link_chance = none_or(new_link_chance, 0.1)
        self.data.bias_link_chance = none_or(bias_link_chance, 0.1)
        self.data.new_link_weight_stdev = none_or(new_link_weight_stdev, 1)
        self.data.new_node_chance = none_or(new_node_chance, 0.1)
        self.data.perturbation_chance = none_or(perturbation_chance, 0.1)
        self.data.reset_weight_chance = none_or(reset_weight_chance, 0.5)
        self.data.perturbation_stdev = none_or(perturbation_stdev, 0.1)
        self.data.disable_mutation_chance = none_or(disable_mutation_chance, 0.1)
        self.data.enable_mutation_chance = none_or(enable_mutation_chance, 0.1)
        self.data.allow_recurrent = none_or(allow_recurrent, True)
        self.data.mutate_loop = none_or(mutate_loop, 1)
        self.data.innovation_number = 0

    def load_from_json(as_json):
        ret = Metaparameters(
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
            ret.data.innovation_number = as_json["innovation_number"]
        return ret

    def load(in_stream, decoder=json):
        as_json = decoder.load(in_stream)
        return Metaparameters.load_from_json(as_json)

    def as_json(self):
        return {
            "innovation_number": self.data.innovation_number,
            "c1": self.data.c1,
            "c2": self.data.c2,
            "c3": self.data.c3,
            "new_link_chance": self.data.new_link_chance,
            "bias_link_chance": self.data.bias_link_chance,
            "new_link_weight_stdev": self.data.new_link_weight_stdev,
            "new_node_chance": self.data.new_node_chance,
            "perturbation_chance": self.data.perturbation_chance,
            "perturbation_stdev": self.data.perturbation_stdev,
            "reset_weight_chance": self.data.reset_weight_chance,
            "disable_mutation_chance": self.data.disable_mutation_chance,
            "enable_mutation_chance": self.data.enable_mutation_chance,
            "allow_recurrent": self.data.allow_recurrent,
            "mutate_loop": self.data.mutate_loop
        }

    def save(self, out_stream, encoder=json):
        out = self.as_json()
        out_stream.write(encoder.dumps(out))
        out_stream.flush()

    def reset_tracking(self):
        self.data.reset_tracking()

cdef class Network:
    cdef _Network* data

    def __cinit__(self, num_sensors, num_outputs):
        self.data = new _Network(num_sensors, num_outputs)

    def __dealloc__(self):
        del self.data

    def feed_sensor_values(self, values, neurons=None):
        """ Run the network with the given input through the given neurons (creates them if not given), returns neuron values """
        if neurons is None:
            neurons = array.array("f", [0] * (self.data.total_neurons() * 1))
        neurons[0] = 1
        for i in range(self.data.input_count()):
            neurons[i + 1] = values[i]
        cdef array.array arr = <array.array?>neurons
        self.data.feed_sensor_values(arr.data.as_floats)
        return neurons

class Genes:

    Metaparameters = Metaparameters

    def __init__(self, num_sensors, num_outputs, metaparameters):
        if num_sensors is not None:
            self.network = Network(num_sensors, num_outputs)
        else:
            self.network = Network(0, 0)
        self._metaparameters = metaparameters
        self.fitness = 0

    def setFitness(self, value):
        self.fitness = value

    def getFitness(self):
        return self.fitness

    def input_node_index(self, input_index):
        cdef Network network = <Network>self.network
        return deref(network.data).input_node_index(input_index)

    def output_node_index(self, index):
        cdef Network network = <Network>self.network
        return deref(network.data).output_node_index(index)

    def feed_sensor_values(self, values, neurons=None):
        cdef Network network = <Network>self.network
        return network.feed_sensor_values(values, neurons)

    def extract_output_values(self, neurons):
        cdef Network network = <Network>self.network
        num_inputs = deref(network.data).input_count() + 1
        num_outputs = deref(network.data).output_count()
        return neurons[num_inputs: num_inputs + num_outputs]


    def add_connection(self, input_index, output_index):
        cdef Network network = <Network>self.network
        cdef Metaparameters meta = <Metaparameters>self._metaparameters
        
        return deref(network.data).add_connection(input_index, output_index, deref(meta.data))

    def perturb(self):
        cdef Network network = <Network>self.network
        cdef Metaparameters meta = <Metaparameters>self._metaparameters
        deref(network.data).perturb(deref(meta.data))

    def mutate(self):
        cdef Network network = <Network>self.network
        cdef Metaparameters meta = <Metaparameters>self._metaparameters
        return deref(network.data).mutate(deref(meta.data))

    def breed(self, other, _ignore=False):
        cdef Network network = <Network>self.network
        cdef Network onetwork = <Network>self.network
        cdef Metaparameters meta = <Metaparameters>self._metaparameters
        ret = Genes(None, None, self._metaparameters)
        net = breed(deref(network.data), deref(onetwork.data), self.fitness, other.fitness, deref(meta.data))
        cdef Network rnetwork = <Network>ret.network
        deref(rnetwork.data).move_from(net)
        return ret

    def distance(self, other):
        cdef Network network = <Network>self.network
        cdef Network onetwork = <Network>self.network
        cdef Metaparameters meta = <Metaparameters>self._metaparameters
        return distance(deref(network.data), deref(onetwork.data), deref(meta.data))

    def clone(self):
        ret = Genes(None, None, self._metaparameters)
        cdef Network rnetwork = <Network>ret.network
        cdef Network network = <Network>self.network
        deref(rnetwork.data).copy_from(deref(network.data))
        return ret

    def as_json(self):
        cdef Network network = <Network>self.network
        connections = []
        net_connections = &deref(network.data)._connections
        begin = net_connections.begin()
        end = net_connections.end()
        while begin != end:
            conn = deref(begin)
            connections.append([conn.in_node, conn.out_node, conn.weight, conn.enabled, conn.innov_number])
            inc(begin)
        """ returns self as a dict """
        return {
            "nodeCount": network.data.total_neurons(), 
            "inputCount": network.data.input_count(), 
            "outputCount": network.data.output_count(), 
            "connections": connections, 
            "fitness": self.fitness
        }

    def save(self, out_stream, encoder=json):
        """ save to the stream using the given encoder, encoder must define dumps function that takes in a JSON-like object"""
        as_json = self.as_json()
        out_stream.write(encoder.dumps(as_json))
        out_stream.flush()

    def load_from_json(json_object, metaparameters):
        """ loads from a dict-like object """
        ret = Genes(0, 0, metaparameters)
        cdef Network network = <Network>ret.network
        net_connections = &deref(network.data)._connections
        net_connections.resize(json_object["nodeCount"])
        network.data._num_inputs = json_object["inputCount"]
        network.data._num_outputs = json_object["outputCount"]
        cdef connection_t conn
        for in_node, out_node, weight, enabled, innov in json_object["connections"]:
            conn.in_node = in_node
            conn.out_node = out_node
            conn.weight = weight
            conn.enabled = enabled
            conn.innov_number = innov
            net_connections.push_back(conn)
        network.data._synchronize_connections()
        ret.fitness = json_object.get("fitness", 0)
        return ret

    def load(in_stream, metaparameters, decoder=json):
        """ load from stream using given decoder, decoder must define load function that takes in a stream and returns a dict-like object"""
        as_json = decoder.load(in_stream)
        return Genes.load_from_json(as_json, metaparameters)


