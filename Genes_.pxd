from libcpp cimport bool
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp.vector cimport vector

cdef extern from "Genes_.cpp":
    pass

cdef extern from "Genes_.h":
    cdef cppclass _Metaparameters:
        _Metaparameters()
        float c1, c2, c3
        float perturbation_chance
        float perturbation_stdev
        float reset_weight_chance
        float new_link_chance
        float bias_link_chance
        float new_link_weight_stdev
        float new_node_chance
        float disable_mutation_chance
        float enable_mutation_chance
        bool allow_recurrent
        unsigned int mutate_loop
        unsigned int innovation_number
        map[pair[int, int], int] _connections
        map[pair[int, pair[int, int]], int] _node_splits
        void reset_tracking()

    cdef cppclass connection_t:
        unsigned int in_node, out_node
        float weight
        bool enabled
        unsigned int innov_number

    cdef cppclass _Network:
        _Network() except +
        _Network(unsigned int, unsigned int) except +
        void move_from(_Network&)
        void copy_from(const _Network&)
        unsigned int _num_inputs;
        unsigned int _num_outputs;
        vector[vector[int]] _dynamic_nodes
        vector[connection_t] _connections
        bool _connections_sorted
        unsigned int input_count() const
        unsigned int output_count() const
        unsigned int total_neurons() const
        void feed_sensor_values(float*) const
        void feed_sensor_values(const float*, float*) const
        unsigned int input_node_index(unsigned int) const
        unsigned int output_node_index(unsigned int) const
        bool add_connection(unsigned int, unsigned int, _Metaparameters&)
        void perturb(_Metaparameters&)
        bool mutate(_Metaparameters&)
        void _synchronize_connections()
    
    _Network breed(const _Network&, const _Network&, double, double, const _Metaparameters&)
    double distance(const _Network&, const _Network&, const _Metaparameters&)