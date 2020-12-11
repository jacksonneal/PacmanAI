#ifndef _GENES_H
#define _GENES_H
#include <map>
#include <utility>
#include <vector>

struct _Metaparameters {

	float c1, c2, c3;
	float perturbation_chance;
	float perturbation_stdev;
	float reset_weight_chance;
	float new_link_chance;
	float bias_link_chance;
	float new_link_weight_stdev;
	float new_node_chance;
	float disable_mutation_chance;
	float enable_mutation_chance;
	bool allow_recurrent;
	unsigned int mutate_loop;
	unsigned int innovation_number;

	std::map<std::pair<unsigned int, unsigned int>, unsigned int> _connections;
	std::map<std::pair<unsigned int, std::pair<unsigned int, unsigned int>>, unsigned int> _node_splits;

	_Metaparameters() noexcept {}

	void reset_tracking();

	unsigned int register_connection(unsigned int in_node, unsigned int out_node);

	std::pair<unsigned int, unsigned int> register_node_split(unsigned int in_node, unsigned int out_node, unsigned int between_node);
};

struct connection_t {
	unsigned int in_node, out_node;
	float weight;
	bool enabled;
	unsigned int innov_number;
};

class _Network {
public:
	unsigned int _num_inputs;
	unsigned int _num_outputs;
	using incoming_indices_t = std::vector<unsigned int>;
	std::vector<incoming_indices_t> _dynamic_nodes;
	std::vector<connection_t> _connections;
	bool _connections_sorted;
	using changed_t = bool;

	constexpr static changed_t changed = true;

	unsigned int input_count() const noexcept
	{
		return _num_inputs;
	}

	unsigned int output_count() const noexcept
	{
		return _num_outputs;
	}

	unsigned int total_neurons() const noexcept
	{
		return _num_inputs + _dynamic_nodes.size() + 1;
	}

	void feed_sensor_values(float const* const input, float* const neurons) const noexcept;

	void feed_sensor_values(float* const neurons) const noexcept;

	template<typename T>
	T* output_neuron_location(T* base) const noexcept
	{
		return base + _num_inputs + 1;
	}

	_Network(): _Network(0, 0) {}
	_Network(unsigned int num_inputs, unsigned int num_outputs);

	void move_from(_Network& other) noexcept
	{
		*this = std::move(other);
	}

	void copy_from(_Network const& other)
	{
		*this = other;
	} 

	unsigned int input_node_index(unsigned int input_index) const noexcept;

	unsigned int output_node_index(unsigned int output_index) const noexcept;

	changed_t add_connection(unsigned int in_node_index, unsigned out_node_index, _Metaparameters& _metaparameters);

	void perturb(_Metaparameters& metaparameters);

	bool mutate(_Metaparameters& metaparameters);

	friend _Network breed(_Network const& a, _Network const& b, double fitness_a, double fitness_b, _Metaparameters const& metaparameters);

	friend double distance(_Network const& a, _Network const& b, _Metaparameters const& metaparameters);

	void _synchronize_connections();

private:

	std::vector<connection_t> const& _get_sorted_connections(std::vector<connection_t>& space) const;

	bool _is_output_node_index(unsigned int index) const noexcept;

	bool _is_hidden_node_index(unsigned int index) const noexcept;

	changed_t _add_connection(_Metaparameters& _metaparameters);

	changed_t _add_node(_Metaparameters& _metaparameters);

	void _enable_mutation(bool enable);

	template<typename Range>
	static void _sort_connections(Range&& range) noexcept
	{
		std::sort(range.begin(), range.end(), [](connection_t const& a, connection_t const& b)
				  {
					  return a.innov_number < b.innov_number;
				  });
	}
};
#endif