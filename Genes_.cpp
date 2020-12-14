// Genes.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <vector>
#include <random>
#include "Genes_.h"
#include <iostream>

auto& random_generator() noexcept
{
	thread_local std::random_device rd;
	thread_local std::mt19937 gen(rd());
	return gen;
}

float random_uniform0(float range) noexcept
{
	std::uniform_real_distribution<float> dis(-range, range);
	return dis(random_generator());
}

bool flip_coin(float p) noexcept
{
	std::uniform_real_distribution<float> dis(0, 1);
	return dis(random_generator()) < p;
}

template<typename T, typename U>
auto randint(T min, U max) noexcept
{
	std::uniform_int_distribution<std::common_type_t<T, U>> dis(min, max);
	return dis(random_generator());
}

template<typename Range>
auto random_choice(Range& range) -> decltype(range.size(), range[0])
{
	return range[randint(0, range.size() - 1)];
}

void _Metaparameters::reset_tracking()
{
	_connections.clear();
	_node_splits.clear();
}

unsigned int _Metaparameters::register_connection(unsigned int in_node, unsigned int out_node)
{
	auto const pair = decltype(_connections)::key_type{ in_node, out_node };
	auto const [loc, inserted] = _connections.insert({ pair, 0 });
	if (inserted)
	{
		loc->second = ++innovation_number;
	}
	return loc->second;
}

std::pair<unsigned int, unsigned int> _Metaparameters::register_node_split(unsigned int in_node, unsigned int out_node, unsigned int between_node)
{
	auto const pair = decltype(_node_splits)::key_type{ in_node, {out_node, between_node} };
	auto const [loc, inserted] = _node_splits.insert({ pair, 0 });
	if (inserted)
	{
		loc->second = ++innovation_number;
		++innovation_number;
	}
	return { loc->second, loc->second + 1 };
}

void _Network::feed_sensor_values(float const* input, float* const neurons) const noexcept{
	neurons[0] = 1;
	std::copy(input, input + _num_inputs, neurons + 1);
	feed_sensor_values(neurons);
}

void _Network::feed_sensor_values(float* const neurons) const noexcept
{
	auto const feed = [&, dynamic_neurons = neurons + _num_inputs + 1](unsigned int node_index)
	{
		auto const& node = _dynamic_nodes[node_index];
		double sum = 0;
		bool has_connections = false;
		for (auto const connection_index : node)
		{
			auto const& connection = _connections[connection_index];
			if (connection.enabled)
			{
				has_connections = true;
				sum += neurons[connection.in_node] * double(connection.weight);
			}
		}
		if (has_connections)
		{
			dynamic_neurons[node_index] = std::tanh(sum);
		}
	};

	for (unsigned int i = _num_outputs; i < _dynamic_nodes.size(); ++i)
	{
		feed(i);
	}
	for (unsigned int i = 0; i < _num_outputs; ++i)
	{
		feed(i);
	}
}

_Network::_Network(unsigned int num_inputs, unsigned int num_outputs):_num_inputs(num_inputs), _num_outputs(num_outputs), _dynamic_nodes(num_outputs), _connections_sorted(true)
{}

unsigned int _Network::input_node_index(unsigned int input_index) const noexcept
{
	return input_index + 1;
}

unsigned int _Network::output_node_index(unsigned int output_index) const noexcept
{
	return output_index + input_count() + 1;
}

_Network::changed_t _Network::add_connection(unsigned int in_node_index, unsigned out_node_index, _Metaparameters& _metaparameters)
{
	if (!_metaparameters.allow_recurrent)
	{
		if (in_node_index == out_node_index)
		{
			return false;
		}
		if (_is_output_node_index(in_node_index))
		{
			if (out_node_index < in_node_index || _is_hidden_node_index(out_node_index))
			{
				std::swap(in_node_index, out_node_index);
			}
		}
		else if (_is_hidden_node_index(out_node_index) && out_node_index < in_node_index)
		{
			std::swap(in_node_index, out_node_index);
		}
	}
	auto& incoming = _dynamic_nodes[out_node_index - input_count() - 1];
	for (auto const& connection_index : incoming)
	{
		if (_connections[connection_index].in_node == in_node_index)
		{
			return false;
		}
	}
	auto const innovation_number = _metaparameters.register_connection(in_node_index, out_node_index);
	incoming.push_back(_connections.size());
	if (_connections.size() && innovation_number < _connections.back().innov_number)
	{
		_connections_sorted = false;
	}
	_connections.push_back({ in_node_index, out_node_index, random_uniform0(_metaparameters.new_link_weight_stdev), true, innovation_number });
	return true;
}

void _Network::perturb(_Metaparameters& metaparameters)
{
	auto const reset_chance = metaparameters.reset_weight_chance;
	auto const reset_std = metaparameters.new_link_weight_stdev;
	auto const pert_std = metaparameters.perturbation_stdev;
	for (auto& connection : _connections)
	{
		if (flip_coin(reset_chance))
		{
			connection.weight = random_uniform0(reset_std);
		}
		else
		{
			connection.weight += random_uniform0(pert_std);
		}
	}
}

bool _Network::mutate(_Metaparameters& metaparameters)
{
	bool changed = false;
	for (unsigned int i = 0; i < metaparameters.mutate_loop; ++i)
	{
		if (flip_coin(metaparameters.new_link_chance))
		{
			changed |= _add_connection(metaparameters);
		}
		if (flip_coin(metaparameters.new_node_chance))
		{
			changed |= _add_node(metaparameters);
		}
		if (flip_coin(metaparameters.perturbation_chance))
		{
			perturb(metaparameters);
			changed = true;
		}
		if (flip_coin(metaparameters.disable_mutation_chance))
		{
			_enable_mutation(false);
			changed = true;
		}
	}
	return changed;
}

_Network breed(_Network const& a, _Network const& b, double fitness_a, double fitness_b, _Metaparameters const& metaparameters)
{
	auto ret = _Network(a._num_inputs, a._num_outputs);
	std::vector<connection_t> _a_connections, _b_connections;
	auto const& a_connections = a._get_sorted_connections(_a_connections);
	auto const& b_connections = b._get_sorted_connections(_b_connections);
	auto const a_len = a_connections.size();
	auto const b_len = b_connections.size();
	std::size_t i = 0;
	std::size_t j = 0;
	auto ret_add_connection = [&](connection_t new_connection)
	{
		if (flip_coin(metaparameters.enable_mutation_chance))
		{
			new_connection.enabled = true;
		}
		ret._connections.push_back(new_connection);
	};
	while (i < a_len && j < b_len)
	{
		auto& sc = a_connections[i];
		auto& so = b_connections[j];
		auto const sci = sc.innov_number;
		auto const soi = so.innov_number;
		if (sci == soi)
		{
			ret_add_connection(flip_coin(0.5) ? sc : so);
			++i;
			++j;
		}
		else if (sci < soi)
		{
			++i;
			if (fitness_a > fitness_b || ((fitness_a == fitness_b) && flip_coin(0.5)))
			{
				ret_add_connection(sc);
			}
		}
		else
		{
			++j;
			if (fitness_a < fitness_b || ((fitness_a == fitness_b) && flip_coin(0.5)))
			{
				ret_add_connection(so);
			}
		}
	}
	if (fitness_a > fitness_b)
	{
		for (; i < a_len; ++i)
		{
			ret_add_connection(a_connections[i]);
		}
	}
	else if (fitness_a == fitness_b)
	{
		for (; i < a_len; ++i)
		{
			if (flip_coin(0.5))
			{
				ret_add_connection(a_connections[i]);
			}
		}
		for (; j < b_len; ++j)
		{
			if (flip_coin(0.5))
			{
				ret_add_connection(b_connections[j]);
			}
		}
	}
	else
	{
		for (; j < b_len; ++j)
		{
			ret_add_connection(b_connections[j]);
		}
	}

	auto const max_node = std::max(a._dynamic_nodes.size(), b._dynamic_nodes.size());
	ret._dynamic_nodes.resize(max_node);
	auto const dynamic_node_base = ret._dynamic_nodes.data() - ret.input_count() - 1;
	for (std::size_t i = 0; i < ret._connections.size(); ++i)
	{
		dynamic_node_base[ret._connections[i].out_node].push_back(static_cast<unsigned int>(i));
	}
	return ret;
}

double distance(_Network const& a, _Network const& b, _Metaparameters const& metaparameters)
{
	std::vector<connection_t> _a_con, _b_con;
	auto const& a_connections = a._get_sorted_connections(_a_con);
	auto const& b_connections = b._get_sorted_connections(_b_con);
	auto const alen = a_connections.size();
	auto const blen = b_connections.size();
	auto const n = std::max(alen, blen);
	if (n == 0)
	{
		return 0;
	}
	auto const c1 = metaparameters.c1;
	auto const c2 = metaparameters.c2;
	auto const c3 = metaparameters.c3;
	double disjoint = 0;
	double weight_difference = 0;
	double shared = 0;
	std::size_t i = 0;
	std::size_t j = 0;
	while (i < alen && j < blen)
	{
		auto const& sc = a_connections[i];
		auto const& so = b_connections[j];
		auto const sci = sc.innov_number;
		auto const soi = so.innov_number;
		if (sci == soi)
		{
			weight_difference += std::abs(double(sc.weight) - so.weight);
			shared += 1;
			++i;
			++j;
		}
		else
		{
			disjoint += 1;
			if (sci < soi)
			{
				++i;
			}
			else
			{
				++j;
			}
		}
	}
	auto const excess = double(blen - j + alen - i);
	return (c1 * excess / n) + (c2 * disjoint / n) + (shared == 0 ? 0.0 : c3 * weight_difference / shared);
}

std::vector<connection_t> const& _Network::_get_sorted_connections(std::vector<connection_t>& space) const
{
	if (_connections_sorted)
	{
		return _connections;
	}
	space = _connections;
	_sort_connections(space);
	return space;
}

bool _Network::_is_output_node_index(unsigned int index) const noexcept
{
	return index > input_count() && index < (1 + input_count() + output_count());
}

bool _Network::_is_hidden_node_index(unsigned int index) const noexcept
{
	return index >= 1 + input_count() + output_count();
}

_Network::changed_t _Network::_add_connection(_Metaparameters& _metaparameters)
{
	auto const total_nodes = total_neurons();
	auto const in_node = flip_coin(_metaparameters.bias_link_chance) ? 0 : randint(0, total_nodes - 1);
	auto const out_node = randint(_num_inputs + 1, total_nodes - 1);
	return add_connection(in_node, out_node, _metaparameters);
}

_Network::changed_t _Network::_add_node(_Metaparameters& _metaparameters)
{
	auto& connections = _connections;
	if (connections.size() == 0)
	{
		return false;
	}
	auto& connection = [&]() -> auto&
	{
		if (_metaparameters.allow_recurrent)
		{
			return random_choice(connections);
		}
		std::vector<unsigned int> choices;
		for (unsigned int i = 0; i < output_count(); ++i)
		{
			auto& node = _dynamic_nodes[i];
			choices.insert(choices.end(), node.begin(), node.end());
		}
		return connections[random_choice(choices)];
	}();
	if (!connection.enabled)
	{
		return false;
	}
	connection.enabled = false;
	auto const new_node_index = static_cast<unsigned int>(total_neurons());
	auto& new_node = _dynamic_nodes.emplace_back();
	auto const [leading_innov, trailing_innov] = _metaparameters.register_node_split(connection.in_node, connection.out_node, new_node_index);
	if (_connections.size() && leading_innov < _connections.back().innov_number)
	{
		_connections_sorted = false;
	}
	_connections.push_back({ connection.in_node, new_node_index, 1.f, true, leading_innov });
	_connections.push_back({ new_node_index, connection.out_node, connection.weight, true, trailing_innov });
	return true;
}

void _Network::_enable_mutation(bool enable)
{
	if (_connections.size())
	{
		random_choice(_connections).enabled = enable;
	}
}

void _Network::_synchronize_connections()
{
	_sort_connections(_connections);
	auto const dynamic_node_base = _dynamic_nodes.data() - input_count() - 1;
	for (std::size_t i = 0; i < _connections.size(); ++i)
	{
		dynamic_node_base[_connections[i].out_node].push_back(static_cast<unsigned int>(i));
	}
}