import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab

import sys

import json

import math

if len(sys.argv) == 3 or len(sys.argv) == 2:
    gene_file = open(sys.argv[1], "r")
    gene = json.load(gene_file)
    gene_file.close()
    graph = nx.DiGraph()
    edge_labels = {}
    node_count = gene["nodeCount"]
    input_count = gene["inputCount"]
    output_count = gene["outputCount"]
    for i in range(node_count):
        graph.add_node(i)
    labels_count = 1
    node_labels = {0: "bias"}
    node_colors = ["#ff7f7f"]
    for i in range(input_count):
        node_labels[labels_count] = f"I{i}"
        labels_count += 1
        node_colors.append("#ff5555")
    for i in range(output_count):
        node_labels[labels_count] = f"O{i}"
        labels_count += 1
        node_colors.append("#5555ff")
    for i in range(node_count - len(node_colors)):
        node_labels[labels_count] = f"H{i}"
        labels_count += 1
        node_colors.append("#eeeeee")
    max_weight = 0
    min_weight = 0
    for in_node, out_node, weight, enabled, innov in gene["connections"]:
        if enabled:
            graph.add_edge(in_node, out_node, weight=weight)
            if weight < min_weight:
                min_weight = weight
            elif weight > max_weight:
                max_weight = weight
    edge_colors = []
    for u, v, d in graph.edges(data=True):
        weight = d["weight"]
        if weight == 0:
            edge_colors.append("#000000")
        elif weight < 0:
            percentage = weight / min_weight
            val = 200 - math.floor((percentage * 0.5 + 0.5) * 200)
            edge_colors.append("#ff%02x%02x" % (val, val))
        else:
            percentage = weight / min_weight
            val = 200 - math.floor((percentage * 0.5 + 0.5) * 200)
            edge_colors.append("#%02x%02xff" % (val, val))
    pos = []
    with_bias = input_count + 1
    for i in range(with_bias):
        pos.append((0, i / input_count))
    if output_count == 1:
        pos.append((1, 0.5))
    else:
        for i in range(output_count):
            pos.append((1, i / (output_count - 1)))

    hidden_count = node_count - with_bias - output_count
    if hidden_count == 1:
        pos.append((0.5, 0.5))
    elif hidden_count > 1:
        y_positions = list(range(0, hidden_count))
        np.random.shuffle(y_positions)
        for i in range(hidden_count):
            pos.append((0.1 + 0.8 * i / (hidden_count - 1), 0.1 + 0.8 * y_positions[i] / (hidden_count - 1)))
    nx.draw(graph, pos, labels=node_labels, node_color=node_colors, with_labels=True, edge_color=edge_colors)
    if len(sys.argv) == 3:
        plt.savefig(sys.argv[2])
    else:
        plt.show()
    pass

else:
    print(f"Usage: {sys.argv[0]} gene_file out_file")
