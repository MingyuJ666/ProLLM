import json
import pandas as pd
import csv
import random
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Generate ProCoT-Format QA Dataset For Training and Testing')
parser.add_argument('--input_file', type=str, default='protein.actions.SHS148K.txt', help='Path to the input file')
parser.add_argument('--train_file', type=str, default='SHS148K_train.csv', help='Path to the output training file')
parser.add_argument('--test_file', type=str, default='SHS148K_test.csv', help='Path to the output testing file')
parser.add_argument('--train_size', type=float, default=0.7, help='Proportion of data to be used for training')
parser.add_argument('--graph_size', type=int, default=10, help='Max size of the graph for DFS')
parser.add_argument('--total_train', type=int, default=10000, help='Total samples to generate for training')
parser.add_argument('--total_test', type=int, default=2000, help='Total samples to generate for testing')
parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')

args = parser.parse_args()

mode_mapping = {
    "activation": 0,
    "binding": 1,
    "catalysis": 2,
    "expression": 3,
    "inhibition": 4,
    "post-translational": 5,
    "reaction": 6,
    "ptmod": 7
}


def map_mode(mode):
    return mode_mapping.get(mode.lstrip("_"), -1)


def process_and_load_data(input_file):
    data = []
    with open(input_file, "r") as infile:
        for line in infile:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # Skip lines that don't have at least 3 parts
            node1, node2, mode = parts[:3]
            node1 = node1.replace('9606.', '')
            node2 = node2.replace('9606.', '')
            mode_mapped = map_mode(mode)
            data.append((node1, node2, mode_mapped))
    return data


def build_graph(data):
    graph = {}
    nodes = set()
    for node1, node2, mode_mapped in data:
        nodes.add(node1)
        nodes.add(node2)
        if node1 not in graph:
            graph[node1] = {}
        graph[node1][node2] = mode_mapped
    return graph, list(nodes)


def dfs(graph, size, total, writer):
    pos_count = 0
    term = True

    unique_rows = set()
    node_list = list(graph.keys())

    while pos_count < total:
        visited = set()

        graph_size = random.randint(4, size)
        first_node = random.choice(node_list)
        visited.add(first_node)
        last_node = ""
        previous_node = first_node
        input_text = ""
        output_text = ""

        while len(visited) < graph_size:
            if previous_node not in graph or set(graph[previous_node].keys()).issubset(visited):
                node = random.choice(node_list)
                while node in visited:
                    node = random.choice(node_list)
                input_text += "{} not connected with {}. ".format(previous_node, node)
                visited.add(node)
                previous_node = node
            else:
                node = random.choice(list(graph[previous_node].keys()))
                while node in visited:
                    node = random.choice(list(graph[previous_node].keys()))
                relation = graph[previous_node][node]
                text_relation = list(mode_mapping.keys())[list(mode_mapping.values()).index(relation)]
                input_text += '{} has relation_{} with {}, which means {} {} {}. '.format(previous_node, relation, node,
                                                                                          previous_node, text_relation,
                                                                                          node)
                visited.add(node)
                previous_node = node
        if len(visited) == graph_size:
            last_node = previous_node

        was = len(unique_rows)
        unique_rows.add(input_text)
        if len(unique_rows) > was:
            if first_node in graph and last_node in graph[first_node] and term:
                relation = graph[first_node][last_node]
                text_relation = list(mode_mapping.keys())[list(mode_mapping.values()).index(relation)]
                output_text += 'The relation is {}.'.format(text_relation)
                prompt = 'What is the relationship between {} and {}?'.format(first_node, last_node)
                writer.writerow({'input_text': input_text + prompt, 'output_text': output_text})
                pos_count += 1
                term = False
            elif last_node in graph and first_node in graph[last_node] and term:
                relation = graph[last_node][first_node]
                text_relation = list(mode_mapping.keys())[list(mode_mapping.values()).index(relation)]
                output_text += 'The relation is {}.'.format(text_relation)
                prompt = 'What is the relationship between {} and {}?'.format(last_node, first_node)
                writer.writerow({'input_text': input_text + prompt, 'output_text': output_text})
                pos_count += 1
                term = False
            else:
                term = True
        else:
            continue

    print(f"Generated {pos_count} data points.")


input_file_path = args.input_file


data = process_and_load_data(input_file_path)
print("Data has been loaded.")


train_data, test_data = train_test_split(data, test_size=1 - args.train_size, random_state=args.random_state)


train_graph, train_nodes = build_graph(train_data)
test_graph, test_nodes = build_graph(test_data)


with open(args.train_file, mode="w", newline='') as tra:
    writer = csv.DictWriter(tra, fieldnames=['input_text', 'output_text'])
    writer.writeheader()
    dfs(train_graph, args.graph_size, args.total_train, writer)

with open(args.test_file, mode="w", newline='') as tes:
    writer = csv.DictWriter(tes, fieldnames=['input_text', 'output_text'])
    writer.writeheader()
    dfs(test_graph, args.graph_size, args.total_test, writer)
