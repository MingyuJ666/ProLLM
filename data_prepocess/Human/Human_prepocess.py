import json
import pandas as pd
from collections import deque
import csv
import random
from sklearn.model_selection import train_test_split
import os
import argparse

# Add parser option for multi-label handling
parser = argparse.ArgumentParser(description='Generate ProCoT-Format QA Dataset For Training and Testing')
parser.add_argument('--input_file', type=str, default='Human_PPI.tsv', help='Path to the input file')
parser.add_argument('--train_file', type=str, default='Human_train.csv', help='Path to the output training file')
parser.add_argument('--test_file', type=str, default='Human_test.csv', help='Path to the output testing file')
parser.add_argument('--train_size', type=float, default=0.7, help='Proportion of data to be used for training')
parser.add_argument('--graph_size', type=int, default=10, help='The max size of the graph for DFS')
parser.add_argument('--total_train', type=int, default=30000, help='Total samples to generate for training')
parser.add_argument('--total_test', type=int, default=5000, help='Total samples to generate for testing')
parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--multi_label', action='store_true', help='Flag to indicate multi-label output (default is single label)')

args = parser.parse_args()

input_file = open(args.input_file, "r")

nodes = set()
next(input_file)
graph = {}

for line in input_file:
    node1, node2, relation = line.strip().split()
    node1 = node1.replace('9606.', '')
    node2 = node2.replace('9606.', '')
    nodes.add(node1)
    relation = int(relation)
    if node1 not in graph:
        graph[node1] = {}
    graph[node1][node2] = relation

node_list = list(nodes)

relation2id = {}
with open("human_PPI_relation.txt", "r") as file:
    for line in file:
        relation, relation_id = line.strip().split("\t")
        relation2id[int(relation_id)] = relation

unique_rows = set()
fieldnames = ['input_text', 'output_text']

def dfs(graph, size, total, writer_tra):
    pos_count = 0
    times = 0
    while times < total:
        visited = set()
        graph_size = random.randint(3, size)
        first_node = random.choice(node_list)
        visited.add(first_node)
        last_node = ""
        previous_node = first_node
        input_text = ""
        output_text = []

        while len(visited) < graph_size:
            if previous_node not in graph or set(graph[previous_node].keys()).issubset(visited):
                node = random.choice(node_list)
                while node in visited:
                    node = random.choice(node_list)
                input_text += "{} not connected with {}.".format(previous_node, node)
                visited.add(node)
                previous_node = node
            else:
                node = random.choice(list(graph[previous_node].keys()))
                while node in visited:
                    node = random.choice(list(graph[previous_node].keys()))
                relation = graph[previous_node][node]
                text_relation = relation2id[int(relation)]
                input_text += '{} has relation_{} with {}, which means {} {} {}.'.format(previous_node, relation, node,
                                                                                         previous_node, text_relation,
                                                                                         node)
                visited.add(node)
                previous_node = node
        if len(visited) == graph_size:
            last_node = previous_node

        was = len(unique_rows)
        unique_rows.add(input_text)
        if len(unique_rows) > was:
            if first_node in graph and last_node in graph[first_node]:
                relation = graph[first_node][last_node]
                text_relation = relation2id[int(relation)]
                output_text.append(f'The relation is {text_relation}.')
                pos_count += 1
                times += 1
            elif last_node in graph and first_node in graph[last_node]:
                relation = graph[last_node][first_node]
                text_relation = relation2id[int(relation)]
                output_text.append(f'The relation is {text_relation}.')
                pos_count += 1
                times += 1

        # Handle multi-label or single-label output
        if args.multi_label:
            output_text = ', '.join(output_text)  # Join multiple relations in a single string for multi-label
        else:
            output_text = output_text[0] if output_text else ''  # Only take the first relation for single-label
        
        # Ensure there is an output_text to write
        if output_text:
            prompt = 'What is the relationship between {} and {}?'.format(first_node, last_node)
            writer_tra.writerow({'input_text': input_text + prompt, 'output_text': output_text})

    print(pos_count)

# Generate training data
with open(args.train_file, mode="w", newline='') as tra:
    writer_tra = csv.DictWriter(tra, fieldnames=fieldnames)
    writer_tra.writeheader()
    dfs(graph, args.graph_size, args.total_train, writer_tra)

# Generate testing data
with open(args.test_file, mode="w", newline='') as test:
    writer_test = csv.DictWriter(test, fieldnames=fieldnames)
    writer_test.writeheader()
    dfs(graph, args.graph_size, args.total_test, writer_test)
