import json
import pandas as pd
from collections import deque
import csv
import random
from sklearn.model_selection import train_test_split
import os



mode_mapping = {
    "activation": 0,
    "binding": 1,
    "catalysis": 2,
    "expression": 3,
    "inhibition": 4,
    "post-translational": 5,
    "reaction": 6,
    "ptmod":7
}

def map_mode(mode):
    # Remove leading underscores and map to number
    return mode_mapping.get(mode.lstrip("_"), -1)

def save_mapping(mapping, file_path):
    with open(file_path, "w") as file:
        for mode, value in mapping.items():
            file.write(f"{mode}\t{value}\n")

def process_file(input_file, output_file, mapping_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:

            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # Skip lines that don't have at least 3 parts
            # Map the mode and retain the first three columns
            parts[2] = str(map_mode(parts[2]))
            outfile.write("\t".join(parts[:3]) + "\n")

    # Save the mapping to a file
    save_mapping(mode_mapping, mapping_file)

def remove_prefix_and_overwrite(input_file):
    # Read and process the file content
    with open(input_file, 'r') as file:
        lines = file.readlines()

        modified_lines = []
        for line in lines:
            parts = line.strip().split('\t')
            parts[0] = parts[0].replace('9606.', '')
            parts[1] = parts[1].replace('9606.', '')
            modified_lines.append('\t'.join(parts) + '\n')

    # Overwrite the original file with modified content
    with open(input_file, 'w') as file:
        file.writelines(modified_lines)

def dfs(graph, size):
            pos_count = 0
            neg_count = 0
            times = 0
            term = True

            while times < total:
                 visited = set()

                 kg = []
                 graph_size = random.randint(4, size)
                 first_node = random.choice(node_list)
                 visited.add(first_node)
                 last_node = ""
                 previous_node = first_node
                 stack = [first_node]
                 input_text = ""
                 output_text = ""
                 while len(visited) < graph_size:
                     if previous_node not in graph or set(graph[previous_node].keys()).issubset(visited):
                         node = random.choice(node_list)
                         while node in visited:
                             node = random.choice(node_list)
                         input_text += "{} not connected with {}.".format(previous_node,node)
                         visited.add(node)
                         previous_node = node

                     else:
                         node = random.choice(list(graph[previous_node].keys()))
                         while node in visited:
                             node = random.choice(list(graph[previous_node].keys()))
                         relation = graph[previous_node][node]

                         text_relation = relation2id[int(relation)]

                         input_text += '{} has relation_{} with {}, which means {} {} {}.'.format(previous_node,relation,node,previous_node,text_relation,node)
                         visited.add(node)
                         previous_node = node
                 if len(visited) == graph_size:
                    last_node = previous_node

                 was =len(unique_rows)
                 unique_rows.add(input_text)
                 if len(unique_rows) > was:
                      if first_node in graph and last_node in graph[first_node] and term:

                          relation = graph[first_node][last_node]
                          text_relation = relation2id[int(relation)]
                          output_text += 'The relation is {}.'.format(text_relation)

                          prompt = 'What is the relationship between {} and {}.'.format(first_node, last_node)
                          writer_pos.writerow({'input_text': input_text + prompt, 'output_text': output_text})
                          writer_tra.writerow({'input_text': input_text + prompt, 'output_text': output_text})
                          pos_count += 1
                          term = False
                          times += 1

                      elif last_node in graph and first_node in graph[last_node] and term:

                          relation = graph[last_node][first_node]
                          text_relation = relation2id[int(relation)]
                          output_text += 'The relation is {}.'.format(text_relation)


                          prompt = 'What is the relationship between {} and {}?'.format(last_node, first_node)
                          writer_pos.writerow({'input_text': input_text + prompt, 'output_text': output_text})
                          writer_tra.writerow({'input_text': input_text + prompt, 'output_text': output_text})
                          pos_count += 1
                          term = False
                          times += 1

                      else:
                          term = True
                          times += 1



                 else:
                     continue
            print(pos_count)



input_file_path = "protein.actions.SHS148K.txt"  # Update this path
output_file_path = "SHS148K.txt"  # Update this path
mapping_file_path = "SHS_String_relation.txt"  # Update this path

# Process the file and save mapping
process_file(input_file_path, output_file_path, mapping_file_path)
print("Files have been processed and saved.")

remove_prefix_and_overwrite(output_file_path)

input_file = open('SHS148K.txt', "r")
output_file = open("output.txt", "w")

# total number of lines


nodes = set()

next(input_file)
graph = {}


for line in input_file:
    node1, node2, relation = line.strip().split()

    nodes.add(node1)
    # nodes.add(node2)

    relation = int(relation)

    # Check if the first node already exists in the dictionary
    if node1 not in graph:
        # If not, create a new dictionary for the node
        graph[node1] = {}
    # Add the neighboring node and the relationship to the dictionary for node1
    graph[node1][node2] = relation

node_list = list(nodes)
node_list2 = list(nodes)


relation2id = {}

with open("SHS_String_relation.txt", "r") as file:
    for line in file:
        relation, relation_id = line.strip().split("\t")
        relation2id[int(relation_id)] = relation
# print(relation2id)

unique_rows = set()
size = 9  # the size of the connection

total = 90000

fieldnames = ['input_text', 'output_text']

with open("SHS148K_10.csv", mode="w", newline='') as tra:
  with open("posi_SHS148K_10.csv", mode="w", newline='') as pos:
    with open("neg_SHS148K_10.csv",  mode="w", newline='') as neg:

        writer_pos = csv.DictWriter(pos, fieldnames=fieldnames)
        writer_pos.writeheader()

        writer_neg = csv.DictWriter(neg, fieldnames=fieldnames)
        writer_neg.writeheader()

        writer_tra = csv.DictWriter(tra, fieldnames=fieldnames)
        writer_tra.writeheader()

        dfs(graph,size)


data = pd.read_csv('SHS148K_10.csv')
train_data, test_data = train_test_split(data, test_size=0.2)
train_data.to_csv('SHS148K_train_10.csv', index=False)
test_data.to_csv('SHS148K_test_10.csv', index=False)