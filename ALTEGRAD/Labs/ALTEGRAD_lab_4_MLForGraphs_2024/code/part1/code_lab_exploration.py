"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
file_path = "ca-HepTh.txt"
G = nx.read_edgelist(
    file_path,
    delimiter='\t',        
    comments='#',          
    create_using=nx.Graph  
)

# Compute and print network characteristics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

############## Task 2
num_connected_components = nx.number_connected_components(G)
print(f"Number of connected components: {num_connected_components}")

if num_connected_components > 1:
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    largest_cc = G.subgraph(largest_cc_nodes)  

    largest_cc_num_nodes = largest_cc.number_of_nodes()
    largest_cc_num_edges = largest_cc.number_of_edges()

    fraction_nodes = largest_cc_num_nodes / G.number_of_nodes()
    fraction_edges = largest_cc_num_edges / G.number_of_edges()

    print(f"Largest connected component - Number of nodes: {largest_cc_num_nodes}")
    print(f"Largest connected component - Number of edges: {largest_cc_num_edges}")
    print(f"Fraction of total nodes in largest connected component: {fraction_nodes:.4f}")
    print(f"Fraction of total edges in largest connected component: {fraction_edges:.4f}")
else:
    print("The graph is connected.")



