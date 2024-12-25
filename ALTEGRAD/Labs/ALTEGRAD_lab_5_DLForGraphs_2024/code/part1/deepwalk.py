"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    walk = [node]  # Start the walk with the initial node

    for _ in range(walk_length - 1):  # Perform the walk of the specified length
        current_node = walk[-1]  # Get the current node
        neighbors = list(G.neighbors(current_node))  # Get the neighbors of the current node
        
        if neighbors:  # If there are neighbors, pick one at random
            next_node = neighbors[randint(0, len(neighbors) - 1)]
            walk.append(next_node)
        else:  # If no neighbors, end the walk early
            break
    
    walk = [str(node) for node in walk]
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    for node in G.nodes:
        for _ in range(num_walks):
            walk = random_walk(G, node, walk_length)
            walks.append(walk)

    permuted_walks = np.random.permutation(walks)

    return permuted_walks.tolist()


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
