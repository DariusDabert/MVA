"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    y = [data.y.item() for data in dataset]

    Gs = []
    for data in dataset:
        Gs.append(to_networkx(data, to_undirected=True))
    return Gs, y

Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
def compute_feature_map(G, graphlets, n_samples):
        """Compute the feature map for a single graph."""
        phi = np.zeros(4)  
        nodes = list(G.nodes())
        
        for _ in range(n_samples):
            if len(nodes) < 3:  # Skip graphs with fewer than 3 nodes
                continue
            sampled_nodes = np.random.choice(nodes, size=3, replace=False)
            subgraph = G.subgraph(sampled_nodes)
            
            # Check isomorphism with each graphlet
            for i, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    phi[i] += 1
                    break  # One subgraph matches only one graphlet
            
        return phi


def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0, 1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0, 1)
    graphlets[2].add_edge(1, 2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0, 1)
    graphlets[3].add_edge(1, 2)
    graphlets[3].add_edge(0, 2)

    phi_train = np.array([compute_feature_map(G, graphlets, n_samples) for G in Gs_train])
    phi_test = np.array([compute_feature_map(G, graphlets, n_samples) for G in Gs_test])

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9
K_train_graphlet, K_test_graphlet = graphlet_kernel(G_train, G_test)

############## Task 10

from sklearn.svm import SVC

clf_sp = SVC( kernel= 'precomputed' )
clf_sp.fit( K_train_sp , y_train )

y_pred_sp = clf_sp.predict(K_test_sp)

clf_graphlet = SVC( kernel= 'precomputed' )
clf_graphlet.fit( K_train_graphlet , y_train )

y_pred_graphlet = clf_graphlet.predict(K_test_graphlet)

acc_sp = accuracy_score(y_pred_sp, y_test)
acc_graphlet = accuracy_score(y_pred_graphlet, y_test)

print("Accuracy shortest path kernel:", acc_sp)
print("Accuracy graphlet kernel:", acc_graphlet)