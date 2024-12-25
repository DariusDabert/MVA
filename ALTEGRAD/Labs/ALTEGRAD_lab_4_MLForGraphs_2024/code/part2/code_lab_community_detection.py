"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
def spectral_clustering(G, k):
    A = nx.adjacency_matrix(G)  
    
    degrees = np.array(A.sum(axis=1)).flatten()  
    D_inv = diags(1.0 / degrees)  
    
    I = eye(A.shape[0]) 
    L_rw = I - D_inv @ A  
    
    eigvals, eigvecs = eigs(L_rw, k=2, which='SM')  
    eigvecs = eigvecs.real 
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(eigvecs)
    labels = kmeans.labels_  
    
    clustering = {node: cluster for node, cluster in zip(G.nodes(), labels)}
    
    return clustering



############## Task 4
file_path = "ca-HepTh.txt" 
G = nx.read_edgelist(file_path, delimiter='\t', comments='#', create_using=nx.Graph)

largest_cc_nodes = max(nx.connected_components(G), key=len)
largest_cc = G.subgraph(largest_cc_nodes)

k = 50 
clusters = spectral_clustering(largest_cc, k)


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    nc = max(clustering.values()) + 1 

    list_lc = [0] * nc  
    list_dc = [0] * nc  
    m = G.number_of_edges()

    for node in clustering:
        cluster = clustering[node]
        list_dc[cluster] += G.degree[node] 

    for edge in G.edges():
        u, v = edge
        if clustering[u] == clustering[v]:  
            list_lc[clustering[u]] += 1

    list_lc = [lc / 2 for lc in list_lc]

    modularity = 0
    for i in range(nc):
        modularity += (list_lc[i] / m) - (list_dc[i] / (2 * m)) ** 2

    return modularity


############## Task 6

##################
# your code here #
##################
def random_clustering(G, k):
    clustering = {node: randint(0, k - 1) for node in G.nodes()}
    return clustering

# Compute random clustering (k=50)
random_clusters = random_clustering(largest_cc, k)

modularity_spectral = modularity(largest_cc, clusters)
modularity_random = modularity(largest_cc, random_clusters)

print(f"Modularity (Spectral Clustering): {modularity_spectral}")
print(f"Modularity (Random Clustering): {modularity_random}")

