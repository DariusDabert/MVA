"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk

from sklearn.cluster import KMeans


# Loads the karate network
G = nx.read_weighted_edgelist('data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

nx.draw_networkx(G, node_color=y, with_labels=True)


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

############## Task 8
# Generates spectral embeddings

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

k = 2
clusters = spectral_clustering(G, k)

cluster_features = np.zeros((n, 1))
for i, node in enumerate(G.nodes()):
    cluster_features[i, :] = clusters[node]

X_train_clusters = cluster_features[idx_train, :]
X_test_clusters = cluster_features[idx_test, :]

clf_cluster = LogisticRegression()
clf_cluster.fit(X_train_clusters, y_train)
y_pred_clusters = clf_cluster.predict(X_test_clusters)
spectral_acc = accuracy_score(y_test, y_pred_clusters)

print(f"Spectral Clustering Features Accuracy: {spectral_acc:.4f}")
print(f"DeepWalk Features Accuracy: {acc:.4f}")

