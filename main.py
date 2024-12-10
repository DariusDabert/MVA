import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from random import shuffle
from scipy.io import loadmat

import torch
import torch.nn.functional as F

from dataset import GenomeDataset, pbmc_definition
from utils import sparse_mx_to_torch_sparse
from models import VariationalAutoEncoder
from training import Trainer
np.random.seed(13)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 5
batch_size = 32

hidden_dim_encoder = [8192, 2048, 1024]
n_layers_encoder = 3
latent_dim = 1024
hidden_dim_decoder = [1024, 2048, 8192]
n_layers_decoder = 3
input_feats = 32738

# Load dataset
G = GenomeDataset(pbmc_definition)
X = G.data
y = G.labels


X_torch = sparse_mx_to_torch_sparse(X).to(device)

# Initialize autoencoder
autoencoder = VariationalAutoEncoder(input_feats, hidden_dim_encoder, hidden_dim_decoder, latent_dim, n_layers_encoder, n_layers_decoder).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

trainer = Trainer(autoencoder, optimizer, device)

trainer.train(X_torch, y, epochs, batch_size)

# Save model
torch.save(autoencoder.state_dict(), 'autoencoder.pth')