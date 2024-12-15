import numpy as np
import torch

from dataset import GenomeDataset, pbmc_definition
from utils import sparse_mx_to_torch_sparse
from models import VariationalAutoEncoder, GMVariationalAutoEncoder, GMVariationalAutoEncoder_transformers
from training import Trainer
np.random.seed(13)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 5
batch_size = 100

# Load dataset
G = GenomeDataset(pbmc_definition, download=True, small=True)
X = G.data
y = G.labels


X_torch = sparse_mx_to_torch_sparse(X).to(device)

# Initialize autoencoder
# for GMVariationalAutoEncoder
# hidden_dim_encoder = [1000, 1000]
# n_layers_encoder = 2
# latent_dim = 100
# hidden_dim_decoder = [1000, 1000]
# n_layers_decoder = 2
# input_feats = 32738
# nb_classes = 9
#autoencoder = GMVariationalAutoEncoder(input_feats, hidden_dim_encoder, hidden_dim_decoder, latent_dim, n_layers_encoder, n_layers_decoder, nb_classes).to(device)

# for GMVariationalAutoEncoder_transformers
n_layers_encoder = 2
latent_dim = 100
hidden_dim_encoder = 1000   
hidden_dim_decoder = 1000
n_layers_decoder = 2
input_feats = 32738
nb_classes = 9
autoencoder = GMVariationalAutoEncoder_transformers(input_feats, hidden_dim_encoder, hidden_dim_decoder, latent_dim, n_layers_encoder, n_layers_decoder, nb_classes).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
distribution = torch.distributions.Poisson 
idx = np.random.default_rng(seed=42).permutation(len(X_torch))

trainer = Trainer(X_torch, idx, autoencoder, optimizer, device)

trainer.train(distribution, epochs, batch_size)

# Save model
torch.save(autoencoder.state_dict(), 'autoencoder.pth')