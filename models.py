import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool


# TO MODIFY
class GaussianMixture(nn.Module):
    def __init__(self, n_components, n_features):
        super(GaussianMixture, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.pi = nn.Parameter(torch.ones(n_components)/n_components)
        self.mu = nn.Parameter(torch.randn(n_components, n_features))
        self.log_sigma = nn.Parameter(torch.zeros(n_components, n_features))
    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, self.n_components, -1)
        pi = self.pi.unsqueeze(0).expand(x.size(0), -1)
        mu = self.mu.unsqueeze(0).expand(x.size(0), -1, -1)
        log_sigma = self.log_sigma.unsqueeze(0).expand(x.size(0), -1, -1)
        log_prob = -0.5 * (torch.log(2 * np.pi) + 2 * log_sigma + (x - mu).pow(2) / log_sigma.exp())
        log_prob = log_prob.sum(-1) + torch.log(pi)
        return log_prob

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, n_layers, dropout=0.2):
        super(Decoder, self).__init__()
        self.n_layers = n_layers

        self.fc = nn.ModuleList()
        self.fc.append(nn.Sequential(nn.Linear(latent_dim, hidden_dims[0]),  
                            nn.ReLU(),
                            nn.LayerNorm(hidden_dims[0]), 
                            nn.Dropout(dropout)
                            ))

        for i in range(1, n_layers):
            self.fc.append(nn.Sequential(nn.Linear(hidden_dims[i-1], hidden_dims[i]),  
                            nn.ReLU(),
                            nn.LayerNorm(hidden_dims[i]), 
                            nn.Dropout(dropout)
                            ))
            
        
        self.fc_proj = nn.Linear(hidden_dims[n_layers-1], output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, z):
        z = self.fc[0](z)

        for i in range(1, self.n_layers):
            z = self.fc[i](z)

        lambda_ = self.fc_proj(z)

        lambda_ = torch.sigmoid(lambda_)
        
        return lambda_

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers):
        super(Encoder, self).__init__()    

        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(input_dim, hidden_dims[0]),  
                            nn.ReLU(),
                            ))

        for layer in range(n_layers-1):
            self.mlps.append(nn.Sequential(nn.Linear(hidden_dims[layer], hidden_dims[layer+1]),  
                            nn.ReLU(),
                            ))
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.mlps[0](x)
        for i in range(1, len(self.mlps)):
            x = self.mlps[i](x)
        
        return x


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, GMM=False):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc[-1], latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, input_dim, n_layers_dec) 

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu):
        out = self.decoder(mu)
        out = torch.sigmoid(out)
        out = out * (1 - torch.eye(out.size(-2), out.size(-1), device=out.device))
        return out

    def loss_function(self, x, beta=0.05):
        x = x.to_dense()
        x_latent = self.encoder(x)
    
        mu = self.fc_mu(x_latent)
        logvar = self.fc_logvar(x_latent)
        
        z = self.reparameterize(mu, logvar)

        lambda_ = self.decoder(z)
        
        # reconstruction loss from a poisson law (To modify)
        recon = - torch.distributions.Poisson(lambda_).log_prob(x).sum()
        kld = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + kld

        return loss, recon, kld
