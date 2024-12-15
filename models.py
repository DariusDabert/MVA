import torch
import torch.nn as nn
import torch.nn.functional as F


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

        for i in range(self.n_layers):
            z = self.fc[i](z)

        lambda_ = self.fc_proj(z)

        lambda_ = torch.sigmoid(lambda_)
        
        return lambda_

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_layers):
        super(Encoder, self).__init__()    

        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(input_dim, hidden_dims[0]),  
                            nn.ReLU(),
                            ))

        for layer in range(n_layers-1):
            self.mlps.append(nn.Sequential(nn.Linear(hidden_dims[layer], hidden_dims[layer+1]),  
                            nn.ReLU(),
                            ))
        self.mlps.append(nn.Linear(hidden_dims[n_layers-1], latent_dim))

    def forward(self, x):

        for i in range(len(self.mlps)):
            x = self.mlps[i](x)
        
        return x


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, input_dim, n_layers_dec) 

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, x, distribution):
        x = x.to_dense()
        x_latent = self.encoder(x)
    
        mu = self.fc_mu(x_latent)
        logvar = self.fc_logvar(x_latent)
        
        z = self.reparameterize(mu, logvar)

        lambda_ = self.decoder(z)
        
        # reconstruction loss from a poisson law (To modify)
        recon = - distribution(lambda_).log_prob(x).sum()
        kld = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + kld

        return loss, recon, kld




# Variational Autoencoder
class GMVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, nb_classes):
        super(GMVariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.nb_classes = nb_classes
        self.encoder = Encoder(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)

        self.fc_pi = nn.Sequential(nn.Linear(latent_dim, nb_classes),
                                      nn.Softmax(dim=1))
        
        self.fc_mus = torch.nn.ModuleList()
        for i in range(nb_classes):
            self.fc_mus.append(nn.Linear(latent_dim,latent_dim))
        self.fc_logvars = torch.nn.ModuleList()
        for i in range(nb_classes):
            self.fc_logvars.append(nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU()))
            
        self.decoder = Decoder(latent_dim, hidden_dim_dec, input_dim, n_layers_dec) 

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, x, distribution):
        x = x.to_dense()
        x_latent = self.encoder(x)

        pi = self.fc_pi(x_latent)
        mus = [fc_mu(x_latent) for fc_mu in self.fc_mus]
        logvars = [fc_logvar(x_latent) for fc_logvar in self.fc_logvars]

        recon = 0
        kld = 0
        kld_pi = 0
        
        for i in range(len(mus)):
            mu = mus[i]
            logvar = logvars[i]
            z = self.reparameterize(mu, logvar)

            lambda_ = self.decoder(z)
            recon -= ((pi[:,i] @ distribution(lambda_).log_prob(x)).sum())
            kld -=  ( 0.5 * torch.sum(pi[:,i] @ (1 + logvar - mu.pow(2) - logvar.exp())))
            kld_pi -= (pi[:,i] * torch.log(self. nb_classes * pi[:,i])).sum()

        loss = recon + kld + kld_pi

        return loss, recon, kld + kld_pi
