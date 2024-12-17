import torch
import torch.nn as nn
import torch.nn.functional as F


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers, dropout=0.2):
        super(Decoder, self).__init__()
        self.n_layers = n_layers

        self.fc = nn.ModuleList()
        self.fc.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim),  
                            nn.ReLU(),
                            nn.LayerNorm(hidden_dim), 
                            nn.Dropout(dropout)
                            ))

        for i in range(1, n_layers):
            self.fc.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.ReLU(),
                            nn.LayerNorm(hidden_dim), 
                            nn.Dropout(dropout)
                            ))
            
        self.fc_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, z):

        for i in range(self.n_layers):
            z = self.fc[i](z)

        lambda_ = self.fc_proj(z)
        
        return lambda_

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers):
        super(Encoder, self).__init__()    

        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.ReLU(),
                            ))

        for layer in range(n_layers-1):
            self.mlps.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.ReLU(),
                            ))
        self.mlps.append(nn.Linear(hidden_dim, latent_dim))

    def forward(self, x):

        for i in range(len(self.mlps)):
            x = self.mlps[i](x)
        
        return x

# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, input_dim, n_layers_dec) 
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-6

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, x, distribution, beta, total_count=None):
        x = x.to_dense()
        x_latent = self.encoder(x)
    
        mu = self.fc_mu(x_latent)
        logvar = self.fc_logvar(x_latent)
        
        z = self.reparameterize(mu, logvar)

        lambda_ = self.decoder(z)

        if distribution == torch.distributions.Poisson:
            lambda_ = self.softplus(lambda_)
            lambda_ = torch.clamp(lambda_, self.eps, 1e6)
            recon = - distribution(lambda_).log_prob(x).sum()
        if distribution == torch.distributions.NegativeBinomial:
            lambda_ = self.sigmoid(lambda_)
            lambda_ = torch.clamp(lambda_, self.eps, 1 - self.eps)
            recon = - distribution(total_count= total_count, probs=lambda_).log_prob(x).sum()
        
        kld = 0.5 * torch.sum(self.latent_dim * logvar.exp().pow(2) + mu.pow(2) - self.latent_dim - 2 * self.latent_dim * logvar)
        loss = recon + beta*kld

        return loss, recon, kld

# Variational Autoencoder
class GMVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, nb_classes):
        super(GMVariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nb_classes = nb_classes
        self.encoder = Encoder(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)

        self.fc_pi = nn.Sequential(nn.Linear(latent_dim, nb_classes),
                                      nn.Softmax(dim=1))
        
        self.fc_mus = torch.nn.ModuleList()
        for i in range(nb_classes):
            self.fc_mus.append(nn.Linear(latent_dim,latent_dim))

        self.mus = nn.Parameter(torch.randn(nb_classes, latent_dim))


        self.fc_logvars = torch.nn.ModuleList()
        for i in range(nb_classes):
            self.fc_logvars.append(nn.Linear(latent_dim, latent_dim))
        
        # logvar for each class
        self.logvars = nn.Parameter(torch.randn(nb_classes, latent_dim))

            
        self.decoder = Decoder(latent_dim, hidden_dim_dec, input_dim, n_layers_dec) 
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-6

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, x, distribution, beta, total_count=None):
        x = x.to_dense()
        x_latent = self.encoder(x)

        pi = self.fc_pi(x_latent)
        pi = torch.clamp(pi, self.eps, 1-self.eps)
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
            
            if distribution == torch.distributions.Poisson:
                lambda_ = self.softplus(lambda_)
                lambda_ = torch.clamp(lambda_, self.eps, 1e6)
                recon -= ((pi[:,i] @ distribution(lambda_).log_prob(x)).sum())
            if distribution == torch.distributions.NegativeBinomial:
                lambda_ = self.sigmoid(lambda_)
                lambda_ = torch.clamp(lambda_, self.eps, 1 - self.eps)
                recon -= ((pi[:,i] @ distribution(total_count = total_count, probs=lambda_).log_prob(x)).sum())

            kld += 0.5 * torch.sum(pi[:,i] @ ( self.latent_dim * (logvar.exp()/ self.logvars[i]).pow(2) + (mu - self.mus[i]).pow(2)/self.logvars[i].pow(2) - self.latent_dim - 2 * self.latent_dim * logvar + 2 * self.latent_dim * self.logvars[i]))
            kld_pi += (pi[:,i] * torch.log(pi[:,i] * self.nb_classes)).sum()

        loss = recon + beta*(kld + kld_pi)

        return loss, recon, beta*(kld + kld_pi)
    

# Decoder with Transformers
class Decoder_transformers(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_heads, n_layers, dropout=0.2):
        super(Decoder_transformers, self).__init__()
        
        self.embedding = nn.Linear(latent_dim, hidden_dim)
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.fc_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        z = self.embedding(z).unsqueeze(0)  

        z = z + self.positional_encoding

        memory = torch.zeros_like(z)  

        z = self.transformer_decoder(tgt=z, memory=memory)

        z = z.squeeze(0)
        lambda_ = self.fc_proj(z)
        
        return lambda_
    

# Encoder with Transformers
class Encoder_transformers(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_heads, n_layers, dropout=0.2):
        super(Encoder_transformers, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)  

        x = x + self.positional_encoding

        x = self.transformer_encoder(x)

        x = x.squeeze(0)
        x = self.fc_proj(x)

        return x

# Variational Autoencoder
class GMVariationalAutoEncoder_transformers(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, nb_classes):
        super(GMVariationalAutoEncoder_transformers, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nb_classes = nb_classes
        self.encoder = Encoder_transformers(input_dim, hidden_dim_enc, latent_dim, 8, n_layers_enc)

        self.fc_pi = nn.Sequential(nn.Linear(latent_dim, nb_classes),
                                      nn.Softmax(dim=1))
        
        self.fc_mus = torch.nn.ModuleList()
        for i in range(nb_classes):
            self.fc_mus.append(nn.Linear(latent_dim,latent_dim))

        # mean for each class
        self.mus = nn.Parameter(torch.randn(nb_classes, latent_dim))


        self.fc_logvars = torch.nn.ModuleList()
        for i in range(nb_classes):
            self.fc_logvars.append(nn.Linear(latent_dim, latent_dim))
            
        self.logvars = nn.Parameter(torch.randn(nb_classes, latent_dim))
            
        self.decoder = Decoder_transformers(latent_dim, hidden_dim_dec, input_dim, 8, n_layers_dec) 
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-6

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    # def loss_function(self, x, distribution, beta, total_count=None):
    #     x = x.to_dense()
    #     x_latent = self.encoder(x)

    #     pi = self.fc_pi(x_latent)
    #     pi = torch.clamp(pi, self.eps, 1-self.eps)
    #     mus = [fc_mu(x_latent) for fc_mu in self.fc_mus]
    #     logvars = [fc_logvar(x_latent) for fc_logvar in self.fc_logvars]

    #     recon = 0
    #     kld = 0
    #     kld_pi = 0
    #     log_probs = []
        
    #     for i in range(len(mus)):
    #         mu = mus[i]
    #         logvar = logvars[i]
    #         z = self.reparameterize(mu, logvar)

    #         lambda_ = self.decoder(z)

    #         if distribution == torch.distributions.Poisson:
    #             lambda_ = self.softplus(lambda_)
    #             lambda_ = torch.clamp(lambda_, self.eps, 1e6)
    #             log_prob = distribution(lambda_).log_prob(x).sum(dim=1)  # Log-prob for current component
    #             log_probs.append(torch.log(pi[:, i] + self.eps) + log_prob)
    #             #recon -= ((pi[:,i] @ distribution(lambda_).log_prob(x)).sum())
    #         if distribution == torch.distributions.NegativeBinomial:
    #             lambda_ = self.sigmoid(lambda_)
    #             lambda_ = torch.clamp(lambda_, self.eps, 1-self.eps)
    #             log_prob = distribution(total_count=total_count, probs=lambda_).log_prob(x).sum(dim=1)  # Log-prob for current component
    #             log_probs.append(torch.log(pi[:, i] + self.eps) + log_prob)
    #             #recon -= ((pi[:,i] @ (distribution(total_count=total_count, probs=lambda_).log_prob(x))).sum())

            
    #         kld += 0.5 * torch.sum(pi[:,i] @ ( self.latent_dim * (logvar.exp()/ self.logvars[i]).pow(2) + (mu - self.mus[i]).pow(2)/self.logvars[i].pow(2) - self.latent_dim - 2 * self.latent_dim * logvar + 2 * self.latent_dim * self.logvars[i]))
    #     log_prob_sum = torch.logsumexp(torch.stack(log_probs, dim=1), dim=1)
    #     recon = -log_prob_sum.mean()
    #     kld_pi += (pi * torch.log(pi * self.nb_classes)).sum()

    #     loss = recon + beta*(kld + kld_pi)

    #     return loss, recon, beta*(kld + kld_pi)

def loss_function(self, x, distribution, beta, total_count=None):
    x = x.to_dense()
    x_latent = self.encoder(x)

    # Mixing coefficients q(y|x)
    pi = self.fc_pi(x_latent)  # Logits for q(y|x)
    pi = torch.clamp(pi, self.eps, 1 - self.eps)  # Avoid numerical instability

    # Parameters for q(z|x, y=k)
    mus_q = [fc_mu(x_latent) for fc_mu in self.fc_mus]
    logvars_q = [fc_logvar(x_latent) for fc_logvar in self.fc_logvars]

    # Parameters of the prior p(z|y=k)
    mus_p = self.mus  # Learnable means for p(z|y=k)
    logvars_p = self.logvars  # Learnable log variances for p(z|y=k)

    log_probs = []  # Store log p(x|z) for each component
    kld_latent_terms = []  # Store KL divergences for q(z|x,y=k) || p(z|y=k)

    # Prior p(y): uniform
    log_prior_y = torch.log(torch.tensor(1.0 / self.nb_classes, device=pi.device))

    for k in range(self.nb_classes):
        # Parameters for q(z|x, y=k)
        mu_q = mus_q[k]
        logvar_q = logvars_q[k]
        sigma_q = torch.exp(0.5 * logvar_q)  # Standard deviation

        # Parameters for p(z|y=k)
        mu_p = self.mus[k]
        logvar_p = self.logvars[k]
        sigma_p = torch.exp(0.5 * logvar_p)

        # Reparameterize: sample z ~ q(z|x,y=k)
        z = self.reparameterize(mu_q, logvar_q)

        # Reconstruction term: log p(x|z)
        lambda_ = self.decoder(z)
        if distribution == torch.distributions.Poisson:
            lambda_ = torch.clamp(self.softplus(lambda_), self.eps, 1e6)
            log_prob = distribution(lambda_).log_prob(x).sum(dim=1)
        elif distribution == torch.distributions.NegativeBinomial:
            lambda_ = torch.clamp(self.sigmoid(lambda_), self.eps, 1 - self.eps)
            log_prob = distribution(total_count=total_count, probs=lambda_).log_prob(x).sum(dim=1)
        else:
            raise NotImplementedError("Distribution not supported")

        # KL divergence: q(z|x,y=k) || p(z|y=k)
        kld_latent = 0.5 * torch.sum(
            logvar_p - logvar_q + (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / sigma_p.pow(2) - 1, dim=1
        )
        kld_latent_terms.append(kld_latent)

        # Log probability of this component
        log_probs.append(torch.log(pi[:, k] + self.eps) + log_prob - kld_latent)

    # Log-sum-exp for reconstruction term
    log_prob_sum = torch.logsumexp(torch.stack(log_probs, dim=1), dim=1)
    recon = -log_prob_sum.mean()

    # KL divergence for q(y|x) || p(y)
    kl_y = torch.sum(pi * torch.log(pi * self.nb_classes + self.eps), dim=1).mean()

    # Total loss
    loss = recon + beta * (kl_y)
    return loss, recon, beta * kl_y
