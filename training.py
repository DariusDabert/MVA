
import numpy as np
from tqdm import tqdm

from random import shuffle

import torch

from dataset import GenomeDataset, pbmc_definition
from utils import sparse_mx_to_torch_sparse
from models import VariationalAutoEncoder

class Trainer():
    def __init__(self, X, idx, model, optimizer, model_name, device, seed = 42):
        self.X = X
        self.idx = idx
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name

    
    def train(self, distribution, epochs, batch_size, total_count):
        # Slit into training and validation sets

        train_idx = [int(i) for i in self.idx[:int(0.81*self.idx.size)]]
        val_idx = [int(i) for i in self.idx[int(0.81*self.idx.size):int(0.9*self.idx.size)]]

        n_train = len(train_idx)
        n_val = len(val_idx)


        # Train autoencoder
        best_val_loss = np.inf
        betas = np.linspace(0.1, 1, epochs)
        for epoch in range(1, epochs+1):
            self.model.train()
            train_loss_all = 0
            train_count = 0
            train_loss_all_recon = 0
            train_loss_all_kld = 0

            shuffle(train_idx)

            for i in tqdm(range(0, n_train, batch_size)):
                x_batch = list()
                for j in range(i, min(n_train, i+batch_size)):
                    x_batch.append(self.X[train_idx[j]])
                    train_count += 1
                
                x_batch = torch.stack(x_batch, dim=0)
                
                self.optimizer.zero_grad()
                loss, recon, kld  = self.model.loss_function(x_batch, distribution, betas[epoch], total_count, batch_size)
                train_loss_all_recon += recon.item()
                train_loss_all_kld += kld.item()
                loss.backward()
                train_loss_all += loss.item()
                self.optimizer.step()

            self.model.eval()
            val_loss_all = 0
            val_count = 0
            val_loss_all_recon = 0
            val_loss_all_kld = 0

            for i in tqdm(range(0, n_val, batch_size)):
                x_batch = list()

                for j in range(i, min(n_val, i+batch_size)):
                    x_batch.append(self.X[val_idx[j]])
                    val_count += 1
                
                x_batch = torch.stack(x_batch, dim=0)

                loss, recon, kld  = self.model.loss_function(x_batch, distribution, betas[epoch], total_count, batch_size)
                val_loss_all_recon += recon.item()
                val_loss_all_kld += kld.item()
                val_loss_all += loss.item()
            
            if best_val_loss > val_loss_all:
                best_val_loss = val_loss_all
                torch.save(self.model.state_dict(), f'{self.model_name}.pth')
                print('Model saved')

            if epoch % 1 == 0:
                print('Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(epoch, train_loss_all, train_loss_all_recon, train_loss_all_kld, val_loss_all, val_loss_all_recon, val_loss_all_kld))
                

