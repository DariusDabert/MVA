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

class Trainer():
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train(self, X, y, distribution, epochs, batch_size):
        # Slit into training and validation sets
        idx = np.random.permutation(len(X))
        train_idx = [int(i) for i in idx[:int(0.81*idx.size)]]
        val_idx = [int(i) for i in idx[int(0.81*idx.size):int(0.9*idx.size)]]
        test_idx = [int(i) for i in idx[int(0.9*idx.size):]]

        n_train = len(train_idx)
        n_val = len(val_idx)
        n_test = len(test_idx)


        # Train autoencoder
        best_val_loss = np.inf
        for epoch in range(1, epochs+1):
            self.model.train()
            train_loss_all = 0
            train_count = 0
            train_loss_all_recon = 0
            train_loss_all_kld = 0

            shuffle(train_idx)

            for i in tqdm(range(0, batch_size*10, batch_size)):
                x_batch = list()
                y_batch = list()
                for j in range(i, min(n_train, i+batch_size)):
                    x_batch.append(X[train_idx[j]])
                    y_batch.append(y[train_idx[j]])
                    train_count += 1
                
                x_batch = torch.stack(x_batch, dim=0)
                y_batch = np.vstack(y_batch)

                y_batch = torch.FloatTensor(y_batch).to(self.device)
                
                self.optimizer.zero_grad()
                loss, recon, kld  = self.model.loss_function(x_batch, distribution)
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

            for i in tqdm(range(0, 10 * batch_size, batch_size)):
                x_batch = list()
                y_batch = list()
                for j in range(i, min(n_val, i+batch_size)):
                    x_batch.append(X[val_idx[j]])
                    y_batch.append(y[val_idx[j]])
                    val_count += 1
                
                x_batch = torch.stack(x_batch, dim=0)
                y_batch = np.vstack(y_batch)

                y_batch = torch.FloatTensor(y_batch).to(self.device)

                loss, recon, kld  = self.model.loss_function(x_batch, distribution)
                val_loss_all_recon += recon.item()
                val_loss_all_kld += kld.item()
                val_loss_all += loss.item()

            if epoch % 1 == 0:
                print('Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(epoch, train_loss_all/train_count, train_loss_all_recon/train_count, train_loss_all_kld/train_count, val_loss_all/train_count, val_loss_all_recon/train_count, val_loss_all_kld/train_count))
                

