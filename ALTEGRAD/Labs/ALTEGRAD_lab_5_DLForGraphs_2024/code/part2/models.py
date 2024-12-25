"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# class GNN(nn.Module):
#     """Simple GNN model"""
#     def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
#         super(GNN, self).__init__()

#         self.fc1 = nn.Linear(n_feat, n_hidden_1)
#         self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
#         self.fc3 = nn.Linear(n_hidden_2, n_class)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()

#     def forward(self, x_in, adj):
#         ############## Tasks 10 
#         x = self.fc1(x_in)  
#         x = torch.mm(adj, x)  
#         x = self.relu(x) 

#         x = self.dropout(x)

#         x = self.fc2(x)  
#         x = torch.mm(adj, x)  
#         x = self.relu(x)  

#         x = self.fc3(x)  

#         return F.log_softmax(x, dim=1)
    
class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 13
        x = self.fc1(x_in)  
        x = torch.mm(adj, x)  
        x = self.relu(x) 

        x = self.dropout(x)

        x = self.fc2(x)  
        x = torch.mm(adj, x)  
        x = self.relu(x)  

        o = self.fc3(x)  

        return F.log_softmax(o, dim=1), x
