import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch, Data

class MatsimGNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv1 = GCNConv(input_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.fc = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward_gnn(self, x, edge_index):
        """Pass data through GCN layers"""
        edge_index = edge_index.to(torch.int64)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        return x

    def forward(self, observations):
        data_list = []
        for i in range(len(observations['x'])):
            data_list.append(Data(
                x=observations['x'][i], 
                edge_index=observations['edge_index'][i]
            ))

        batch = Batch.from_data_list(data_list)

        # Forward pass through GNN
        x = self.forward_gnn(batch.x, batch.edge_index)

        # Pool to graph-level representation
        x = global_mean_pool(x, batch.batch)
