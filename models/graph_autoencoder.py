import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
    def forward(self, x, edge_index):
        # Check if edge_index is empty
        if edge_index.numel() == 0:
            # Handle the case of an empty graph, e.g., return a tensor of zeros
            # with the same shape as the expected output but with the latent dimension
            batch_size = x.size(0)
            return torch.zeros(batch_size, self.conv2.out_channels, device=x.device)
        x = torch.relu(self.conv1(x, edge_index))
        # Check again before the second convolution
        if edge_index.numel() == 0:
             batch_size = x.size(0)
             return torch.zeros(batch_size, self.conv2.out_channels, device=x.device)
        return self.conv2(x, edge_index)


class GraphDecoder(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, 16)
        self.lin3 = nn.Linear(16, 16)
        self.lin4 = nn.Linear(16, 16)
        self.lin2 = nn.Linear(16, out_channels)
    def forward(self, z):
        return self.lin2(torch.relu(self.lin1(z)))