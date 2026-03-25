import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

from simulation import history
from config import timesteps, num_boids

# SECTION 2: Graph Construction
def get_edge_index(positions, threshold=75):
    edges = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                edges.append((i, j))
                edges.append((j, i))
    if not edges:  # Return an empty tensor with shape (2, 0) if no edges
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t()

scaler = MinMaxScaler()
history_scaled = history.reshape(-1, 4)
history_scaled = scaler.fit_transform(history_scaled)
history_scaled = history_scaled.reshape(timesteps, num_boids, 4)

graph_data = []
for t in range(timesteps):
    x = torch.tensor(history_scaled[t], dtype=torch.float)
    edge_index = get_edge_index(history[t, :, :2])
    data = Data(x=x, edge_index=edge_index)
    graph_data.append(data)
    