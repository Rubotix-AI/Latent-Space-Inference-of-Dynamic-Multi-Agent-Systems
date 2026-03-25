import torch
import torch.nn as nn
import torch.optim as optim

from graph_construct import graph_data
from utils import encoder

# SECTION 4: LSTM in Latent Space
latent_seq = []
with torch.no_grad():
    for data in graph_data:
        z = encoder(data.x, data.edge_index).mean(dim=0)
        latent_seq.append(z)
latent_seq = torch.stack(latent_seq)

seq_len = 10
X_lstm, Y_lstm = [], []
for i in range(len(latent_seq) - seq_len):
    X_lstm.append(latent_seq[i:i+seq_len])
    Y_lstm.append(latent_seq[i+seq_len])
X_lstm, Y_lstm = torch.stack(X_lstm), torch.stack(Y_lstm)

class LatentLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

