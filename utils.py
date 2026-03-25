import torch
import torch.nn as nn
import torch.optim as optim

from models.graph_autoencoder import GraphDecoder, GraphEncoder
from models.latent_lstm import LatentLSTM
from graph_construct import graph_data

encoder = GraphEncoder()
decoder = GraphDecoder()
lstm_model = LatentLSTM()
ae_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

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