import torch.nn as nn
import torch.optim as optim

from models.graph_autoencoder import GraphDecoder, GraphEncoder
from models.latent_lstm import LatentLSTM

encoder = GraphEncoder()
decoder = GraphDecoder()
lstm_model = LatentLSTM()
ae_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()
