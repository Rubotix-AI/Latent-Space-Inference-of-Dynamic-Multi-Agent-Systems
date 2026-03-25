import torch.nn as nn

# SECTION 4: LSTM in Latent Space
class LatentLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])