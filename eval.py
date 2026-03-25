import torch
from models.latent_lstm import lstm_model, X_lstm, seq_len
from utils import decoder
from graph_construct import graph_data

# SECTION 5: Forecast Evaluation
with torch.no_grad():
    predicted_latents = lstm_model(X_lstm)
    decoded_predictions = decoder(predicted_latents)

gt = graph_data[seq_len].x.mean(dim=0)
pred = decoded_predictions.mean(dim=0)
error = torch.norm(gt - pred).item()
print(f"Reconstruction Error (forecasted): {error:.4f}")
