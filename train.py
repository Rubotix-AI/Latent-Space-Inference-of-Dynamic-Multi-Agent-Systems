from utils import ae_optimizer, encoder, decoder, loss_fn
from graph_construct import graph_data
from models.latent_lstm import lstm_optimizer, lstm_model, X_lstm, Y_lstm

for epoch in range(50):
    total_loss = 0
    for data in graph_data:
        ae_optimizer.zero_grad()
        z = encoder(data.x, data.edge_index)
        x_hat = decoder(z)
        loss = loss_fn(x_hat, data.x)
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


for epoch in range(40):
    lstm_optimizer.zero_grad()
    output = lstm_model(X_lstm)
    loss = loss_fn(output, Y_lstm)
    loss.backward()
    lstm_optimizer.step()
    print(f"LSTM Epoch {epoch+1}, Loss: {loss.item():.4f}")
