"""
Baseline 1 - Per-agent LSTM encoder.

The simplest sensible baseline. Each agent's dynamics sequence is encoded
independently by a shared LSTM; the final hidden states are pooled over agents
(permutation invariant, mask-aware) and classified.

This model has **no explicit interaction mechanism** - agents never see each
other. It therefore measures how much of the interaction law is recoverable
from single-agent motion statistics alone, and serves as the reference point
the interaction-aware models (Transformer, GNN) must beat.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common import ClassifierHead, masked_mean


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = ClassifierHead(hidden_dim, hidden_dim, n_classes, dropout)

    def encode(
        self,
        feats: torch.Tensor,   # (B, W, N, F)
        pos: torch.Tensor,     # (B, W, N, 2) - unused
        mask: torch.Tensor,    # (B, N)
    ) -> torch.Tensor:
        B, W, N, F = feats.shape

        # Fold agents into the batch: encode every agent sequence in parallel.
        x = feats.permute(0, 2, 1, 3).reshape(B * N, W, F)  # (B*N, W, F)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                                          # (B*N, H)
        h = h.reshape(B, N, self.hidden_dim)                # (B, N, H)

        # Pool over real agents only.
        latent = masked_mean(h, mask)                       # (B, H)
        return latent

    def forward(self, feats, pos, mask):
        latent = self.encode(feats, pos, mask)
        logits = self.head(latent)
        return logits, latent
