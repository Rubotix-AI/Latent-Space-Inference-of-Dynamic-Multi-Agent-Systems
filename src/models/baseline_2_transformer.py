"""
Baseline 2 - Transformer encoder (temporal + agent attention).

Two-stage attention:

1. **Temporal encoder** - a Transformer encoder attends over the ``W``
   timesteps of each agent independently and produces one embedding per agent
   (a shared encoder applied to every agent).

2. **Agent (interaction) encoder** - a second Transformer encoder attends
   *across agents* within the scene, using the padding mask so padded agents
   are ignored. This is where the model can discover who-interacts-with-whom;
   the attention matrix is a soft, learned interaction graph and is exposed via
   ``last_agent_attention`` for the attention-map visualisations.

Pooled over agents, then classified.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.common import ClassifierHead, masked_mean


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the time axis."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_temporal_layers: int = 2,
        n_agent_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=n_temporal_layers
        )

        self.agent_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.agent_encoder = nn.TransformerEncoder(
            self.agent_layer, num_layers=n_agent_layers
        )

        self.head = ClassifierHead(d_model, d_model, n_classes, dropout)

        # Populated on every forward pass for visualisation.
        self.last_agent_attention: torch.Tensor | None = None

    def encode(self, feats, pos, mask):
        B, W, N, F = feats.shape

        # ---- Stage 1: temporal attention, per agent ----
        x = feats.permute(0, 2, 1, 3).reshape(B * N, W, F)  # (B*N, W, F)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.temporal_encoder(x)                        # (B*N, W, d)
        agent_emb = x.mean(dim=1)                           # (B*N, d)
        agent_emb = agent_emb.reshape(B, N, self.d_model)   # (B, N, d)

        # ---- Stage 2: attention across agents ----
        key_padding_mask = ~mask                            # True == ignore
        agent_out = self.agent_encoder(
            agent_emb, src_key_padding_mask=key_padding_mask
        )                                                   # (B, N, d)

        # Record a soft interaction graph for visualisation (no grad needed).
        with torch.no_grad():
            self.last_agent_attention = self._agent_attention_matrix(
                agent_emb, key_padding_mask
            )

        latent = masked_mean(agent_out, mask)               # (B, d)
        return latent

    def _agent_attention_matrix(self, agent_emb, key_padding_mask):
        """Recompute the first agent-layer self-attention weights for viz."""
        attn = self.agent_layer.self_attn
        _, weights = attn(
            agent_emb,
            agent_emb,
            agent_emb,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return weights  # (B, N, N)

    def forward(self, feats, pos, mask):
        latent = self.encode(feats, pos, mask)
        logits = self.head(latent)
        return logits, latent
