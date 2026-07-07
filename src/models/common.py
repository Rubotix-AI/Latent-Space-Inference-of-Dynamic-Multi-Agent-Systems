"""
Shared building blocks for the interaction-law classifiers.

Every model here follows the same contract so the trainer can treat them
interchangeably::

    logits, latent = model(feats, pos, mask)

    feats  : (B, W, N, F)  normalised per-agent dynamics features
    pos    : (B, W, N, 2)  centred positions (only the GNN uses these)
    mask   : (B, N) bool   True for real agents, False for padding
    logits : (B, n_classes)
    latent : (B, D)        pooled scene embedding (for t-SNE / probing)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean of ``x`` over the agent axis, ignoring padded agents.

    x    : (B, N, D)
    mask : (B, N) bool
    -> (B, D)
    """
    m = mask.unsqueeze(-1).to(x.dtype)          # (B, N, 1)
    summed = (x * m).sum(dim=1)                 # (B, D)
    count = m.sum(dim=1).clamp_min(1.0)         # (B, 1)
    return summed / count


class ClassifierHead(nn.Module):
    """Small MLP mapping a pooled latent to class logits."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
