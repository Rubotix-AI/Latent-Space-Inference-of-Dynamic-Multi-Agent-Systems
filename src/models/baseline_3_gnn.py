"""
Baseline 3 - Graph Neural Network on a dynamic proximity graph.

This is the interaction-aware model with the strongest inductive bias for the
task: it is told *that* agents interact through space, and must learn *how*.

Pipeline:

1. **Node encoder** - a shared GRU encodes each agent's dynamics sequence into
   a node embedding ``h_i``.
2. **Proximity graph** - agents are connected to their ``k`` nearest
   neighbours (Euclidean distance on window-averaged, centred positions).
   Padded agents are excluded. The graph is made undirected, given self-loops,
   and symmetrically normalised. It is exposed via ``last_adjacency`` so it can
   be compared against the ground-truth interaction structure.
3. **Message passing** - a few graph-conv layers propagate information along
   the proximity graph.
4. **Readout** - masked mean pooling over nodes, then classification.

Implemented with dense adjacency matrices so the project has **no
torch_geometric dependency**; agent counts here are small (<=~30).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common import ClassifierHead, masked_mean


def build_knn_adjacency(
    pos: torch.Tensor,     # (B, W, N, 2)
    mask: torch.Tensor,    # (B, N)
    k: int,
) -> torch.Tensor:
    """
    Build a symmetric, self-looped, normalised kNN adjacency (B, N, N).

    Distances use positions averaged over the window. Padded agents can neither
    be chosen as neighbours nor have neighbours.
    """
    B, W, N, _ = pos.shape
    mean_pos = pos.mean(dim=1)                              # (B, N, 2)

    diff = mean_pos.unsqueeze(2) - mean_pos.unsqueeze(1)    # (B, N, N, 2)
    dist = torch.linalg.norm(diff, dim=-1)                 # (B, N, N)

    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)      # (B, N, N)
    eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
    selectable = valid_pair & ~eye

    # Push non-selectable entries out of range so top-k never picks them.
    big = dist.max().detach() + 1.0
    dist_masked = torch.where(selectable, dist, big + dist)

    kk = min(k, max(N - 1, 1))
    knn_idx = dist_masked.topk(kk, dim=-1, largest=False).indices  # (B, N, kk)

    adj = torch.zeros(B, N, N, device=pos.device)
    adj.scatter_(2, knn_idx, 1.0)
    adj = adj * selectable.float()                          # drop spurious edges

    # Undirected + self loops (only for real agents).
    adj = ((adj + adj.transpose(1, 2)) > 0).float()
    self_loops = torch.eye(N, device=pos.device).unsqueeze(0) * mask.unsqueeze(1).float()
    adj = adj + self_loops
    adj = (adj > 0).float()

    # Symmetric normalisation D^-1/2 A D^-1/2.
    deg = adj.sum(dim=-1).clamp_min(1.0)                    # (B, N)
    dinv = deg.pow(-0.5)
    adj_norm = adj * dinv.unsqueeze(1) * dinv.unsqueeze(2)
    return adj_norm


class GraphConv(nn.Module):
    """Dense graph-convolution layer: H' = act(A H W_n + H W_s)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        neigh = torch.bmm(adj, self.lin_neigh(h))
        out = self.act(neigh + self.lin_self(h))
        return self.dropout(out)


class GNNClassifier(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        k_neighbours: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbours = k_neighbours

        self.node_encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.convs = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.head = ClassifierHead(hidden_dim, hidden_dim, n_classes, dropout)

        self.last_adjacency: torch.Tensor | None = None

    def encode(self, feats, pos, mask):
        B, W, N, F = feats.shape

        # ---- Node encoder ----
        x = feats.permute(0, 2, 1, 3).reshape(B * N, W, F)   # (B*N, W, F)
        _, h_n = self.node_encoder(x)
        h = h_n[-1].reshape(B, N, self.hidden_dim)           # (B, N, H)

        # ---- Proximity graph ----
        adj = build_knn_adjacency(pos, mask, self.k_neighbours)
        self.last_adjacency = adj.detach()

        # ---- Message passing ----
        m = mask.unsqueeze(-1).float()
        for conv in self.convs:
            h = conv(h, adj) * m                             # zero padded nodes

        latent = masked_mean(h, mask)                        # (B, H)
        return latent

    def forward(self, feats, pos, mask):
        latent = self.encode(feats, pos, mask)
        logits = self.head(latent)
        return logits, latent
