from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric normalization for dense adjacency.

    Supports:
    - [N, N]
    - [B, N, N]
    """
    if adj.dim() == 2:
        degree = adj.sum(dim=1).clamp(min=eps)
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)
    if adj.dim() == 3:
        degree = adj.sum(dim=2).clamp(min=eps)
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt.unsqueeze(2) * adj * inv_sqrt.unsqueeze(1)
    raise ValueError(f"adj must be [N,N] or [B,N,N], got {tuple(adj.shape)}")


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], norm_adj: [N, N] or [B, N, N]
        x = self.linear(x)
        if norm_adj.dim() == 2:
            return torch.einsum("ij,bjf->bif", norm_adj, x)
        if norm_adj.dim() == 3:
            return torch.einsum("bij,bjf->bif", norm_adj, x)
        raise ValueError(f"norm_adj must be [N,N] or [B,N,N], got {tuple(norm_adj.shape)}")


class MCMGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(GCNLayer(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,N,F], got {tuple(x.shape)}")
        if mask_idx.dim() != 1 or mask_idx.size(0) != x.size(0):
            raise ValueError(
                f"mask_idx must be [B] aligned with x batch, got mask_idx={tuple(mask_idx.shape)} x={tuple(x.shape)}"
            )
        if norm_adj.dim() == 2:
            if norm_adj.size(0) != x.size(1) or norm_adj.size(1) != x.size(1):
                raise ValueError(
                    f"norm_adj [N,N] must match x N={x.size(1)}, got {tuple(norm_adj.shape)}"
                )
        elif norm_adj.dim() == 3:
            if (
                norm_adj.size(0) != x.size(0)
                or norm_adj.size(1) != x.size(1)
                or norm_adj.size(2) != x.size(1)
            ):
                raise ValueError(
                    f"norm_adj [B,N,N] must align with x [B,N,F], got norm_adj={tuple(norm_adj.shape)} x={tuple(x.shape)}"
                )
        else:
            raise ValueError(f"norm_adj must be [N,N] or [B,N,N], got {tuple(norm_adj.shape)}")

        h = x
        for layer in self.layers:
            h = layer(h, norm_adj)
            h = F.relu(h)
            h = self.dropout(h)

        # Gather masked node embedding per sample.
        bsz = h.size(0)
        gather = h[torch.arange(bsz, device=h.device), mask_idx]
        logits = self.head(gather)
        return logits
