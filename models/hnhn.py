from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HNHNLayer(nn.Module):
    """A compact node-hyperedge-node message passing layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.node_to_edge = nn.Linear(in_dim, out_dim)
        self.edge_to_node = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], h: [N, E] or [B, N, E]
        x_proj = self.node_to_edge(x)  # [B, N, O]

        if h.dim() == 2:
            node_deg = h.sum(dim=1).clamp(min=1.0)  # [N]
            edge_deg = h.sum(dim=0).clamp(min=1.0)  # [E]

            # node -> edge
            edge_feat = torch.einsum("ne,bno->beo", h, x_proj)
            edge_feat = edge_feat / edge_deg.unsqueeze(0).unsqueeze(-1)

            # edge -> node
            node_feat = torch.einsum("ne,beo->bno", h, edge_feat)
            node_feat = node_feat / node_deg.unsqueeze(0).unsqueeze(-1)
        elif h.dim() == 3:
            node_deg = h.sum(dim=2).clamp(min=1.0)  # [B, N]
            edge_deg = h.sum(dim=1).clamp(min=1.0)  # [B, E]

            # node -> edge
            edge_feat = torch.einsum("bne,bno->beo", h, x_proj)
            edge_feat = edge_feat / edge_deg.unsqueeze(-1)

            # edge -> node
            node_feat = torch.einsum("bne,beo->bno", h, edge_feat)
            node_feat = node_feat / node_deg.unsqueeze(-1)
        else:
            raise ValueError("h must have shape [N, E] or [B, N, E]")

        out = self.edge_to_node(node_feat)
        return out


class MCMHNHN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 24,
        pos_emb_dim: int = 0,
        use_positional_embedding: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.use_positional_embedding = use_positional_embedding and pos_emb_dim > 0
        self.pos_embedding = nn.Embedding(seq_len, pos_emb_dim) if self.use_positional_embedding else None
        input_dim = in_dim + (pos_emb_dim if self.use_positional_embedding else 0)

        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(HNHNLayer(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        mask_idx: torch.Tensor,
        position_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = x
        if self.pos_embedding is not None:
            if position_idx is None:
                position_idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
            elif position_idx.dim() == 1:
                position_idx = position_idx.unsqueeze(0).expand(x.size(0), -1)
            pos_emb = self.pos_embedding(position_idx)
            out = torch.cat([out, pos_emb], dim=-1)

        for layer in self.layers:
            out = layer(out, h)
            out = F.relu(out)
            out = self.dropout(out)

        bsz = out.size(0)
        gather = out[torch.arange(bsz, device=out.device), mask_idx]
        logits = self.head(gather)
        return logits
