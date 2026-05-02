from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_mcm.models.mcm_interface import validate_mcm_inputs


def _expand_incidence(h: torch.Tensor, batch_size: int) -> torch.Tensor:
    if h.dim() == 2:
        return h.unsqueeze(0).expand(batch_size, -1, -1)
    if h.dim() == 3:
        return h
    raise ValueError("h must have shape [N, E] or [B, N, E]")


class HCHALayer(nn.Module):
    """A compact dense HCHA-style layer with node-edge attention."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.node_proj = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(out_dim, out_dim)
        self.attn_proj = nn.Linear(out_dim * 2, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], h: [N, E] or [B, N, E]
        bsz, n_nodes, _ = x.shape
        h_bne = _expand_incidence(h, bsz)

        x_proj = self.node_proj(x)
        edge_deg = h_bne.sum(dim=1, keepdim=False).clamp(min=1.0)
        edge_feat = torch.einsum("bne,bnd->bed", h_bne, x_proj)
        edge_feat = edge_feat / edge_deg.unsqueeze(-1)
        edge_feat = self.edge_proj(edge_feat)

        edge_feat_expand = edge_feat.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        node_feat_expand = x_proj.unsqueeze(2).expand(-1, -1, edge_feat.size(1), -1)
        attn_logits = self.attn_proj(torch.cat([node_feat_expand, edge_feat_expand], dim=-1)).squeeze(-1)
        attn = torch.sigmoid(attn_logits) * h_bne

        node_deg = h_bne.sum(dim=2).clamp(min=1.0)
        node_update = torch.einsum("bne,bed->bnd", attn, edge_feat)
        node_update = node_update / node_deg.unsqueeze(-1)
        return node_update


class MCMHCHA(nn.Module):
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
            layers.append(HCHALayer(dims[i], dims[i + 1]))
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
        validate_mcm_inputs(x, h, mask_idx, expected_feat_dim=3)
        out = x
        if self.pos_embedding is not None:
            if position_idx is None:
                position_idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
            elif position_idx.dim() == 1:
                position_idx = position_idx.unsqueeze(0).expand(x.size(0), -1)
            out = torch.cat([out, self.pos_embedding(position_idx)], dim=-1)

        for layer in self.layers:
            out = F.relu(layer(out, h))
            out = self.dropout(out)

        gather = out[torch.arange(out.size(0), device=out.device), mask_idx]
        return self.head(gather)


class AllDeepSetsLayer(nn.Module):
    """Dense AllDeepSets-style node-hyperedge-node aggregation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        h_bne = _expand_incidence(h, bsz)

        edge_deg = h_bne.sum(dim=1).clamp(min=1.0)
        edge_raw = torch.einsum("bne,bnd->bed", h_bne, x) / edge_deg.unsqueeze(-1)
        edge_feat = self.edge_mlp(edge_raw)

        node_deg = h_bne.sum(dim=2).clamp(min=1.0)
        node_context = torch.einsum("bne,bed->bnd", h_bne, edge_feat) / node_deg.unsqueeze(-1)
        node_input = torch.cat([x, node_context], dim=-1)
        return self.node_mlp(node_input)


class MCMAllDeepSets(nn.Module):
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
            layers.append(AllDeepSetsLayer(dims[i], dims[i + 1]))
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
        validate_mcm_inputs(x, h, mask_idx, expected_feat_dim=3)
        out = x
        if self.pos_embedding is not None:
            if position_idx is None:
                position_idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
            elif position_idx.dim() == 1:
                position_idx = position_idx.unsqueeze(0).expand(x.size(0), -1)
            out = torch.cat([out, self.pos_embedding(position_idx)], dim=-1)

        for layer in self.layers:
            out = F.relu(layer(out, h))
            out = self.dropout(out)

        gather = out[torch.arange(out.size(0), device=out.device), mask_idx]
        return self.head(gather)
