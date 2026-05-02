from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TDHNNDiscovererConfig:
    seq_len: int = 24
    in_dim: int = 3
    hidden_dim: int = 64
    num_edges: int = 3
    k_n: int = 4
    k_e: int = 3
    eps: float = 1e-8
    dynamic_edges: bool = False
    min_num_edges: int = 3
    low_bound: float = 0.9
    up_bound: float = 0.95


class TDHNNDiscoverer(nn.Module):
    """
    Batch-capable TDHNN-style hypergraph discoverer.

    Input:  x [B, N, F]
    Output: incidence [B, N, E], attn [B, E, N]
    """

    def __init__(self, cfg: TDHNNDiscovererConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_edges = cfg.num_edges

        f_dim = cfg.hidden_dim
        self.input_proj = nn.Linear(cfg.in_dim, f_dim)
        self.scale = f_dim ** -0.5

        self.edges_mu = nn.Parameter(torch.randn(1, f_dim) * 0.02)
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))

        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        self.edge_update = nn.Sequential(
            nn.Linear(f_dim * 2, f_dim),
            nn.ReLU(inplace=True),
            nn.Linear(f_dim, f_dim),
        )

        self.norm_input = nn.LayerNorm(f_dim)
        self.norm_edges = nn.LayerNorm(f_dim)

    @staticmethod
    def _mask_topk(attn: torch.Tensor, k: int) -> torch.Tensor:
        if k >= attn.size(-1):
            return attn
        idx = torch.topk(attn, k=k, dim=-1).indices
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, value=True)
        return attn * mask.to(attn.dtype)

    def _adjust_edges(self, h: torch.Tensor) -> None:
        if not self.cfg.dynamic_edges:
            return
        # h: [B, N, E]
        with torch.no_grad():
            occupancy = (h > 0).float().sum(dim=1)  # [B, E]
            non_empty = (occupancy > 0).float().mean(dim=1)  # [B]
            s_level = float(non_empty.mean().item())

            if s_level > self.cfg.up_bound:
                self.num_edges += 1
            elif s_level < self.cfg.low_bound:
                self.num_edges = max(self.cfg.min_num_edges, self.num_edges - 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,N,F], got shape={tuple(x.shape)}")
        bsz, n_nodes, _ = x.shape
        if n_nodes != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got N={n_nodes}")

        node_feat = self.norm_input(F.relu(self.input_proj(x)))  # [B,N,H]

        mu = self.edges_mu.expand(self.num_edges, -1)
        sigma = self.edges_logsigma.exp().expand(self.num_edges, -1)
        edge_feat = mu + sigma * torch.randn_like(mu)
        edge_feat = edge_feat.unsqueeze(0).expand(bsz, -1, -1).contiguous()  # [B,E,H]

        q_e = F.relu(self.to_q(self.norm_edges(edge_feat)))  # [B,E,H]
        k_n = F.relu(self.to_k(node_feat))  # [B,N,H]
        v_n = F.relu(self.to_v(node_feat))  # [B,N,H]

        # edge <- node
        logits_en = torch.einsum("beh,bnh->ben", q_e, k_n) * self.scale  # [B,E,N]
        attn_en = torch.softmax(logits_en, dim=-1) + self.cfg.eps
        attn_en = attn_en / attn_en.sum(dim=-1, keepdim=True).clamp(min=self.cfg.eps)
        attn_en = self._mask_topk(attn_en, self.cfg.k_n)

        updates = torch.einsum("ben,bnh->beh", attn_en, v_n)  # [B,E,H]
        edge_feat = self.edge_update(torch.cat([edge_feat, updates], dim=-1))

        # node -> edge
        q_n = F.relu(self.to_q(node_feat))  # [B,N,H]
        k_e = F.relu(self.to_k(edge_feat))  # [B,E,H]
        logits_ne = torch.einsum("bnh,beh->bne", q_n, k_e) * self.scale  # [B,N,E]
        attn_ne = torch.softmax(logits_ne, dim=-1)
        attn_ne = self._mask_topk(attn_ne, self.cfg.k_e)

        self._adjust_edges(attn_ne)

        return {
            "incidence": attn_ne,          # [B,N,E]
            "attn": attn_en,               # [B,E,N]
            "raw_scores": logits_ne,       # [B,N,E]
            "num_edges": torch.tensor(self.num_edges, device=x.device),
        }

