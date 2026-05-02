from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EvolveHypergraphMCMConfig:
    seq_len: int
    in_dim: int = 3
    hidden_dim: int = 128
    num_edges: int = 3
    edge_size: int = 4
    temperature: float = 1.0
    gumbel_noise: bool = True
    straight_through: bool = True


class EvolveHypergraphMCMDiscoverer(nn.Module):
    """
    Static EvolveHypergraph-style discoverer for MCM.

    Output contract is SPHINX-compatible while incidence is always [B, N, E].
    """

    def __init__(self, cfg: EvolveHypergraphMCMConfig) -> None:
        super().__init__()
        if cfg.num_edges < 1:
            raise ValueError("num_edges must be >= 1")
        if cfg.edge_size < 1 or cfg.edge_size > cfg.seq_len:
            raise ValueError("edge_size must be in [1, seq_len]")
        self.cfg = cfg

        # Shared node encoder + node-to-edge membership head.
        self.node_encoder = _MLP(cfg.in_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2)
        self.membership_head = _MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.num_edges, num_layers=2)

    def _sample_gumbel(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u.clamp(min=1e-10, max=1.0 - 1e-10)))

    def _sample_topk(self, logits_bmn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # logits_bmn: [B, M, N]
        noisy_logits = logits_bmn
        if self.cfg.gumbel_noise and self.training:
            noisy_logits = logits_bmn + self._sample_gumbel(logits_bmn.shape, logits_bmn.device)

        idx = torch.topk(noisy_logits, k=self.cfg.edge_size, dim=-1).indices  # [B,M,k]
        hard = torch.zeros_like(logits_bmn)
        hard.scatter_(2, idx, 1.0)

        soft = torch.sigmoid(logits_bmn / max(self.cfg.temperature, 1e-6))
        soft = soft * (float(self.cfg.edge_size) / soft.sum(dim=-1, keepdim=True).clamp(min=1e-6))
        bridge = hard + (soft - soft.detach()) if self.cfg.straight_through else soft
        return hard, soft, bridge

    def forward(self, x: torch.Tensor, position_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        del position_idx
        if x.dim() != 3:
            raise ValueError(f"x must be [B, N, F], got {tuple(x.shape)}")
        bsz, n_nodes, _ = x.shape
        if n_nodes != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {n_nodes}")

        node_h = self.node_encoder(x.reshape(bsz * n_nodes, -1)).reshape(bsz, n_nodes, -1)  # [B,N,H]
        logits_bne = self.membership_head(node_h)  # [B,N,E]
        logits_bmn = logits_bne.transpose(1, 2).contiguous()  # [B,E,N]

        hard_m, soft_m, bridge_m = self._sample_topk(logits_bmn)
        incidence = bridge_m.transpose(1, 2).contiguous()  # [B,N,E]

        return {
            "estimator": "st",
            "scores": logits_bmn,
            "adjusted_scores": logits_bmn,
            "attn": bridge_m,  # [B,E,N], SPHINX-compatible key
            "hard_membership": hard_m,
            "soft_membership": soft_m,
            "bridge_membership": bridge_m,
            "incidence": incidence,
        }
