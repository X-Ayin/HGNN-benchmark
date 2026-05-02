from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _SeqSlotAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        iters: int = 2,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        nonlin: str = "sigmoid",
        deterministic: bool = True,
        num_total_slots: int = 1,
        long_history: bool = False,
    ) -> None:
        super().__init__()
        self.num_slots_per_graph = 1
        self.num_history_slots = num_total_slots if long_history else 1
        self.iters = iters
        self.eps = eps
        self.deterministic = deterministic
        self.scale = dim ** -0.5
        self.temperature = temperature
        self.nonlin = nonlin

        self.slots_mu = nn.Parameter(torch.randn(1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.norm_input = nn.LayerNorm(dim - self.num_history_slots)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def _step(self, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        dots = dots / max(self.temperature, 1e-6)

        if self.nonlin == "softmax":
            attn = F.softmax(dots, dim=-1) + self.eps
        else:
            attn = torch.sigmoid(dots) + self.eps

        updates = torch.einsum("bjd,bij->bid", v, attn)
        slots = updates + self.mlp(self.norm_pre_ff(updates))
        return slots, attn, dots

    def forward(
        self,
        inputs: torch.Tensor,
        prev_attn: torch.Tensor,
        num_slots: int,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs: [B*N, D-1], prev_attn: [B, H, N]
        device, dtype = inputs.device, inputs.dtype
        batch_size = num_slots // self.num_slots_per_graph

        mu = self.slots_mu.expand(num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(num_slots, -1)
        if self.deterministic:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            slots_init = mu + sigma * torch.normal(0, 1, size=self.slots_mu.shape, generator=g, device=device)
        else:
            slots_init = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

        slots = slots_init.reshape(-1, self.num_slots_per_graph, slots_init.shape[-1])
        inputs = self.norm_input(inputs)

        prev_attn = prev_attn.permute(0, 2, 1).reshape(-1, self.num_history_slots)
        inputs = torch.cat((inputs, prev_attn), dim=-1)
        inputs = inputs.reshape(batch_size, -1, inputs.shape[-1])
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(max(1, self.iters)):
            slots, attn, dots = self._step(slots, k, v)

        return slots, attn, dots


@dataclass
class SphinxNativeMCMConfig:
    seq_len: int
    in_dim: int = 3
    hidden_dim: int = 128
    slot_dim: int = 64
    num_edges: int = 3
    edge_size: int = 4
    num_iters: int = 2
    temperature: float = 1.0
    slot_nonlin: str = "sigmoid"
    long_history: bool = False
    deterministic: bool = True
    gumbel_noise: bool = True
    straight_through: bool = True


class SphinxNativeMCMDiscoverer(nn.Module):
    """SPHINX-style sequential slot predictor adapted to [B, N, F] MCM inputs."""

    def __init__(self, cfg: SphinxNativeMCMConfig) -> None:
        super().__init__()
        if cfg.num_edges < 1:
            raise ValueError("num_edges must be >= 1")
        if cfg.edge_size < 1 or cfg.edge_size > cfg.seq_len:
            raise ValueError("edge_size must be in [1, seq_len]")

        self.cfg = cfg
        self.encoder = _MLP(cfg.in_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2)
        history_slots = cfg.num_edges if cfg.long_history else 1
        input_proj_dim = cfg.slot_dim - history_slots
        if input_proj_dim <= 0:
            raise ValueError("slot_dim must be larger than history slot count")
        self.input_mlp2 = _MLP(cfg.hidden_dim, cfg.hidden_dim, input_proj_dim, num_layers=3)
        self.slot_process = _SeqSlotAttention(
            dim=cfg.slot_dim,
            iters=cfg.num_iters,
            hidden_dim=cfg.hidden_dim,
            temperature=cfg.temperature,
            nonlin=cfg.slot_nonlin,
            deterministic=cfg.deterministic,
            num_total_slots=cfg.num_edges,
            long_history=cfg.long_history,
        )

    def _sample_gumbel(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u.clamp(min=1e-10, max=1.0 - 1e-10)))

    def _sample_topk(self, dots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # dots: [B, N]
        logits = dots
        noisy_logits = logits
        if self.cfg.gumbel_noise and self.training:
            noisy_logits = logits + self._sample_gumbel(logits.shape, logits.device)

        idx = torch.topk(noisy_logits, k=self.cfg.edge_size, dim=-1).indices
        hard = torch.zeros_like(logits)
        hard.scatter_(1, idx, 1.0)

        soft = torch.sigmoid(logits / max(self.cfg.temperature, 1e-6))
        soft = soft * (float(self.cfg.edge_size) / soft.sum(dim=-1, keepdim=True).clamp(min=1e-6))
        bridge = hard + (soft - soft.detach()) if self.cfg.straight_through else soft
        return hard, soft, bridge

    def forward(self, x: torch.Tensor, position_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        del position_idx
        # x: [B, N, F]
        if x.dim() != 3:
            raise ValueError("x must have shape [B, N, F]")

        bsz, n_nodes, _ = x.shape
        if n_nodes != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {n_nodes}")

        node_embed = self.encoder(x.reshape(bsz * n_nodes, -1))
        node_embed = self.input_mlp2(node_embed)  # [B*N, slot_dim-1]

        if self.cfg.long_history:
            history_attn = torch.zeros((bsz, self.cfg.num_edges, n_nodes), device=x.device)
        else:
            history_attn = torch.zeros((bsz, 1, n_nodes), device=x.device)

        all_hard = []
        all_soft = []
        all_bridge = []
        all_dots = []

        for edge_idx in range(self.cfg.num_edges):
            _, attn, dots = self.slot_process(node_embed, history_attn, num_slots=bsz, seed=edge_idx)
            logits = dots[:, 0, :]
            hard, soft, bridge = self._sample_topk(logits)

            if self.cfg.long_history:
                history_attn[:, edge_idx, :] = hard.detach()
            else:
                history_attn[:, 0, :] = hard.detach()

            all_hard.append(hard)
            all_soft.append(soft)
            all_bridge.append(bridge)
            all_dots.append(logits)

        hard_m = torch.stack(all_hard, dim=1)  # [B, M, N]
        soft_m = torch.stack(all_soft, dim=1)
        bridge_m = torch.stack(all_bridge, dim=1)
        adjusted = torch.stack(all_dots, dim=1)

        incidence = bridge_m.transpose(1, 2)  # [B, N, M]
        return {
            "estimator": "st",
            "scores": adjusted,
            "adjusted_scores": adjusted,
            "hard_membership": hard_m,
            "soft_membership": soft_m,
            "bridge_membership": bridge_m,
            "incidence": incidence,
        }
