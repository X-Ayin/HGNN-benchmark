from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SphinxDiscovererConfig:
    seq_len: int
    in_dim: int = 3
    pos_emb_dim: int = 16
    use_positional_embedding: bool = True
    hidden_dim: int = 64
    num_edges: int = 1
    edge_size: int = 4
    temperature: float = 1.0
    gumbel_noise: bool = True
    straight_through: bool = True
    sequential_penalty: float = 8.0
    estimator: str = "st"  # st | simple
    simple_num_samples: int = 4
    baseline_momentum: float = 0.95
    append_singletons: bool = False


class SphinxDiscoverer(nn.Module):
    """
    Learn task-useful hyperedges from node features.

    Output incidence keeps singleton hyperedges and appends discovered hyperedges.
    """

    def __init__(self, cfg: SphinxDiscovererConfig) -> None:
        super().__init__()
        if cfg.num_edges < 1:
            raise ValueError("num_edges must be >= 1")
        if cfg.edge_size < 1:
            raise ValueError("edge_size must be >= 1")
        if cfg.edge_size > cfg.seq_len:
            raise ValueError("edge_size must be <= seq_len")
        if cfg.estimator not in {"st", "simple"}:
            raise ValueError("estimator must be 'st' or 'simple'")
        if cfg.simple_num_samples < 1:
            raise ValueError("simple_num_samples must be >= 1")
        if cfg.pos_emb_dim < 0:
            raise ValueError("pos_emb_dim must be >= 0")

        self.cfg = cfg
        self.pos_embedding = (
            nn.Embedding(cfg.seq_len, cfg.pos_emb_dim) if cfg.use_positional_embedding and cfg.pos_emb_dim > 0 else None
        )
        node_input_dim = cfg.in_dim + (cfg.pos_emb_dim if self.pos_embedding is not None else 0)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.slots = nn.Parameter(torch.randn(cfg.num_edges, cfg.hidden_dim) * 0.02)
        self.register_buffer("running_baseline", torch.tensor(0.0))

    def _sample_gumbel(self, shape: torch.Size, device: torch.device, eps: float = 1e-10) -> torch.Tensor:
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u.clamp(min=eps, max=1.0 - eps)))

    def _hard_topk(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        # logits: [B, N]
        _, idx = torch.topk(logits, k=k, dim=-1)
        hard = torch.zeros_like(logits)
        hard.scatter_(dim=-1, index=idx, value=1.0)
        return hard

    def _pl_log_prob(self, logits: torch.Tensor, ordered_idx: torch.Tensor) -> torch.Tensor:
        # logits: [B, N], ordered_idx: [B, K]
        masked_logits = logits
        log_prob = torch.zeros(logits.size(0), device=logits.device, dtype=logits.dtype)
        for step in range(ordered_idx.size(1)):
            log_p = torch.log_softmax(masked_logits, dim=-1)
            chosen = ordered_idx[:, step]
            log_prob = log_prob + log_p.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
            masked_logits = masked_logits.clone()
            masked_logits.scatter_(1, chosen.unsqueeze(-1), float("-inf"))
        return log_prob

    def _sample_simple_topk(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        noisy = logits
        if self.cfg.gumbel_noise and self.training:
            noisy = logits + self._sample_gumbel(logits.shape, logits.device)

        _, ordered_idx = torch.topk(noisy, k=self.cfg.edge_size, dim=-1)
        hard = torch.zeros_like(logits)
        hard.scatter_(dim=-1, index=ordered_idx, value=1.0)
        log_prob = self._pl_log_prob(logits, ordered_idx)
        return {
            "hard": hard,
            "log_prob": log_prob,
        }

    def _topk_with_bridge(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        # logits: [B, N]
        noisy_logits = logits
        if self.cfg.gumbel_noise and self.training:
            noisy_logits = logits + self._sample_gumbel(logits.shape, logits.device)

        hard = self._hard_topk(noisy_logits, self.cfg.edge_size)
        soft = torch.softmax(logits / max(self.cfg.temperature, 1e-6), dim=-1) * float(self.cfg.edge_size)

        if self.cfg.straight_through:
            bridge = hard + (soft - soft.detach())
        else:
            bridge = soft if self.training else hard

        return {
            "hard": hard,
            "soft": soft,
            "bridge": bridge,
        }

    def get_running_baseline(self) -> torch.Tensor:
        return self.running_baseline

    def update_running_baseline(self, value: torch.Tensor) -> None:
        val = value.detach().to(self.running_baseline.device)
        self.running_baseline.mul_(self.cfg.baseline_momentum).add_(
            (1.0 - self.cfg.baseline_momentum) * val
        )

    def forward(self, x: torch.Tensor, position_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # x: [B, N, F]
        if x.dim() != 3:
            raise ValueError("x must have shape [B, N, F]")
        bsz, n_nodes, _ = x.shape
        if n_nodes != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {n_nodes}")

        node_input = x
        if self.pos_embedding is not None:
            if position_idx is None:
                position_idx = torch.arange(n_nodes, device=x.device).unsqueeze(0).expand(bsz, -1)
            elif position_idx.dim() == 1:
                position_idx = position_idx.unsqueeze(0).expand(bsz, -1)
            pos_emb = self.pos_embedding(position_idx)
            node_input = torch.cat([x, pos_emb], dim=-1)

        node_emb = self.node_encoder(node_input)  # [B, N, H]
        scores = torch.einsum("bnh,mh->bmn", node_emb, self.slots) / math.sqrt(self.cfg.hidden_dim)

        if self.cfg.estimator == "st":
            used_mask = torch.zeros((bsz, n_nodes), device=x.device)
            hard_members = []
            soft_members = []
            bridge_members = []
            adjusted_scores = []

            for edge_idx in range(self.cfg.num_edges):
                logits = scores[:, edge_idx, :]
                if edge_idx > 0:
                    logits = logits - self.cfg.sequential_penalty * used_mask

                topk = self._topk_with_bridge(logits)
                hard_members.append(topk["hard"])
                soft_members.append(topk["soft"])
                bridge_members.append(topk["bridge"])
                adjusted_scores.append(logits)

                used_mask = torch.clamp(used_mask + topk["hard"], max=1.0)

            hard_m = torch.stack(hard_members, dim=1)  # [B, M, N]
            soft_m = torch.stack(soft_members, dim=1)  # [B, M, N]
            bridge_m = torch.stack(bridge_members, dim=1)  # [B, M, N]
            adjusted = torch.stack(adjusted_scores, dim=1)  # [B, M, N]

            learned_incidence = bridge_m.transpose(1, 2)  # [B, N, M]
            incidence = learned_incidence

            return {
                "estimator": "st",
                "scores": scores,
                "adjusted_scores": adjusted,
                "hard_membership": hard_m,
                "soft_membership": soft_m,
                "bridge_membership": bridge_m,
                "incidence": incidence,
            }

        # SIMPLE-style path: sample perturbed Top-K structures and expose sample log-prob.
        num_samples = self.cfg.simple_num_samples if self.training else 1
        sample_hard_members = []
        sample_log_probs = []
        sample_adjusted = []

        for _ in range(num_samples):
            used_mask = torch.zeros((bsz, n_nodes), device=x.device)
            hard_members = []
            adjusted_scores = []
            edge_log_probs = []

            for edge_idx in range(self.cfg.num_edges):
                logits = scores[:, edge_idx, :]
                if edge_idx > 0:
                    logits = logits - self.cfg.sequential_penalty * used_mask

                sampled = self._sample_simple_topk(logits)
                hard_members.append(sampled["hard"])
                adjusted_scores.append(logits)
                edge_log_probs.append(sampled["log_prob"])
                used_mask = torch.clamp(used_mask + sampled["hard"], max=1.0)

            hard_m = torch.stack(hard_members, dim=1)  # [B, M, N]
            sample_hard_members.append(hard_m)
            sample_adjusted.append(torch.stack(adjusted_scores, dim=1))
            sample_log_probs.append(torch.stack(edge_log_probs, dim=1).sum(dim=1))  # [B]

        hard_samples = torch.stack(sample_hard_members, dim=0)  # [S, B, M, N]
        adjusted = torch.stack(sample_adjusted, dim=0).mean(dim=0)  # [B, M, N]
        sample_log_prob = torch.stack(sample_log_probs, dim=0)  # [S, B]
        hard_expectation = hard_samples.float().mean(dim=0)  # [B, M, N]

        incidence_samples = []
        for sample_idx in range(hard_samples.size(0)):
            learned_incidence = hard_samples[sample_idx].transpose(1, 2)  # [B, N, M]
            incidence_samples.append(learned_incidence)
        incidence_samples = torch.stack(incidence_samples, dim=0)  # [S, B, N, N+M]

        incidence = incidence_samples.mean(dim=0)
        soft_proxy = (
            torch.softmax(adjusted / max(self.cfg.temperature, 1e-6), dim=-1) * float(self.cfg.edge_size)
        )

        return {
            "estimator": "simple",
            "scores": scores,
            "adjusted_scores": adjusted,
            "hard_membership": hard_expectation,
            "soft_membership": soft_proxy,
            "bridge_membership": hard_expectation,
            "incidence": incidence,
            "incidence_samples": incidence_samples,
            "sample_log_prob": sample_log_prob,
        }


def discover_incidence(
    discoverer: SphinxDiscoverer,
    x: torch.Tensor,
    position_idx: Optional[torch.Tensor] = None,
    reduce: Optional[str] = "mean",
) -> Dict[str, torch.Tensor]:
    """Helper to derive incidence in [N, E] for legacy HNHN code path."""
    out = discoverer(x, position_idx=position_idx)
    incidence_bne = out["incidence"]

    if reduce == "mean":
        out["incidence_reduced"] = incidence_bne.mean(dim=0)
    elif reduce == "first":
        out["incidence_reduced"] = incidence_bne[0]
    elif reduce is None:
        out["incidence_reduced"] = incidence_bne
    else:
        raise ValueError(f"Unsupported reduce mode: {reduce}")

    if "incidence_samples" in out:
        # [S, B, N, M] -> [S, N, M]
        out["incidence_samples_reduced"] = out["incidence_samples"].mean(dim=1)

    return out
