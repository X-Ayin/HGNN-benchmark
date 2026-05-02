from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def _gaussian_elimination_gf2(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, List[int], List[int]]:
    """Solve A x = b over GF(2), return one solution and pivot/free columns."""
    a = a.clone().to(torch.int64)
    b = b.clone().to(torch.int64)

    rows, cols = a.shape
    pivot_cols: List[int] = []
    row = 0

    for col in range(cols):
        pivot = None
        for r in range(row, rows):
            if int(a[r, col].item()) == 1:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            a[[row, pivot]] = a[[pivot, row]]
            b[row], b[pivot] = b[pivot].clone(), b[row].clone()

        for r in range(rows):
            if r != row and int(a[r, col].item()) == 1:
                a[r] = (a[r] ^ a[row])
                b[r] = (b[r] ^ b[row])

        pivot_cols.append(col)
        row += 1
        if row == rows:
            break

    for r in range(rows):
        if int(a[r].sum().item()) == 0 and int(b[r].item()) == 1:
            raise ValueError("Parity constraints are inconsistent (no solution).")

    free_cols = [c for c in range(cols) if c not in pivot_cols]

    x = torch.zeros(cols, dtype=torch.int64)
    for c in free_cols:
        x[c] = random.randint(0, 1)

    for r in range(len(pivot_cols) - 1, -1, -1):
        pcol = pivot_cols[r]
        rhs = int(b[r].item())
        for c in range(pcol + 1, cols):
            rhs ^= int(a[r, c].item()) & int(x[c].item())
        x[pcol] = rhs

    return x, pivot_cols, free_cols


@dataclass
class ParityConfig:
    seq_len: int
    parity_groups: List[List[int]]
    mask_token_value: float = 0.0
    positional_encoding_dim: int = 1

    def validate(self) -> None:
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if not self.parity_groups:
            raise ValueError("parity_groups must not be empty")
        for group in self.parity_groups:
            if len(group) < 2:
                raise ValueError("Each parity group must contain at least 2 positions")
            if len(set(group)) != len(group):
                raise ValueError(f"Duplicated indices in parity group: {group}")
            for idx in group:
                if idx < 0 or idx >= self.seq_len:
                    raise ValueError(f"Parity index out of range: {idx}")
        if self.positional_encoding_dim not in {1, 2}:
            raise ValueError("positional_encoding_dim must be 1 or 2")


class ParitySequenceDataset(Dataset):
    """
    Generates binary sequences that satisfy parity constraints:
    for each group G, sum(x[i] for i in G) mod 2 == 0.

    MCM target: randomly mask one position and predict the masked bit.
    """

    def __init__(self, config: ParityConfig, num_samples: int, seed: int = 42) -> None:
        super().__init__()
        self.config = config
        self.config.validate()
        self.num_samples = num_samples
        random.seed(seed)
        torch.manual_seed(seed)

        a = torch.zeros((len(config.parity_groups), config.seq_len), dtype=torch.int64)
        for r, group in enumerate(config.parity_groups):
            for c in group:
                a[r, c] = 1

        b = torch.zeros(len(config.parity_groups), dtype=torch.int64)
        self.constraint_a = a
        self.constraint_b = b

        self._samples = [self._generate_one() for _ in range(self.num_samples)]

    def _generate_one(self) -> Tuple[torch.Tensor, int]:
        bits, _, _ = _gaussian_elimination_gf2(self.constraint_a, self.constraint_b)
        seq = bits.to(torch.float32)
        mask_idx = random.randrange(0, self.config.seq_len)
        return seq, mask_idx

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        seq, mask_idx = self._samples[idx]

        target = int(seq[mask_idx].item())
        position_idx = torch.arange(self.config.seq_len, dtype=torch.long)

        # Node feature:
        # - 1D pos: [value_or_mask_token, is_masked, normalized_position]
        # - 2D pos: [value_or_mask_token, is_masked, pos_sin, pos_cos]
        feat_dim = 2 + self.config.positional_encoding_dim
        features = torch.zeros((self.config.seq_len, feat_dim), dtype=torch.float32)
        features[:, 0] = seq
        features[:, 1] = 0.0
        pos = torch.arange(self.config.seq_len, dtype=torch.float32) / max(1, (self.config.seq_len - 1))
        if self.config.positional_encoding_dim == 1:
            features[:, 2] = pos
        else:
            phase = 2 * torch.pi * pos
            features[:, 2] = torch.sin(phase)
            features[:, 3] = torch.cos(phase)

        features[mask_idx, 0] = self.config.mask_token_value
        features[mask_idx, 1] = 1.0

        return {
            "node_features": features,
            "position_idx": position_idx,
            "target": torch.tensor(target, dtype=torch.long),
            "mask_idx": torch.tensor(mask_idx, dtype=torch.long),
            "raw_seq": seq,
        }


def build_adjacency(
    seq_len: int,
    parity_groups: Sequence[Sequence[int]],
    mode: str,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build dense adjacency matrix.

    mode:
    - none: self-loop only
    - wrong: random edges, no true parity edges
    - one_correct: include edges from one true parity group only
    - full: include all true parity-group clique edges
    """
    if mode not in {"none", "wrong", "one_correct", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")

    g = random.Random(seed)
    adj = torch.zeros((seq_len, seq_len), dtype=torch.float32)

    # Always keep self-loops.
    for i in range(seq_len):
        adj[i, i] = 1.0

    def add_group_edges(group: Sequence[int]) -> None:
        for i in group:
            for j in group:
                adj[i, j] = 1.0

    if mode == "full":
        for group in parity_groups:
            add_group_edges(group)

    elif mode == "one_correct":
        add_group_edges(parity_groups[0])

    elif mode == "wrong":
        # Keep same rough number of edges as full mode for fair comparison.
        edge_budget = 0
        for group in parity_groups:
            edge_budget += len(group) * len(group)
        edge_budget = max(0, edge_budget - seq_len)  # minus self loops already present

        candidates = [(i, j) for i in range(seq_len) for j in range(seq_len) if i != j]
        g.shuffle(candidates)

        # Exclude true parity edges so "wrong" really means wrong structure.
        true_edges = set()
        for group in parity_groups:
            for i in group:
                for j in group:
                    if i != j:
                        true_edges.add((i, j))

        used = 0
        for e in candidates:
            if e in true_edges:
                continue
            adj[e[0], e[1]] = 1.0
            used += 1
            if used >= edge_budget:
                break

    # mode == "none": self-loop only.

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = adj.sum(dim=1).clamp(min=1e-6)
    inv_sqrt = torch.pow(degree, -0.5)
    norm_adj = inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)
    return norm_adj


def build_incidence(
    seq_len: int,
    parity_groups: Sequence[Sequence[int]],
    mode: str,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build dense node-hyperedge incidence matrix H with shape [N, E].

    mode:
    - none: singleton hyperedges only (no cross-position relation)
    - wrong: one random hyperedge (same size as first parity group), excluding true groups
    - one_correct: one true hyperedge from parity_groups[0]
    - all_correct: all true hyperedges from parity_groups
    """
    if mode not in {"none", "wrong", "one_correct", "all_correct"}:
        raise ValueError(f"Unsupported mode for incidence: {mode}")

    g = random.Random(seed)
    hyperedges: List[List[int]] = []

    # Singleton hyperedges ensure every node has at least one incident hyperedge.
    for i in range(seq_len):
        hyperedges.append([i])

    if mode == "one_correct":
        hyperedges.append(list(parity_groups[0]))

    elif mode == "all_correct":
        for group in parity_groups:
            hyperedges.append(list(group))

    elif mode == "wrong":
        target_size = len(parity_groups[0])
        true_sets = {tuple(sorted(group)) for group in parity_groups}
        all_nodes = list(range(seq_len))

        for _ in range(5000):
            cand = sorted(g.sample(all_nodes, k=target_size))
            if tuple(cand) not in true_sets:
                hyperedges.append(cand)
                break
        else:
            raise RuntimeError("Failed to sample a wrong hyperedge distinct from true groups")

    # mode == "none": singleton-only hypergraph.

    h = torch.zeros((seq_len, len(hyperedges)), dtype=torch.float32)
    for e_idx, group in enumerate(hyperedges):
        for v in group:
            h[v, e_idx] = 1.0
    return h
