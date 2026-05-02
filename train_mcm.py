from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.parity_dataset import ParityConfig, ParitySequenceDataset
from models.simple_gnn import MCMGNN, normalize_adjacency


@dataclass
class TrainConfig:
    seq_len: int = 24
    parity_groups: List[List[int]] = None
    train_samples: int = 4000
    val_samples: int = 1000
    test_samples: int = 1000
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.parity_groups is None:
            self.parity_groups = [
                [1, 5, 9, 13],
                [2, 7, 11, 19],
                [4, 8, 12, 16, 20],
            ]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_weighted_adjacency(
    seq_len: int,
    parity_groups: List[List[int]],
    mode: str,
    seed: int,
) -> torch.Tensor:
    """
    Build weighted dense adjacency [N,N] for GNN input.

    All modes use floating weights (not binary hard edges):
    - self-loop weight = 1.0
    - relation weight = 0.8
    - wrong mode uses random float weights in [0.2, 0.8] on non-true edges
    """
    if mode not in {"none", "wrong", "one_correct", "all_correct"}:
        raise ValueError(f"Unsupported mode: {mode}")

    # Start from zero and keep self-loops.
    adj = torch.zeros((seq_len, seq_len), dtype=torch.float32)
    for i in range(seq_len):
        adj[i, i] = 1.0

    rel_weight = 0.8
    rng = torch.Generator().manual_seed(seed)

    def add_group_weighted_edges(group: List[int]) -> None:
        for i in group:
            for j in group:
                if i != j:
                    adj[i, j] = rel_weight

    if mode == "one_correct":
        add_group_weighted_edges(parity_groups[0])
    elif mode == "all_correct":
        for group in parity_groups:
            add_group_weighted_edges(group)
    elif mode == "wrong":
        # Match edge budget with all_correct (excluding self-loops).
        edge_budget = 0
        for group in parity_groups:
            edge_budget += len(group) * (len(group) - 1)

        true_edges = set()
        for group in parity_groups:
            for i in group:
                for j in group:
                    if i != j:
                        true_edges.add((i, j))

        candidates = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                if (i, j) in true_edges:
                    continue
                candidates.append((i, j))

        order = torch.randperm(len(candidates), generator=rng).tolist()
        for idx in order[:edge_budget]:
            i, j = candidates[idx]
            w = torch.rand(1, generator=rng).item() * 0.6 + 0.2
            adj[i, j] = float(w)

    # mode == none: self-loop only, still weighted float input.
    return normalize_adjacency(adj)


def evaluate(model: nn.Module, loader: DataLoader, norm_adj: torch.Tensor, device: str) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["node_features"].to(device)
            y = batch["target"].to(device)
            mask_idx = batch["mask_idx"].to(device)

            logits = model(x, norm_adj, mask_idx)
            loss = criterion(logits, y)

            total_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == y).sum().item())
            total_count += y.size(0)

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
    }


def run_one_experiment(cfg: TrainConfig, graph_mode: str) -> Dict[str, float]:
    set_seed(cfg.seed)

    train_ds = ParitySequenceDataset(
        ParityConfig(seq_len=cfg.seq_len, parity_groups=cfg.parity_groups),
        num_samples=cfg.train_samples,
        seed=cfg.seed,
    )
    val_ds = ParitySequenceDataset(
        ParityConfig(seq_len=cfg.seq_len, parity_groups=cfg.parity_groups),
        num_samples=cfg.val_samples,
        seed=cfg.seed + 1,
    )
    test_ds = ParitySequenceDataset(
        ParityConfig(seq_len=cfg.seq_len, parity_groups=cfg.parity_groups),
        num_samples=cfg.test_samples,
        seed=cfg.seed + 2,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Weighted adjacency input for all modes.
    norm_adj = build_weighted_adjacency(
        seq_len=cfg.seq_len,
        parity_groups=cfg.parity_groups,
        mode=graph_mode,
        seed=cfg.seed,
    ).to(cfg.device)

    model = MCMGNN(
        in_dim=3,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["node_features"].to(cfg.device)
            y = batch["target"].to(cfg.device)
            mask_idx = batch["mask_idx"].to(cfg.device)

            optimizer.zero_grad()
            logits = model(x, norm_adj, mask_idx)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, norm_adj, cfg.device)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"[{graph_mode}] epoch {epoch:02d} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, norm_adj, cfg.device)
    return {
        "mode": graph_mode,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "best_val_acc": best_val_acc,
    }


def main() -> None:
    cfg = TrainConfig()
    print("=" * 72)
    print("GNN MCM experiment on parity-constrained binary sequences")
    print("=" * 72)
    print(f"device={cfg.device}")
    print(f"seq_len={cfg.seq_len}")
    print(f"parity_groups={cfg.parity_groups}")
    print()

    # Four weighted-graph settings:
    # 1) none
    # 2) wrong
    # 3) one_correct
    # 4) all_correct
    modes = ["none", "wrong", "one_correct", "all_correct"]
    results = []

    for m in modes:
        print(f"\n--- Running mode: {m} ---")
        res = run_one_experiment(cfg, m)
        results.append(res)

    print("\n" + "=" * 72)
    print("Final comparison (higher acc is better)")
    print("=" * 72)
    for r in results:
        print(
            f"mode={r['mode']:<12} "
            f"test_acc={r['test_acc']:.4f} "
            f"test_loss={r['test_loss']:.4f} "
            f"best_val_acc={r['best_val_acc']:.4f}"
        )

    base = next(x for x in results if x["mode"] == "none")
    for r in results:
        if r["mode"] != "none":
            delta = r["test_acc"] - base["test_acc"]
            print(f"improvement vs none ({r['mode']}): {delta:+.4f}")


if __name__ == "__main__":
    main()
