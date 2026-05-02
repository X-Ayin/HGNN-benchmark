"""End-to-end training: NRI encoder frontend + MCM GNN backend."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_mcm.data.parity_dataset import ParityConfig, ParitySequenceDataset
from gnn_mcm.models.simple_gnn import MCMGNN, normalize_adjacency

# Add NRI source path (folder name contains '-').
NRI_ROOT = Path(__file__).parent.parent / "NRI-master"
sys.path.insert(0, str(NRI_ROOT))
from modules import MLPEncoder  # type: ignore  # noqa: E402
from utils import gumbel_softmax, my_softmax  # type: ignore  # noqa: E402


def build_rel_matrices(num_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    off_diag = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
    send_idx, recv_idx = torch.where(off_diag > 0)
    num_edges = send_idx.numel()

    rel_send = torch.zeros(num_edges, num_nodes, dtype=torch.float32)
    rel_rec = torch.zeros(num_edges, num_nodes, dtype=torch.float32)
    rel_send[torch.arange(num_edges), send_idx] = 1.0
    rel_rec[torch.arange(num_edges), recv_idx] = 1.0
    return rel_rec.to(device), rel_send.to(device), send_idx.to(device), recv_idx.to(device)


def kl_categorical_uniform(prob: torch.Tensor, num_nodes: int, num_edge_types: int, eps: float = 1e-16) -> torch.Tensor:
    kl = prob * torch.log(prob + eps)
    return kl.sum() / (num_nodes * prob.size(0))


class NRIGNNEndToEnd(nn.Module):
    def __init__(
        self,
        seq_len: int,
        backend_in_dim: int = 3,
        frontend_posenc_dim: int = 2,
        edge_types: int = 2,
        encoder_hidden: int = 128,
        encoder_dropout: float = 0.0,
        gnn_hidden: int = 64,
        gnn_layers: int = 2,
        gnn_dropout: float = 0.1,
        tau: float = 0.5,
        hard: bool = False,
        symmetrize: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.edge_types = edge_types
        self.tau = tau
        self.hard = hard
        self.symmetrize = symmetrize
        self.backend_in_dim = backend_in_dim
        self.frontend_posenc_dim = frontend_posenc_dim
        self.frontend_in_dim = backend_in_dim - 1 + frontend_posenc_dim

        if backend_in_dim != 3:
            raise ValueError("Current MCM backend contract expects backend_in_dim=3.")
        if frontend_posenc_dim not in {1, 2}:
            raise ValueError("frontend_posenc_dim must be 1 or 2.")

        self.encoder = MLPEncoder(
            n_in=self.frontend_in_dim,  # T=1 => T*D == D
            n_hid=encoder_hidden,
            n_out=edge_types,
            do_prob=encoder_dropout,
            factor=True,
        )
        self.backend = MCMGNN(
            in_dim=backend_in_dim,
            hidden_dim=gnn_hidden,
            num_layers=gnn_layers,
            dropout=gnn_dropout,
        )

        self._rel_rec: torch.Tensor | None = None
        self._rel_send: torch.Tensor | None = None
        self._send_idx: torch.Tensor | None = None
        self._recv_idx: torch.Tensor | None = None

    def _ensure_rel(self, x: torch.Tensor) -> None:
        if self._rel_rec is not None:
            return
        rel_rec, rel_send, send_idx, recv_idx = build_rel_matrices(self.seq_len, x.device)
        self._rel_rec = rel_rec
        self._rel_send = rel_send
        self._send_idx = send_idx
        self._recv_idx = recv_idx

    def _build_frontend_features(self, x: torch.Tensor) -> torch.Tensor:
        # backend x is fixed as [value, is_masked, normalized_pos]
        value_mask = x[:, :, :2]
        pos = x[:, :, 2]
        if self.frontend_posenc_dim == 1:
            return torch.cat([value_mask, pos.unsqueeze(-1)], dim=-1)
        phase = 2 * torch.pi * pos
        pos2 = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
        return torch.cat([value_mask, pos2], dim=-1)

    def forward(self, x: torch.Tensor, mask_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,N,F], got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"x N must be {self.seq_len}, got {x.size(1)}")
        if x.size(2) != self.backend_in_dim:
            raise ValueError(f"x F must be {self.backend_in_dim}, got {x.size(2)}")

        self._ensure_rel(x)
        assert self._rel_rec is not None and self._rel_send is not None
        assert self._send_idx is not None and self._recv_idx is not None

        # NRI encoder expects [B, N, T, D]. We use T=1 static setting.
        x_frontend = self._build_frontend_features(x)
        x4d = x_frontend.unsqueeze(2)
        edge_logits = self.encoder(x4d, self._rel_rec, self._rel_send)  # [B, E, K]
        edge_sample = gumbel_softmax(edge_logits, tau=self.tau, hard=self.hard)
        edge_prob = my_softmax(edge_logits, -1)
        edge_weight = edge_sample[:, :, 1]

        bsz = x.size(0)
        adj = torch.zeros((bsz, self.seq_len, self.seq_len), device=x.device, dtype=x.dtype)
        adj[:, self._send_idx, self._recv_idx] = edge_weight
        if self.symmetrize:
            adj = 0.5 * (adj + adj.transpose(1, 2))
        eye = torch.eye(self.seq_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        adj = adj + eye
        norm_adj = normalize_adjacency(adj)

        logits = self.backend(x, norm_adj, mask_idx)
        return logits, edge_logits, edge_prob


def evaluate(model: NRIGNNEndToEnd, loader: DataLoader, criterion: nn.Module, device: torch.device, kl_weight: float) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0
    total_kl = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["node_features"].to(device)
            y = batch["target"].to(device)
            mask_idx = batch["mask_idx"].to(device)
            logits, _, edge_prob = model(x, mask_idx)
            cls_loss = criterion(logits, y)
            kl_loss = kl_categorical_uniform(edge_prob, num_nodes=x.size(1), num_edge_types=edge_prob.size(-1))
            loss = cls_loss + kl_weight * kl_loss
            total_loss += float(loss.item()) * y.size(0)
            total_kl += float(kl_loss.item()) * y.size(0)
            total_acc += int((logits.argmax(dim=1) == y).sum().item())
            total += y.size(0)
    return total_loss / total, total_acc / total, total_kl / total


def main() -> None:
    parser = argparse.ArgumentParser(description="NRI frontend + GNN backend end-to-end")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--encoder_hidden", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_dropout", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--hard", action="store_true", default=False)
    parser.add_argument("--no_symmetrize", action="store_true", default=False)
    parser.add_argument("--frontend_posenc_dim", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 24
    parity_groups = [[1, 5, 9, 13], [2, 7, 11, 19], [4, 8, 12, 16, 20]]
    backend_in_dim = 3
    frontend_in_dim = 2 + args.frontend_posenc_dim

    print(f"{'=' * 64}")
    print("NRI + GNN end-to-end")
    print(
        f"device={device} seed={args.seed} backend_feat_dim={backend_in_dim} "
        f"frontend_feat_dim={frontend_in_dim} frontend_pos_dim={args.frontend_posenc_dim}"
    )
    print(f"{'=' * 64}")

    config = ParityConfig(
        seq_len=seq_len,
        parity_groups=parity_groups,
        positional_encoding_dim=1,
    )
    train_ds = ParitySequenceDataset(config, num_samples=args.train_samples, seed=args.seed)
    val_ds = ParitySequenceDataset(config, num_samples=args.val_samples, seed=args.seed + 1)
    test_ds = ParitySequenceDataset(config, num_samples=args.test_samples, seed=args.seed + 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = NRIGNNEndToEnd(
        seq_len=seq_len,
        backend_in_dim=backend_in_dim,
        frontend_posenc_dim=args.frontend_posenc_dim,
        encoder_hidden=args.encoder_hidden,
        encoder_dropout=args.encoder_dropout,
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        gnn_dropout=args.gnn_dropout,
        tau=args.tau,
        hard=args.hard,
        symmetrize=not args.no_symmetrize,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_nri_gnn_seed{args.seed}"
    log_file = Path("gnn_mcm/logs") / f"{run_tag}.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0
        total = 0

        for batch in train_loader:
            x = batch["node_features"].to(device)
            y = batch["target"].to(device)
            mask_idx = batch["mask_idx"].to(device)

            optimizer.zero_grad()
            logits, _, edge_prob = model(x, mask_idx)
            cls_loss = criterion(logits, y)
            kl_loss = kl_categorical_uniform(edge_prob, num_nodes=x.size(1), num_edge_types=edge_prob.size(-1))
            loss = cls_loss + args.kl_weight * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total_acc += int((logits.argmax(dim=1) == y).sum().item())
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = total_acc / total
        val_loss, val_acc, val_kl = evaluate(model, val_loader, criterion, device, args.kl_weight)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        with log_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_kl": val_kl,
                    }
                )
                + "\n"
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_kl={val_kl:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_kl = evaluate(model, test_loader, criterion, device, args.kl_weight)
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f} Test Loss: {test_loss:.4f} Test KL: {test_kl:.4f}")

    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "lr": args.lr,
        "kl_weight": args.kl_weight,
        "tau": args.tau,
        "hard": args.hard,
        "symmetrize": not args.no_symmetrize,
        "backend_feature_dim": backend_in_dim,
        "frontend_feature_dim": frontend_in_dim,
        "frontend_posenc_dim": args.frontend_posenc_dim,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_kl": test_kl,
        "run_tag": run_tag,
        "log_file": str(log_file),
    }
    summary_file = Path("gnn_mcm/logs") / f"{run_tag}_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

