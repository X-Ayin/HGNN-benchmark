"""End-to-end training: TDHNN-style frontend + gnn_mcm backend."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Disable torch.compile to avoid triton dependency issues
import torch._dynamo

from gnn_mcm.data.parity_dataset import ParityConfig, ParitySequenceDataset
from gnn_mcm.models.hyper_backends import MCMHCHA, MCMAllDeepSets
from gnn_mcm.models.mcm_interface import validate_mcm_inputs
from gnn_mcm.models.tdhnn_discoverer import TDHNNDiscoverer, TDHNNDiscovererConfig

torch._dynamo.config.disable = True


class TDHNNEndToEnd(nn.Module):
    """TDHNN frontend + gnn_mcm backend end-to-end model."""

    def __init__(
        self,
        backend_type: str = "all_deepsets",
        num_slots: int = 3,
        slot_k: int = 4,
        seq_len: int = 24,
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pos_emb_dim: int = 0,
        use_positional_embedding: bool = False,
        frontend_posenc_dim: int = 0,
        k_e: int = 3,
        dynamic_edges: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.frontend_posenc_dim = frontend_posenc_dim
        self.frontend_in_dim = in_dim + frontend_posenc_dim

        self.discoverer = TDHNNDiscoverer(
            TDHNNDiscovererConfig(
                seq_len=seq_len,
                in_dim=self.frontend_in_dim,
                hidden_dim=hidden_dim,
                num_edges=num_slots,
                k_n=max(1, slot_k),
                k_e=max(1, min(k_e, num_slots)),
                dynamic_edges=dynamic_edges,
                min_num_edges=max(1, num_slots),
            )
        )

        if backend_type == "hcha":
            self.backend = MCMHCHA(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                seq_len=seq_len,
                pos_emb_dim=pos_emb_dim,
                use_positional_embedding=use_positional_embedding,
            )
        elif backend_type == "all_deepsets":
            self.backend = MCMAllDeepSets(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                seq_len=seq_len,
                pos_emb_dim=pos_emb_dim,
                use_positional_embedding=use_positional_embedding,
            )
        else:
            raise ValueError(f"Unknown backend_type: {backend_type}")

    def forward(self, x: torch.Tensor, mask_idx: torch.Tensor):
        if x.dim() != 3 or x.size(-1) != self.in_dim:
            raise ValueError(f"x must be [B,N,{self.in_dim}], got shape={tuple(x.shape)}")
        if mask_idx.dim() != 1 or mask_idx.size(0) != x.size(0):
            raise ValueError(
                f"mask_idx must be [B] aligned with x batch, got mask_idx={tuple(mask_idx.shape)} x={tuple(x.shape)}"
            )

        x_frontend = self._build_frontend_features(x)
        discovery = self.discoverer(x_frontend)
        incidence = discovery["incidence"]  # [B, N, E]
        validate_mcm_inputs(x, incidence, mask_idx, expected_feat_dim=self.in_dim)

        logits = self.backend(x, incidence, mask_idx)
        return logits, discovery

    def _build_frontend_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.frontend_posenc_dim == 0:
            return x
        if self.frontend_posenc_dim != 2:
            raise ValueError("Only frontend_posenc_dim in {0,2} is supported.")

        n_nodes = x.size(1)
        device = x.device
        dtype = x.dtype
        pos = torch.arange(n_nodes, device=device, dtype=dtype) / max(1, n_nodes - 1)
        phase = 2 * torch.pi * pos
        pos_2d = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
        pos_2d = pos_2d.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([x, pos_2d], dim=-1)


def train_epoch(model, loader, backend_optimizer, discoverer_optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["node_features"].to(device)
        mask_idx = batch["mask_idx"].to(device)
        target = batch["target"].to(device)

        backend_optimizer.zero_grad()
        discoverer_optimizer.zero_grad()
        logits, _ = model(x, mask_idx)
        loss = criterion(logits, target)
        loss.backward()
        backend_optimizer.step()
        discoverer_optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, return_structures=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_structures = [] if return_structures else None
    # Structure health aggregates on incidence [B,N,E].
    total_incidence_nonzero = 0.0
    total_incidence_mean = 0.0
    total_slot_coverage = 0.0
    total_slot_overlap = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["node_features"].to(device)
            mask_idx = batch["mask_idx"].to(device)
            target = batch["target"].to(device)

            logits, discovery = model(x, mask_idx)
            loss = criterion(logits, target)

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == target).sum().item()
            total += x.size(0)

            incidence = discovery["incidence"]  # [B,N,E]
            attn = discovery["attn"]  # [B,E,N]
            total_incidence_nonzero += float((incidence > 0).float().mean().item())
            total_incidence_mean += float(incidence.mean().item())

            # Per-slot coverage: fraction of nodes in top-k with score > 0.
            k_cov = min(6, attn.size(2))
            topk_values, topk_indices = torch.topk(attn, k=k_cov, dim=2)  # [B,E,K]
            slot_coverage = (topk_values > 0).float().mean(dim=2)  # [B,E]
            total_slot_coverage += float(slot_coverage.mean().item())

            # Slot overlap: Jaccard among slots using top-k>0 masks.
            bsz, num_slots, _ = attn.shape
            overlap_acc = 0.0
            overlap_pairs = 0
            if num_slots >= 2:
                node_masks = torch.zeros((bsz, num_slots, attn.size(2)), device=attn.device)
                node_masks.scatter_(2, topk_indices, (topk_values > 0).float())
                for i in range(num_slots):
                    for j in range(i + 1, num_slots):
                        mi = node_masks[:, i, :]
                        mj = node_masks[:, j, :]
                        inter = (mi * mj).sum(dim=1)
                        union = ((mi + mj) > 0).float().sum(dim=1).clamp(min=1.0)
                        overlap_acc += float((inter / union).mean().item())
                        overlap_pairs += 1
            total_slot_overlap += overlap_acc / max(1, overlap_pairs)
            total_batches += 1

            if return_structures:
                for b in range(attn.size(0)):
                    sample_structures = []
                    for s in range(attn.size(1)):
                        slot_scores = attn[b, s, :]
                        topk_values, topk_indices = torch.topk(slot_scores, k=min(6, attn.size(2)))
                        mask = topk_values > 0.5
                        discovered_nodes = topk_indices[mask].cpu().tolist()
                        sample_structures.append(
                            {
                                "slot": s,
                                "nodes": discovered_nodes,
                                "scores": topk_values[mask].cpu().tolist(),
                            }
                        )
                    all_structures.append(sample_structures)

    metrics = {
        "incidence_nonzero_ratio": total_incidence_nonzero / max(1, total_batches),
        "incidence_mean": total_incidence_mean / max(1, total_batches),
        "slot_topk_coverage": total_slot_coverage / max(1, total_batches),
        "slot_overlap_jaccard": total_slot_overlap / max(1, total_batches),
    }

    if return_structures:
        return total_loss / total, correct / total, all_structures, metrics
    return total_loss / total, correct / total, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TDHNN end-to-end training")
    parser.add_argument("--backend", type=str, default="all_deepsets", choices=["hcha", "all_deepsets"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ld", type=float, default=10.0, help="discoverer_lr = lr / ld")
    parser.add_argument("--num_slots", type=int, default=3)
    parser.add_argument("--slot_k", type=int, default=4)
    parser.add_argument("--k_e", type=int, default=3)
    parser.add_argument("--dynamic_edges", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_emb_dim", type=int, default=0)
    parser.add_argument("--frontend_posenc_dim", type=int, default=0, choices=[0, 2])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    seq_len = 24
    parity_groups = [[1, 5, 9, 13], [2, 7, 11, 19], [4, 8, 12, 16, 20]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("TDHNN End-to-End Training")
    print(f"Backend: {args.backend}, Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    config = ParityConfig(seq_len=seq_len, parity_groups=parity_groups)
    train_dataset = ParitySequenceDataset(config, num_samples=args.train_samples, seed=args.seed)
    val_dataset = ParitySequenceDataset(config, num_samples=args.val_samples, seed=args.seed + 1)
    test_dataset = ParitySequenceDataset(config, num_samples=args.test_samples, seed=args.seed + 2)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TDHNNEndToEnd(
        backend_type=args.backend,
        num_slots=args.num_slots,
        slot_k=args.slot_k,
        seq_len=seq_len,
        in_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pos_emb_dim=args.pos_emb_dim,
        use_positional_embedding=args.pos_emb_dim > 0,
        frontend_posenc_dim=args.frontend_posenc_dim,
        k_e=args.k_e,
        dynamic_edges=args.dynamic_edges,
    ).to(device)

    backend_optimizer = torch.optim.Adam(model.backend.parameters(), lr=args.lr)
    discoverer_optimizer = torch.optim.Adam(model.discoverer.parameters(), lr=args.lr / args.ld)
    criterion = nn.CrossEntropyLoss()

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters\n")

    run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tdhnn_e2e_{args.backend}_seed{args.seed}"
    log_file = f"gnn_mcm/logs/{run_tag}.jsonl"

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, backend_optimizer, discoverer_optimizer, criterion, device
        )
        val_loss, val_acc, val_struct_metrics = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_incidence_nonzero_ratio": val_struct_metrics["incidence_nonzero_ratio"],
            "val_incidence_mean": val_struct_metrics["incidence_mean"],
            "val_slot_topk_coverage": val_struct_metrics["slot_topk_coverage"],
            "val_slot_overlap_jaccard": val_struct_metrics["slot_overlap_jaccard"],
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

    test_loss, test_acc, test_structures, test_struct_metrics = evaluate(
        model, test_loader, criterion, device, return_structures=True
    )

    print("\nFinal Results:")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    summary = {
        "frontend": "tdhnn",
        "backend": args.backend,
        "seed": args.seed,
        "epochs": args.epochs,
        "num_slots": args.num_slots,
        "slot_k": args.slot_k,
        "k_e": args.k_e,
        "dynamic_edges": args.dynamic_edges,
        "ld": args.ld,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pos_emb_dim": args.pos_emb_dim,
        "frontend_posenc_dim": args.frontend_posenc_dim,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_incidence_nonzero_ratio": test_struct_metrics["incidence_nonzero_ratio"],
        "test_incidence_mean": test_struct_metrics["incidence_mean"],
        "test_slot_topk_coverage": test_struct_metrics["slot_topk_coverage"],
        "test_slot_overlap_jaccard": test_struct_metrics["slot_overlap_jaccard"],
        "run_tag": run_tag,
        "log_file": log_file,
        "sample_structures": test_structures[:10],
        "true_parity_groups": parity_groups,
    }

    summary_file = f"gnn_mcm/logs/{run_tag}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {summary_file}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    main()

