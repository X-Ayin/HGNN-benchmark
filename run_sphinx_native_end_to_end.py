"""End-to-end training: SPHINX native frontend + gnn_mcm backend."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from datetime import datetime

# Disable torch.compile to avoid triton dependency issues
import torch._dynamo
torch._dynamo.config.disable = True

from gnn_mcm.data.parity_dataset import ParityConfig, ParitySequenceDataset
from gnn_mcm.models.sphinx_native_wrapper import SPHINXNativeDiscoverer
from gnn_mcm.models.evolvehypergraph_mcm import (
    EvolveHypergraphMCMConfig,
    EvolveHypergraphMCMDiscoverer,
)
from gnn_mcm.models.hyper_backends import MCMHCHA, MCMAllDeepSets
from gnn_mcm.models.mcm_interface import validate_mcm_inputs


class SPHINXEndToEnd(nn.Module):
    """Frontend discoverer + gnn_mcm backend end-to-end model."""
    
    def __init__(
        self,
        backend_type: str = "hcha",
        num_slots: int = 3,
        slot_k: int = 4,
        seq_len: int = 24,
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pos_emb_dim: int = 0,
        use_positional_embedding: bool = False,
        selector: str = "simple",
        temperature: float = 1.0,
        frontend_type: str = "sphinx_native",
        omega_strategy: str = "zeros",
        temporal_len: int = 1,
        temporal_mode: str = "repeat",
        frontend_posenc_dim: int = 0,
        frontend_feature_encoding: str = "ternary_onehot",
        sequential: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.frontend_posenc_dim = frontend_posenc_dim
        self.frontend_feature_encoding = frontend_feature_encoding
        self.frontend_base_dim = self._frontend_base_dim()
        self.frontend_in_dim = self.frontend_base_dim + frontend_posenc_dim

        # Frontend discoverer (SPHINX native or static EvolveHypergraph).
        if frontend_type == "sphinx_native":
            self.discoverer = SPHINXNativeDiscoverer(
                num_slots=num_slots,
                dim=hidden_dim,
                num_iter=3,
                temperature=temperature,
                slot_k=slot_k,
                seq_len=seq_len,
                in_feat=self.frontend_in_dim,
                hidden_dim=hidden_dim,
                selector=selector,
                sequential=sequential,
                omega_strategy=omega_strategy,
                temporal_len=temporal_len,
                temporal_mode=temporal_mode,
            )
        elif frontend_type == "evolvehypergraph_static":
            self.discoverer = EvolveHypergraphMCMDiscoverer(
                EvolveHypergraphMCMConfig(
                    seq_len=seq_len,
                    in_dim=self.frontend_in_dim,
                    hidden_dim=hidden_dim,
                    num_edges=num_slots,
                    edge_size=slot_k,
                    temperature=temperature,
                    gumbel_noise=True,
                    straight_through=True,
                )
            )
        else:
            raise ValueError(
                f"Unknown frontend_type: {frontend_type}. "
                "Use 'sphinx_native' or 'evolvehypergraph_static'."
            )
        
        # gnn_mcm backend (verified in experiment 5)
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
        """
        Forward pass.
        
        Args:
            x: [B, N, 3] node features
            mask_idx: [B] masked position indices
            
        Returns:
            logits: [B, 2] classification logits
            discovery_info: dict with discovered structure info
        """
        # Freeze the contract: x[B,N,3], learned incidence[B,N,E], mask_idx[B].
        if x.dim() != 3 or x.size(-1) != self.in_dim:
            raise ValueError(f"x must be [B,N,{self.in_dim}], got shape={tuple(x.shape)}")
        if mask_idx.dim() != 1 or mask_idx.size(0) != x.size(0):
            raise ValueError(
                f"mask_idx must be [B] aligned with x batch, got mask_idx={tuple(mask_idx.shape)} x={tuple(x.shape)}"
            )

        # Build frontend-only features (can include extra positional channels).
        x_frontend = self._build_frontend_features(x)

        # Discover hypergraph structure
        discovery = self.discoverer(x_frontend)
        incidence = discovery["incidence"]  # [B, N, num_slots]
        validate_mcm_inputs(x, incidence, mask_idx, expected_feat_dim=self.in_dim)
        
        # Use discovered structure in backend
        logits = self.backend(x, incidence, mask_idx)
        
        return logits, discovery

    def _build_frontend_features(self, x: torch.Tensor) -> torch.Tensor:
        """Build frontend input feature tensor from configured encoding."""
        x_frontend = self._encode_frontend_features(x)
        if self.frontend_posenc_dim == 0:
            return x_frontend
        if self.frontend_posenc_dim != 2:
            raise ValueError("Only frontend_posenc_dim in {0,2} is supported.")

        n_nodes = x_frontend.size(1)
        device = x_frontend.device
        dtype = x_frontend.dtype
        pos = torch.arange(n_nodes, device=device, dtype=dtype) / max(1, n_nodes - 1)
        phase = 2 * torch.pi * pos
        pos_2d = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)  # [N,2]
        pos_2d = pos_2d.unsqueeze(0).expand(x_frontend.size(0), -1, -1)  # [B,N,2]
        return torch.cat([x_frontend, pos_2d], dim=-1)

    def _frontend_base_dim(self) -> int:
        if self.frontend_feature_encoding in {"legacy_raw", "ternary_onehot"}:
            return 3
        raise ValueError(
            "Unsupported frontend_feature_encoding. "
            "Use 'legacy_raw' or 'ternary_onehot'."
        )

    def _encode_frontend_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode frontend features from x=[value, is_masked, normalized_pos].

        - legacy_raw: pass through original 3 channels.
        - ternary_onehot: map to [is_plus, is_minus, is_mask], aligned with
          transformer_mcm's {-1, 0, +1} state semantics.
        """
        if self.frontend_feature_encoding == "legacy_raw":
            return x
        if self.frontend_feature_encoding != "ternary_onehot":
            raise ValueError(
                "Unsupported frontend_feature_encoding. "
                "Use 'legacy_raw' or 'ternary_onehot'."
            )

        value = x[:, :, 0]
        is_masked = x[:, :, 1] > 0.5
        is_plus = (~is_masked) & (value > 0.5)
        is_minus = (~is_masked) & (~is_plus)
        is_mask = is_masked
        return torch.stack(
            [is_plus.to(x.dtype), is_minus.to(x.dtype), is_mask.to(x.dtype)], dim=-1
        )


def train_epoch(model, loader, backend_optimizer, discoverer_optimizer, criterion, device):
    """Train for one epoch."""
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
        logits, discovery = model(x, mask_idx)
        loss = criterion(logits, target)
        loss.backward()
        backend_optimizer.step()
        discoverer_optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, return_structures=False):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_structures = [] if return_structures else None
    
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
            
            if return_structures:
                # Extract discovered structures from attn matrix.
                attn = discovery.get("attn")
                if attn is None:
                    # fallback: reconstruct [B, E, N] view from incidence [B, N, E]
                    attn = discovery["incidence"].transpose(1, 2).contiguous()
                for b in range(attn.size(0)):
                    sample_structures = []
                    for s in range(attn.size(1)):
                        # Get top-k nodes for this slot
                        slot_scores = attn[b, s, :]  # [N]
                        topk_values, topk_indices = torch.topk(slot_scores, k=min(6, attn.size(2)))
                        # Filter by threshold
                        mask = topk_values > 0.5
                        discovered_nodes = topk_indices[mask].cpu().tolist()
                        sample_structures.append({
                            "slot": s,
                            "nodes": discovered_nodes,
                            "scores": topk_values[mask].cpu().tolist()
                        })
                    all_structures.append(sample_structures)
    
    if return_structures:
        return total_loss / total, correct / total, all_structures
    return total_loss / total, correct / total


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPHINX end-to-end training")

    def str2bool(v: str) -> bool:
        return v.lower() in {"1", "true", "t", "yes", "y"}
    parser.add_argument("--backend", type=str, default="hcha", choices=["hcha", "all_deepsets"])
    parser.add_argument(
        "--frontend_type",
        type=str,
        default="sphinx_native",
        choices=["sphinx_native", "evolvehypergraph_static"],
        help="Structure discoverer frontend type",
    )
    parser.add_argument("--selector", type=str, default="simple", choices=["simple", "imle", "aimle"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ld", type=float, default=10.0,
                        help="Frontend discoverer LR divisor; discoverer_lr = lr / ld")
    parser.add_argument("--num_slots", type=int, default=3)
    parser.add_argument("--slot_k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--omega_strategy", type=str, default="zeros",
                        choices=["zeros", "positional", "sincos", "fourier", "learned"],
                        help="Encoding strategy for node position (omega parameter)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_emb_dim", type=int, default=0)
    parser.add_argument("--temporal_len", type=int, default=2)
    parser.add_argument("--temporal_mode", type=str, default="repeat", choices=["repeat"])
    parser.add_argument("--frontend_posenc_dim", type=int, default=2, choices=[0, 2],
                        help="Extra positional channels for frontend discoverer only")
    parser.add_argument(
        "--frontend_feature_encoding",
        type=str,
        default="ternary_onehot",
        choices=["legacy_raw", "ternary_onehot"],
        help="Frontend feature encoding mode before discoverer",
    )
    parser.add_argument("--sequential", type=str2bool, default=True,
                        help="Use SPHINX sequential discoverer (default: true)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Configuration
    seq_len = 24
    parity_groups = [[1, 5, 9, 13], [2, 7, 11, 19], [4, 8, 12, 16, 20]]
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"SPHINX End-to-End Training")
    print(
        f"Frontend: {args.frontend_type}, Backend: {args.backend}, "
        f"Selector: {args.selector}, Seed: {args.seed}"
    )
    print(
        f"Frontend feature encoding: {args.frontend_feature_encoding}, "
        f"frontend_posenc_dim={args.frontend_posenc_dim}, temporal_len={args.temporal_len}"
    )
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create datasets
    config = ParityConfig(seq_len=seq_len, parity_groups=parity_groups)
    train_dataset = ParitySequenceDataset(config, num_samples=args.train_samples, seed=args.seed)
    val_dataset = ParitySequenceDataset(config, num_samples=args.val_samples, seed=args.seed + 1)
    test_dataset = ParitySequenceDataset(config, num_samples=args.test_samples, seed=args.seed + 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SPHINXEndToEnd(
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
        selector=args.selector,
        temperature=args.temperature,
        frontend_type=args.frontend_type,
        omega_strategy=args.omega_strategy,
        temporal_len=args.temporal_len,
        temporal_mode=args.temporal_mode,
        frontend_posenc_dim=args.frontend_posenc_dim,
        frontend_feature_encoding=args.frontend_feature_encoding,
        sequential=args.sequential,
    ).to(device)
    
    backend_optimizer = torch.optim.Adam(model.backend.parameters(), lr=args.lr)
    discoverer_optimizer = torch.optim.Adam(model.discoverer.parameters(), lr=args.lr / args.ld)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Training loop
    run_tag = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_sphinx_e2e_"
        f"{args.frontend_type}_{args.backend}_{args.selector}_seed{args.seed}"
    )
    log_file = f"gnn_mcm/logs/{run_tag}.jsonl"
    
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, backend_optimizer, discoverer_optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    # Test with structure extraction
    test_loss, test_acc, test_structures = evaluate(model, test_loader, criterion, device, return_structures=True)
    
    print(f"\nFinal Results:")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    # Show sample discovered structures
    print(f"\nDiscovered Structures (first 3 samples):")
    print(f"True Parity Groups: {parity_groups}")
    for i in range(min(3, len(test_structures))):
        print(f"\nSample {i}:")
        for slot_info in test_structures[i]:
            if slot_info["nodes"]:
                print(f"  Slot {slot_info['slot']}: nodes {slot_info['nodes']} (scores: {[f'{s:.3f}' for s in slot_info['scores']]})") 
    print(f"\n{'='*60}\n")
    
    # Save summary with structures
    summary = {
        "backend": args.backend,
        "frontend_type": args.frontend_type,
        "selector": args.selector,
        "seed": args.seed,
        "epochs": args.epochs,
        "num_slots": args.num_slots,
        "slot_k": args.slot_k,
        "temperature": args.temperature,
        "ld": args.ld,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pos_emb_dim": args.pos_emb_dim,
        "temporal_len": args.temporal_len,
        "temporal_mode": args.temporal_mode,
        "frontend_posenc_dim": args.frontend_posenc_dim,
        "frontend_feature_encoding": args.frontend_feature_encoding,
        "sequential": args.sequential,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "run_tag": run_tag,
        "log_file": log_file,
        "sample_structures": test_structures[:10],  # Save first 10 samples
        "true_parity_groups": parity_groups,
    }
    
    summary_file = f"gnn_mcm/logs/{run_tag}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    main()
