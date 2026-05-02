"""Test script to verify SPHINX native backend works correctly."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gnn_mcm.data.parity_dataset import ParityConfig, ParitySequenceDataset
from gnn_mcm.models.sphinx_native_mcm_backend import SPHINXMCMBackend


def create_incidence_matrix(mode: str, seq_len: int = 24) -> torch.Tensor:
    """Create incidence matrix for different modes."""
    parity_groups = [[1, 5, 9, 13], [2, 7, 11, 19], [4, 8, 12, 16, 20]]
    
    if mode == "none":
        # Only singleton edges
        h = torch.eye(seq_len)
    elif mode == "wrong":
        # Singleton + 1 wrong edge
        h = torch.eye(seq_len)
        wrong_edge = torch.zeros(seq_len, 1)
        wrong_edge[[0, 3, 6, 10]] = 1.0  # Wrong group
        h = torch.cat([h, wrong_edge], dim=1)
    elif mode == "one_correct":
        # Singleton + 1 correct edge
        h = torch.eye(seq_len)
        correct_edge = torch.zeros(seq_len, 1)
        correct_edge[parity_groups[0]] = 1.0
        h = torch.cat([h, correct_edge], dim=1)
    elif mode == "all_correct":
        # Singleton + all 3 correct edges
        h = torch.eye(seq_len)
        for group in parity_groups:
            edge = torch.zeros(seq_len, 1)
            edge[group] = 1.0
            h = torch.cat([h, edge], dim=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return h


def test_backend(model_type: str = "hcha", mode: str = "all_correct", omega_strategy: str = "zeros"):
    """Test SPHINX native backend on Parity MCM task."""
    print(f"\n{'='*60}")
    print(f"Testing SPHINX Native Backend: {model_type.upper()}")
    print(f"Mode: {mode}, Omega Strategy: {omega_strategy}")
    print(f"{'='*60}\n")
    
    # Configuration
    seq_len = 24
    parity_groups = [[1, 5, 9, 13], [2, 7, 11, 19], [4, 8, 12, 16, 20]]
    batch_size = 128
    epochs = 30
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Create datasets
    config = ParityConfig(seq_len=seq_len, parity_groups=parity_groups)
    train_dataset = ParitySequenceDataset(config, num_samples=4000, seed=42)
    val_dataset = ParitySequenceDataset(config, num_samples=1000, seed=43)
    test_dataset = ParitySequenceDataset(config, num_samples=1000, seed=44)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create incidence matrix
    h = create_incidence_matrix(mode, seq_len).to(device)
    print(f"Incidence shape: {h.shape}")
    
    # Create model
    model = SPHINXMCMBackend(
        model_type=model_type,
        in_dim=3,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        seq_len=seq_len,
        omega_strategy=omega_strategy,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            x = batch["node_features"].to(device)
            mask_idx = batch["mask_idx"].to(device)
            target = batch["target"].to(device)
            
            optimizer.zero_grad()
            logits = model(x, h, mask_idx)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == target).sum().item()
            train_total += x.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["node_features"].to(device)
                mask_idx = batch["mask_idx"].to(device)
                target = batch["target"].to(device)
                logits = model(x, h, mask_idx)
                loss = criterion(logits, target)
                
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == target).sum().item()
                val_total += x.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    # Test
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch["node_features"].to(device)
            mask_idx = batch["mask_idx"].to(device)
            target = batch["target"].to(device)
            logits = model(x, h, mask_idx)
            loss = criterion(logits, target)
            
            test_loss += loss.item() * x.size(0)
            test_correct += (logits.argmax(dim=1) == target).sum().item()
            test_total += x.size(0)
    
    test_loss /= test_total
    test_acc = test_correct / test_total
    
    print(f"\nFinal Results:")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"\n{'='*60}\n")
    
    return {
        "model_type": model_type,
        "mode": mode,
        "omega_strategy": omega_strategy,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SPHINX native backend")
    parser.add_argument("--model", type=str, default="hcha", choices=["hcha", "all_deepsets"])
    parser.add_argument("--mode", type=str, default="all_correct", 
                        choices=["none", "wrong", "one_correct", "all_correct"])
    parser.add_argument("--omega", type=str, default="zeros",
                        choices=["zeros", "positional", "sincos"])
    
    args = parser.parse_args()
    
    result = test_backend(model_type=args.model, mode=args.mode, omega_strategy=args.omega)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in result.items():
        print(f"{k}: {v}")
