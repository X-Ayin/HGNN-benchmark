"""Hyperparameter search for SPHINX end-to-end model."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from datetime import datetime
import itertools
from typing import Dict, List, Any

# Disable torch.compile to avoid triton dependency issues
import torch._dynamo
torch._dynamo.config.disable = True

from gnn_mcm.data.parity_dataset import ParityConfig, ParitySequenceDataset
from gnn_mcm.run_sphinx_native_end_to_end import SPHINXEndToEnd, train_epoch, evaluate


def run_single_config(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
) -> Dict[str, Any]:
    """
    Train a single configuration and return results.
    
    Args:
        config: Hyperparameter configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        
    Returns:
        Dictionary with results and config
    """
    # Create model
    model = SPHINXEndToEnd(
        backend_type=config["backend"],
        num_slots=config["num_slots"],
        slot_k=config["slot_k"],
        seq_len=24,
        in_dim=3,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pos_emb_dim=config.get("pos_emb_dim", 0),
        use_positional_embedding=config.get("pos_emb_dim", 0) > 0,
        selector=config["selector"],
        temperature=config["temperature"],
        omega_strategy=config.get("omega_strategy", "zeros"),
        temporal_len=config.get("temporal_len", 1),
        temporal_mode=config.get("temporal_mode", "repeat"),
        frontend_posenc_dim=config.get("frontend_posenc_dim", 0),
    ).to(device)
    
    backend_optimizer = torch.optim.Adam(model.backend.parameters(), lr=config["lr"])
    discoverer_optimizer = torch.optim.Adam(
        model.discoverer.parameters(), lr=config["lr"] / config.get("ld", 10.0)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, backend_optimizer, discoverer_optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
    
    return {
        "config": config,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_acc": val_accs[-1],
        "train_losses": train_losses,
        "val_accs": val_accs,
    }


def grid_search(
    param_grid: Dict[str, List[Any]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Perform grid search over hyperparameters.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of epochs per configuration
        top_k: Number of top configurations to return
        
    Returns:
        List of top-k configurations with results
    """
    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nGrid Search: Testing {len(configs)} configurations")
    print(f"Parameters: {keys}")
    print(f"{'='*60}\n")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing config: {config}")
        
        try:
            result = run_single_config(config, train_loader, val_loader, device, epochs)
            results.append(result)
            
            print(f"  -> Best Val Acc: {result['best_val_acc']:.4f} (epoch {result['best_epoch']})")
            print()
            
        except Exception as e:
            print(f"  -> ERROR: {str(e)}")
            print()
            continue
    
    # Sort by best validation accuracy
    results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    
    return results[:top_k]


def random_search(
    param_space: Dict[str, List[Any]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_trials: int = 20,
    epochs: int = 30,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Perform random search over hyperparameters.
    
    Args:
        param_space: Dictionary mapping parameter names to lists of possible values
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        n_trials: Number of random configurations to try
        epochs: Number of epochs per configuration
        seed: Random seed
        
    Returns:
        List of all configurations with results, sorted by performance
    """
    import random
    random.seed(seed)
    
    print(f"\nRandom Search: Testing {n_trials} random configurations")
    print(f"Parameters: {list(param_space.keys())}")
    print(f"{'='*60}\n")
    
    results = []
    for i in range(n_trials):
        # Sample random configuration
        config = {k: random.choice(v) for k, v in param_space.items()}
        
        print(f"[{i+1}/{n_trials}] Testing config: {config}")
        
        try:
            result = run_single_config(config, train_loader, val_loader, device, epochs)
            results.append(result)
            
            print(f"  -> Best Val Acc: {result['best_val_acc']:.4f} (epoch {result['best_epoch']})")
            print()
            
        except Exception as e:
            print(f"  -> ERROR: {str(e)}")
            print()
            continue
    
    # Sort by best validation accuracy
    results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    
    return results


def main():
    """Main hyperparameter search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPHINX hyperparameter search")
    parser.add_argument("--search_type", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for random search")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per configuration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=5, help="Number of top configs to save")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--pos_emb_dim", type=int, default=0)
    parser.add_argument("--omega_strategy", type=str, default="zeros")
    parser.add_argument("--temporal_len", type=int, default=1)
    parser.add_argument("--temporal_mode", type=str, default="repeat", choices=["repeat"])
    parser.add_argument("--frontend_posenc_dim", type=int, default=0, choices=[0, 2])
    parser.add_argument("--backend", type=str, default="all_deepsets", choices=["hcha", "all_deepsets"])
    parser.add_argument("--num_slots", type=int, default=3)
    parser.add_argument("--slot_k", type=int, default=4)
    parser.add_argument("--selector_space", type=str, default="simple,imle")
    parser.add_argument("--temperature_space", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--hidden_dim_space", type=str, default="64,128")
    parser.add_argument("--num_layers_space", type=str, default="2,3")
    parser.add_argument("--dropout_space", type=str, default="0.0,0.1,0.2")
    parser.add_argument("--lr_space", type=str, default="0.001,0.0005,0.0002")
    parser.add_argument("--ld_space", type=str, default="10,5,2")
    
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
    print(f"SPHINX Hyperparameter Search")
    print(f"Search Type: {args.search_type}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create datasets
    config = ParityConfig(seq_len=seq_len, parity_groups=parity_groups)
    train_dataset = ParitySequenceDataset(config, num_samples=args.train_samples, seed=args.seed)
    val_dataset = ParitySequenceDataset(config, num_samples=args.val_samples, seed=args.seed + 1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define search space
    selector_space = [s.strip() for s in args.selector_space.split(",") if s.strip()]
    temperature_space = [float(s.strip()) for s in args.temperature_space.split(",") if s.strip()]
    hidden_dim_space = [int(s.strip()) for s in args.hidden_dim_space.split(",") if s.strip()]
    num_layers_space = [int(s.strip()) for s in args.num_layers_space.split(",") if s.strip()]
    dropout_space = [float(s.strip()) for s in args.dropout_space.split(",") if s.strip()]
    lr_space = [float(s.strip()) for s in args.lr_space.split(",") if s.strip()]
    ld_space = [float(s.strip()) for s in args.ld_space.split(",") if s.strip()]

    if args.search_type == "grid":
        # Smaller grid for computational efficiency
        param_grid = {
            "backend": [args.backend],
            "selector": selector_space,
            "num_slots": [args.num_slots],
            "slot_k": [args.slot_k],
            "hidden_dim": hidden_dim_space,
            "num_layers": num_layers_space,
            "dropout": dropout_space,
            "lr": lr_space,
            "ld": ld_space,
            "temperature": temperature_space,
            "pos_emb_dim": [args.pos_emb_dim],
            "omega_strategy": [args.omega_strategy],
            "temporal_len": [args.temporal_len],
            "temporal_mode": [args.temporal_mode],
            "frontend_posenc_dim": [args.frontend_posenc_dim],
        }
        
        results = grid_search(
            param_grid=param_grid,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            top_k=args.top_k,
        )
        
    else:  # random search
        param_space = {
            "backend": [args.backend],
            "selector": selector_space,
            "num_slots": [args.num_slots],
            "slot_k": [args.slot_k],
            "hidden_dim": hidden_dim_space,
            "num_layers": num_layers_space,
            "dropout": dropout_space,
            "lr": lr_space,
            "ld": ld_space,
            "temperature": temperature_space,
            "pos_emb_dim": [args.pos_emb_dim],
            "omega_strategy": [args.omega_strategy],
            "temporal_len": [args.temporal_len],
            "temporal_mode": [args.temporal_mode],
            "frontend_posenc_dim": [args.frontend_posenc_dim],
        }
        
        results = random_search(
            param_space=param_space,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_trials=args.n_trials,
            epochs=args.epochs,
            seed=args.seed,
        )
        results = results[:args.top_k]  # Keep only top k
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Configurations:")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"Rank {i}:")
        print(f"  Config: {result['config']}")
        print(f"  Best Val Acc: {result['best_val_acc']:.4f} (epoch {result['best_epoch']})")
        print(f"  Final Val Acc: {result['final_val_acc']:.4f}")
        print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"gnn_mcm/logs/hyperparam_search_{args.search_type}_{timestamp}.json"
    
    summary = {
        "search_type": args.search_type,
        "n_trials": args.n_trials if args.search_type == "random" else len(results),
        "epochs_per_config": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "frontend_posenc_dim": args.frontend_posenc_dim,
        "top_results": results,
        "best_config": results[0]["config"] if results else None,
        "best_val_acc": results[0]["best_val_acc"] if results else None,
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"\nBest configuration:")
    print(json.dumps(results[0]["config"], indent=2))
    print(f"Best Val Acc: {results[0]['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
