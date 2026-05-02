"""Hyperparameter search for TDHNN end-to-end model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import itertools
import json
import random
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch._dynamo

from gnn_mcm.data.parity_dataset import ParityConfig, ParitySequenceDataset
from gnn_mcm.run_tdhnn_native_end_to_end import TDHNNEndToEnd, train_epoch, evaluate

torch._dynamo.config.disable = True


def run_single_config(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    model = TDHNNEndToEnd(
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
        frontend_posenc_dim=config.get("frontend_posenc_dim", 0),
        k_e=config["k_e"],
        dynamic_edges=config["dynamic_edges"],
    ).to(device)

    backend_optimizer = torch.optim.Adam(model.backend.parameters(), lr=config["lr"])
    discoverer_optimizer = torch.optim.Adam(
        model.discoverer.parameters(), lr=config["lr"] / config["ld"]
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = -1
    best_val_metrics = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, backend_optimizer, discoverer_optimizer, criterion, device
        )
        val_loss, val_acc, val_struct_metrics = evaluate(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_val_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                **val_struct_metrics,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }

    test_loss, test_acc, test_structures, test_struct_metrics = evaluate(
        model, test_loader, criterion, device, return_structures=True
    )

    return {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_struct_metrics": test_struct_metrics,
        "sample_structures": test_structures[:5],
    }


def make_grid_configs(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def make_random_configs(param_space: Dict[str, List[Any]], n_trials: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    return [{k: random.choice(v) for k, v in param_space.items()} for _ in range(n_trials)]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TDHNN hyperparameter search")
    parser.add_argument("--search_type", type=str, default="random", choices=["grid", "random"])
    parser.add_argument("--n_trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--backend", type=str, default="all_deepsets", choices=["hcha", "all_deepsets"])
    parser.add_argument("--num_slots_space", type=str, default="10,12,15")
    parser.add_argument("--slot_k_space", type=str, default="4,5,6")
    parser.add_argument("--k_e_space", type=str, default="2,3,4")
    parser.add_argument("--dynamic_edges_space", type=str, default="true,false")
    parser.add_argument("--hidden_dim_space", type=str, default="64")
    parser.add_argument("--num_layers_space", type=str, default="2")
    parser.add_argument("--dropout_space", type=str, default="0.1")
    parser.add_argument("--lr_space", type=str, default="0.001,0.0005")
    parser.add_argument("--ld_space", type=str, default="10,5")
    parser.add_argument("--pos_emb_dim", type=int, default=0)
    parser.add_argument("--frontend_posenc_dim", type=int, default=0, choices=[0, 2])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    seq_len = 24
    parity_groups = [[1, 5, 9, 13], [2, 7, 11, 19], [4, 8, 12, 16, 20]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("TDHNN Hyperparameter Search")
    print(f"search_type={args.search_type}, n_trials={args.n_trials}, epochs={args.epochs}")
    print(f"device={device}")
    print(f"{'='*70}\n")

    config = ParityConfig(seq_len=seq_len, parity_groups=parity_groups)
    train_dataset = ParitySequenceDataset(config, num_samples=args.train_samples, seed=args.seed)
    val_dataset = ParitySequenceDataset(config, num_samples=args.val_samples, seed=args.seed + 1)
    test_dataset = ParitySequenceDataset(config, num_samples=args.test_samples, seed=args.seed + 2)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    dynamic_edges_space = [s.strip().lower() == "true" for s in args.dynamic_edges_space.split(",") if s.strip()]
    param_space = {
        "backend": [args.backend],
        "num_slots": [int(s.strip()) for s in args.num_slots_space.split(",") if s.strip()],
        "slot_k": [int(s.strip()) for s in args.slot_k_space.split(",") if s.strip()],
        "k_e": [int(s.strip()) for s in args.k_e_space.split(",") if s.strip()],
        "dynamic_edges": dynamic_edges_space,
        "hidden_dim": [int(s.strip()) for s in args.hidden_dim_space.split(",") if s.strip()],
        "num_layers": [int(s.strip()) for s in args.num_layers_space.split(",") if s.strip()],
        "dropout": [float(s.strip()) for s in args.dropout_space.split(",") if s.strip()],
        "lr": [float(s.strip()) for s in args.lr_space.split(",") if s.strip()],
        "ld": [float(s.strip()) for s in args.ld_space.split(",") if s.strip()],
        "pos_emb_dim": [args.pos_emb_dim],
        "frontend_posenc_dim": [args.frontend_posenc_dim],
    }

    if args.search_type == "grid":
        configs = make_grid_configs(param_space)
    else:
        configs = make_random_configs(param_space, args.n_trials, args.seed)

    results = []
    for idx, cfg in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] config={cfg}")
        try:
            res = run_single_config(
                cfg, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device, epochs=args.epochs
            )
            results.append(res)
            print(
                "  -> best_val_acc={:.4f}, test_acc={:.4f}, slot_overlap={:.4f}".format(
                    res["best_val_acc"], res["test_acc"], res["test_struct_metrics"]["slot_overlap_jaccard"]
                )
            )
        except Exception as e:
            print(f"  -> ERROR: {e}")

    results.sort(key=lambda r: (r["best_val_acc"], r["test_acc"]), reverse=True)
    top_results = results[: args.top_k]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"gnn_mcm/logs/tdhnn_hyperparam_search_{args.search_type}_{timestamp}.json"
    payload = {
        "search_type": args.search_type,
        "n_configs": len(configs),
        "epochs_per_config": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "param_space": param_space,
        "top_results": top_results,
        "best_config": top_results[0]["config"] if top_results else None,
        "best_val_acc": top_results[0]["best_val_acc"] if top_results else None,
        "best_test_acc": top_results[0]["test_acc"] if top_results else None,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved search results to: {out_file}")
    if top_results:
        print("Best config:")
        print(json.dumps(top_results[0]["config"], indent=2, ensure_ascii=False))
        print(
            "best_val_acc={:.4f}, test_acc={:.4f}".format(
                top_results[0]["best_val_acc"], top_results[0]["test_acc"]
            )
        )


if __name__ == "__main__":
    main()

