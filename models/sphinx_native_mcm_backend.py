"""Simplified SPHINX backend for MCM task, avoiding residual connection issues."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add SPHINX-main to path
from gnn_mcm.models.sphinx_path import resolve_sphinx_path
from gnn_mcm.models.mcm_interface import validate_mcm_inputs

SPHINX_PATH = resolve_sphinx_path()
if str(SPHINX_PATH) not in sys.path:
    sys.path.insert(0, str(SPHINX_PATH))

try:
    from layers import HCHAConv, DenseAllDeepSetsConvNBAOrig, MLP_model
except ImportError as e:
    warnings.warn(f"Failed to import SPHINX layers: {e}")
    raise

from gnn_mcm.adapters.sphinx_adapter import ParityToSPHINXAdapter


class SPHINXMCMBackend(nn.Module):
    """
    Simplified SPHINX backend for MCM task.
    
    Uses SPHINX's HCHAConv layers but without the problematic residual connection
    that requires num_features == num_classes.
    """
    
    def __init__(
        self,
        model_type: str = "hcha",
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 24,
        omega_strategy: str = "zeros",
    ):
        super().__init__()
        
        self.model_type = model_type
        self.seq_len = seq_len
        self.adapter = ParityToSPHINXAdapter(seq_len=seq_len, omega_strategy=omega_strategy)
        
        # Input projection: in_dim -> hidden_dim
        self.input_proj = MLP_model(
            in_dim, hidden_dim, hidden_dim, 2,
            dropout=dropout, normalization="None"
        )
        
        # SPHINX conv layers
        self.convs = nn.ModuleList()
        if model_type == "hcha":
            for _ in range(num_layers):
                self.convs.append(
                    HCHAConv(
                        hidden_dim, hidden_dim, hidden_dim, in_dim,
                        dropout, 2, "None", False, False
                    )
                )
        elif model_type == "all_deepsets":
            # Use AllDeepSets-style layers (simplified, no omega)
            for _ in range(num_layers):
                self.convs.append(
                    DenseAllDeepSetsConvNBAOrig(
                        hidden_dim, hidden_dim, hidden_dim, in_dim,
                        dropout, 2, "None", False
                    )
                )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        mask_idx: torch.Tensor,
        position_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, N, F] node features
            h: [N, E] incidence matrix
            mask_idx: [B] masked position indices
            position_idx: Optional position indices
            
        Returns:
            logits: [B, 2] classification logits
        """
        validate_mcm_inputs(x, h, mask_idx, expected_feat_dim=3)
        B, N, F = x.shape
        
        # Convert to SPHINX format
        feats, omega = self.adapter.parity_to_sphinx(x)  # [1, B*N, F], [B*N]
        
        # Process each sample separately
        outputs = []
        for b in range(B):
            # Extract this sample's features: [N, 1, F]
            sample_feats = feats[:, b*N:(b+1)*N, :].transpose(0, 1)  # [N, 1, F]
            sample_omega = omega[b*N:(b+1)*N]  # [N]
            
            # Transpose for SPHINX: [N, 1, F] -> [1, N, F]
            sample_feats = sample_feats.transpose(0, 1)  # [1, N, F]
            
            # Initial projection
            out = self.input_proj(sample_feats)  # [1, N, hidden_dim]
            
            # Apply conv layers
            for conv in self.convs:
                if self.model_type == "hcha":
                    # HCHAConv expects (X, Z, H, X0) where X is [ts, num_nodes, feat]
                    # Z is omega, H is incidence
                    out = conv(out, sample_omega.unsqueeze(-1), h, sample_feats)
                else:
                    # AllDeepSets doesn't use omega
                    out = conv(out, h, sample_feats)
                out = F.relu(out)
                out = self.dropout(out)
            
            # Extract features: [1, N, hidden_dim] -> [N, hidden_dim]
            outputs.append(out.squeeze(0))
        
        # Stack outputs: [B, N, hidden_dim]
        out_batch = torch.stack(outputs, dim=0)
        
        # Gather masked positions
        masked_features = out_batch[torch.arange(B, device=out_batch.device), mask_idx]
        
        # Classify
        logits = self.classifier(masked_features)
        
        return logits
