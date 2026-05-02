"""Wrappers for SPHINX original code components."""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

import torch
import torch.nn as nn

# Add SPHINX-main to path for imports
from gnn_mcm.models.sphinx_path import resolve_sphinx_path
from gnn_mcm.models.mcm_interface import validate_mcm_inputs

SPHINX_PATH = resolve_sphinx_path()
if str(SPHINX_PATH) not in sys.path:
    sys.path.insert(0, str(SPHINX_PATH))

# Now import SPHINX components
try:
    from hypergraph_predictor import HypergraphPredictor, HypergraphPredictorSeq
    from models import HCHA, DenseAllDeepSets
    from layers import MLP_model  # MLP_model is in layers.py, not utils.py
except ImportError as e:
    warnings.warn(f"Failed to import SPHINX modules: {e}. Make sure SPHINX-main is in the correct location.")
    raise

# Import our adapter
from gnn_mcm.adapters.sphinx_adapter import ParityToSPHINXAdapter


class SPHINXConfig:
    """Configuration for SPHINX models matching their expected format."""
    def __init__(
        self,
        num_features: int = 3,
        num_classes: int = 2,
        MLP_hidden: int = 64,
        num_layers: int = 2,
        MLP_num_layers: int = 2,
        dropout: float = 0.1,
        MLP_norm: str = "None",
        add_self_loop: bool = False,
        degree_detached: bool = False,
        add_velocity: bool = False,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.MLP_hidden = MLP_hidden
        self.num_layers = num_layers
        self.MLP_num_layers = MLP_num_layers
        self.dropout = dropout
        self.MLP_norm = MLP_norm
        self.add_self_loop = add_self_loop
        self.degree_detached = degree_detached
        self.add_velocity = add_velocity


class SPHINXNativeBackend(nn.Module):
    """
    Wrapper for SPHINX original backend models (HCHA, AllDeepSets).
    
    Adapts Parity MCM format to SPHINX format and back.
    """
    
    def __init__(
        self,
        model_type: str = "hcha",
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 24,
        omega_strategy: str = "positional",
    ):
        super().__init__()
        
        self.model_type = model_type
        self.seq_len = seq_len
        self.adapter = ParityToSPHINXAdapter(seq_len=seq_len, omega_strategy=omega_strategy)
        
        # Create SPHINX config
        config = SPHINXConfig(
            num_features=in_dim,
            num_classes=2,  # Binary classification for MCM
            MLP_hidden=hidden_dim,
            num_layers=num_layers,
            MLP_num_layers=2,
            dropout=dropout,
            MLP_norm="None",
        )
        
        # Create backend model
        if model_type == "hcha":
            self.backend = HCHA(config, use_attention=False)
        elif model_type == "all_deepsets":
            self.backend = DenseAllDeepSets(config)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Create classification head for MCM task
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
        Forward pass through SPHINX backend.
        
        Args:
            x: [B, N, F] node features (Parity format)
            h: [N, E] or [B, N, E] incidence matrix
            mask_idx: [B] indices of masked positions
            position_idx: [B, N] position indices (optional)
            
        Returns:
            logits: [B, 2] classification logits
        """
        validate_mcm_inputs(x, h, mask_idx, expected_feat_dim=3)
        B, N, F = x.shape
        
        # Convert to SPHINX format
        feats, omega = self.adapter.parity_to_sphinx(x)  # [1, B*N, F], [B*N, 1]
        
        # Expand incidence matrix if needed
        if h.dim() == 2:
            # [N, E] -> [B, N, E] but SPHINX expects [N, E] format
            H = h  # Keep as is, SPHINX will broadcast
        else:
            # [B, N, E] -> need to handle batch dimension
            # For now, use first sample's structure (assumes same structure across batch)
            H = h[0]  # [N, E]
        
        # Run through SPHINX backend
        # SPHINX expects feats as [num_nodes, ts, feat]
        # We have feats: [1, B*N, F] which is [ts, num_nodes, feat]
        # Need to transpose to [num_nodes, ts, feat]: [B*N, 1, F]
        feats_sphinx = feats.transpose(0, 1)  # [B*N, 1, F]
        
        # SPHINX HCHA forward expects (feats, omega, H) where:
        # - feats: [num_nodes, ts, feat]
        # - omega: [num_nodes, 1] 
        # - H: [num_nodes, num_edges]
        # But we have batched data, so need to process each sample separately
        
        # Process each sample in the batch separately
        outputs = []
        for b in range(B):
            # Extract features for this sample: [N, 1, F]
            sample_feats = feats_sphinx[b*N:(b+1)*N]
            sample_omega = omega[b*N:(b+1)*N]
            
            # Forward through SPHINX backend
            sample_out = self.backend(sample_feats, sample_omega, H)  # [N, 1, hidden_dim]
            outputs.append(sample_out)
        
        # Concatenate outputs: [B*N, 1, hidden_dim]
        out = torch.cat(outputs, dim=0)
        
        # Extract features for masked positions
        # out is [B*N, 1, hidden_dim], we need to gather masked positions
        out_flat = out.squeeze(1)  # [B*N, hidden_dim]
        
        # Reshape to [B, N, hidden_dim]
        out_batch = out_flat.view(B, N, -1)
        
        # Gather masked positions
        masked_features = out_batch[torch.arange(B, device=out.device), mask_idx]  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(masked_features)  # [B, 2]
        
        return logits


class SPHINXNativeDiscoverer(nn.Module):
    """
    Wrapper for SPHINX original hypergraph discoverer.
    
    Adapts Parity MCM format to SPHINX format.
    """
    
    def __init__(
        self,
        num_slots: int = 3,
        dim: int = 64,
        num_iter: int = 3,
        temperature: float = 1.0,
        slot_k: int = 4,
        seq_len: int = 24,
        in_feat: int = 3,
        hidden_dim: int = 64,
        selector: str = "simple",
        sequential: bool = False,
        omega_strategy: str = "positional",
        temporal_len: int = 1,
        temporal_mode: str = "repeat",
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.seq_len = seq_len
        self.sequential = sequential
        self.in_feat = in_feat
        self.adapter = ParityToSPHINXAdapter(
            seq_len=seq_len,
            omega_strategy=omega_strategy,
            temporal_len=temporal_len,
            temporal_mode=temporal_mode,
        )
        
        # SPHINX discoverer config
        discoverer_class = HypergraphPredictorSeq if sequential else HypergraphPredictor
        
        common_args = dict(
            num_slots=num_slots,
            dim=dim,
            num_iter=num_iter,
            temperature=temperature,
            nonlin="sigmoid",
            init_slot="learned",
            slot_k=slot_k,
            hidden_dim=hidden_dim,
            selector=selector,
            MLP_norm="None",
            encoder_type="MLP",  # Use MLP instead of TempCNN for static data
            enc_len=temporal_len,
            in_feat=in_feat,
            num_nodes=seq_len,
        )
        
        if sequential:
            self.discoverer = discoverer_class(
                **common_args,
                deterministic=False,
                long_history=False,
                nb_sample=1,
            )
        else:
            self.discoverer = discoverer_class(**common_args)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Discover hypergraph structure from node features.
        
        Args:
            x: [B, N, F] node features for discoverer
            
        Returns:
            Dictionary with:
                - incidence: [B, N, num_slots] incidence matrix
                - attn: [B, num_slots, N] attention weights
        """
        if x.dim() != 3 or x.size(-1) != self.in_feat:
            raise ValueError(f"x must be [B,N,{self.in_feat}], got shape={tuple(x.shape)}")
        B, N, F = x.shape
        
        # Convert to SPHINX format: [B, N, F] -> [1, B*N, F]
        feats, omega = self.adapter.parity_to_sphinx(x)
        
        # SPHINX expects [ts, bs*num_nodes, feat]
        # We have [1, B*N, F] which matches!
        
        # Run discoverer
        edge_index, edge_weights, attn = self.discoverer(feats)
        
        # Convert outputs back to our format
        # attn is [B, num_slots, N] - exactly what we need!
        incidence = self.adapter.attn_to_incidence(attn)  # [B, N, num_slots]
        
        return {
            "incidence": incidence,
            "attn": attn,
            "edge_index": edge_index,
            "edge_weights": edge_weights,
        }
