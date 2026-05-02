"""Adapter to convert Parity MCM data format to SPHINX-compatible format."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ParityToSPHINXAdapter(nn.Module):
    """
    Adapts Parity MCM data to SPHINX format.
    
    Parity format: [B, N, 3] where features are [value, is_mask, pos_idx]
    SPHINX format: [T, B*N, F] where T is time steps (we use T=1 for static data)
    
    SPHINX also expects omega parameter for rotation angles.
    """
    
    def __init__(
        self,
        seq_len: int = 24,
        omega_strategy: str = "positional",
        omega_dim: int = 16,
        temporal_len: int = 1,
        temporal_mode: str = "repeat",
    ) -> None:
        """
        Args:
            seq_len: Length of sequences (number of nodes)
            omega_strategy: Strategy for generating omega parameter
                - "zeros": All zeros (simplest)
                - "positional": Use normalized position indices
                - "sincos": Use sin/cos encoding of positions
                - "learned": Learnable position embeddings (projected to 1D)
                - "fourier": Fourier features encoding
            omega_dim: Dimension for learnable embeddings (only used with "learned")
            temporal_len: Number of synthetic timesteps fed to SPHINX
            temporal_mode: Packing mode when temporal_len > 1.
                - "repeat": duplicate static features across timesteps
        """
        super().__init__()
        self.seq_len = seq_len
        self.omega_strategy = omega_strategy
        self.omega_dim = omega_dim
        self.temporal_len = temporal_len
        self.temporal_mode = temporal_mode
        if self.temporal_len < 1:
            raise ValueError("temporal_len must be >= 1")
        if self.temporal_mode not in {"repeat"}:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")
        
        # Learnable position embeddings if needed
        if omega_strategy == "learned":
            self.pos_embedding = nn.Embedding(seq_len, omega_dim)
            self.omega_proj = nn.Linear(omega_dim, 1)
            # Initialize
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
            nn.init.xavier_uniform_(self.omega_proj.weight)
        
    def parity_to_sphinx(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Parity format to SPHINX format.
        
        Args:
            x: [B, N, 3] tensor with Parity features
            
        Returns:
            feats: [T, B*N, F] tensor for SPHINX
            omega: [B*N, 1] tensor for rotation angles
        """
        B, N, F = x.shape
        assert N == self.seq_len, f"Expected seq_len={self.seq_len}, got {N}"
        
        # SPHINX expects [T, num_nodes, feat].
        if self.temporal_len == 1:
            feats = x.reshape(1, B * N, F)
        else:
            if self.temporal_mode == "repeat":
                feats = x.reshape(1, B * N, F).expand(self.temporal_len, -1, -1).contiguous()
            else:
                raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")
        
        # Generate omega based on strategy
        omega = self._generate_omega(B, N, x.device)
        
        return feats, omega
    
    def _generate_omega(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate omega parameter for SPHINX.
        
        IMPORTANT: SPHINX's HCHA will do omega.unsqueeze(-1) internally,
        so we need to return [B*N] (1D) instead of [B*N, 1] (2D).
        After unsqueeze(-1), it becomes [B*N, 1] which is what HCHA expects.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device for tensor
            
        Returns:
            omega: [B*N] tensor (1D, will be unsqueezed to [B*N, 1] by SPHINX)
        """
        if self.omega_strategy == "zeros":
            # Simplest: all zeros - return 1D tensor
            omega = torch.zeros(batch_size * seq_len, device=device)
            
        elif self.omega_strategy == "positional":
            # Use normalized position indices [0, 1, 2, ..., N-1] / N
            pos = torch.arange(seq_len, device=device, dtype=torch.float32) / seq_len
            # Repeat for batch: [B, N] -> [B*N] (1D)
            omega = pos.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
            
        elif self.omega_strategy == "sincos":
            # Use phase encoding similar to positional embeddings
            # But SPHINX expects scalar, so we use position * 2*pi / N
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            phase = pos * (2 * 3.14159265359 / seq_len)
            omega = phase.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
            
        elif self.omega_strategy == "learned":
            # Learnable position embeddings
            pos_indices = torch.arange(seq_len, device=device)
            pos_emb = self.pos_embedding(pos_indices)  # [N, omega_dim]
            pos_scalar = self.omega_proj(pos_emb).squeeze(-1)  # [N]
            # Repeat for batch: [B, N] -> [B*N] (1D)
            omega = pos_scalar.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
            
        elif self.omega_strategy == "fourier":
            # Fourier features encoding with multiple frequencies
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            # Use multiple frequencies: 1, 2, 4, 8
            freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device)
            # Compute weighted sum of sin and cos features
            fourier_feats = []
            for freq in freqs:
                fourier_feats.append(torch.sin(2 * 3.14159265359 * freq * pos / seq_len))
                fourier_feats.append(torch.cos(2 * 3.14159265359 * freq * pos / seq_len))
            # Average all fourier features
            omega_pos = torch.stack(fourier_feats).mean(dim=0)  # [N]
            # Repeat for batch: [B, N] -> [B*N] (1D)
            omega = omega_pos.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
            
        else:
            raise ValueError(f"Unknown omega_strategy: {self.omega_strategy}")
            
        return omega
    
    def edge_index_to_incidence(
        self, 
        edge_index: torch.Tensor, 
        edge_weights: torch.Tensor,
        batch_size: int,
        num_slots: int
    ) -> torch.Tensor:
        """
        Convert SPHINX edge_index output to incidence matrix.
        
        SPHINX outputs:
            edge_index: [2, E] with node and hyperedge indices
            edge_weights: [E] with edge weights
            
        We need to convert to:
            incidence: [B, N, M] where M is number of hyperedges
            
        Args:
            edge_index: [2, E] tensor
            edge_weights: [E] tensor
            batch_size: Batch size
            num_slots: Number of hyperedges per sample
            
        Returns:
            incidence: [B, N, M] tensor
        """
        N = self.seq_len
        M = num_slots
        
        # Initialize incidence matrix
        incidence = torch.zeros(batch_size, N, M, device=edge_index.device)
        
        # edge_index[0] contains node indices (flattened across batch)
        # edge_index[1] contains hyperedge indices (flattened across batch)
        node_idx = edge_index[0]  # [E]
        edge_idx = edge_index[1]  # [E]
        
        # Recover batch, node, and slot indices
        batch_idx = node_idx // N
        node_local = node_idx % N
        edge_local = edge_idx % M
        
        # Fill incidence matrix
        for i in range(edge_index.shape[1]):
            b = batch_idx[i].item()
            n = node_local[i].item()
            m = edge_local[i].item()
            incidence[b, n, m] = edge_weights[i].item()
            
        return incidence
    
    def attn_to_incidence(
        self,
        attn: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert SPHINX attn output (alternative format) to incidence matrix.
        
        SPHINX may also output attn: [B, num_slots, N]
        We need: [B, N, num_slots]
        
        Args:
            attn: [B, num_slots, N] tensor
            
        Returns:
            incidence: [B, N, num_slots] tensor
        """
        # Simply transpose last two dimensions
        return attn.transpose(1, 2)
