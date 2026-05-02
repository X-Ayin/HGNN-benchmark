from __future__ import annotations

import torch
import torch.nn as nn


class MCMTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], mask_idx: [B]
        bsz, n, _ = x.shape
        if n != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, but got {n}")

        pos_ids = torch.arange(n, device=x.device).unsqueeze(0).expand(bsz, n)
        h = self.input_proj(x) + self.pos_embed(pos_ids)
        h = self.encoder(h)

        gather = h[torch.arange(bsz, device=h.device), mask_idx]
        logits = self.head(gather)
        return logits
