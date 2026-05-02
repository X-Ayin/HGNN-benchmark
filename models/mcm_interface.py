from __future__ import annotations

import torch


def validate_mcm_inputs(
    x: torch.Tensor,
    incidence: torch.Tensor,
    mask_idx: torch.Tensor,
    expected_feat_dim: int = 3,
) -> None:
    if x.dim() != 3:
        raise ValueError(f"x must be [B,N,F], got shape={tuple(x.shape)}")
    if x.size(-1) != expected_feat_dim:
        raise ValueError(
            f"x last dim must be {expected_feat_dim}, got F={x.size(-1)}"
        )
    if mask_idx.dim() != 1:
        raise ValueError(f"mask_idx must be [B], got shape={tuple(mask_idx.shape)}")
    if mask_idx.size(0) != x.size(0):
        raise ValueError(
            f"mask_idx batch {mask_idx.size(0)} must match x batch {x.size(0)}"
        )
    if incidence.dim() not in (2, 3):
        raise ValueError(
            f"incidence must be [N,E] or [B,N,E], got shape={tuple(incidence.shape)}"
        )
    if incidence.dim() == 2 and incidence.size(0) != x.size(1):
        raise ValueError(
            f"incidence N={incidence.size(0)} must match x N={x.size(1)} for [N,E]"
        )
    if incidence.dim() == 3:
        if incidence.size(0) != x.size(0) or incidence.size(1) != x.size(1):
            raise ValueError(
                "incidence [B,N,E] must align with x [B,N,F], got "
                f"incidence={tuple(incidence.shape)} x={tuple(x.shape)}"
            )
