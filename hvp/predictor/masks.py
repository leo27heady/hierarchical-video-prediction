"""Cross-scale conditioning masks (rule-based, built once, cached as buffers).

The child (finer, more frames) cross-attends the parent (coarser, fewer frames). The
allowed connections are the causal receptive-field overlap — but with the FUTURE-LEAK
FIX from the plan: a child frame ``tc`` may attend a parent frame ``p`` only if
``max(RF(p)) <= tc`` (the parent summarizes child frames no later than the child's own
context), so conditioning never imports content from after the predicted frame.

Verified equivalence to the user's spec: for top(T3)->mid(T5),
``causal_receptive_groups(5)`` = [{0},{0,1,2},{2,3,4}] = t1'->[m0], t2'->[m0,m1,m2],
t3'->[m2,m3,m4].
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def causal_receptive_groups(t_in: int, kernel_t: int = 3, stride_t: int = 2) -> List[List[int]]:
    """For a causal stride-2 kernel-3 downsample, the fine frames each coarse frame sees."""
    pad = kernel_t - 1
    t_out = (t_in - 1) // stride_t + 1
    groups: List[List[int]] = []
    for p in range(t_out):
        start = p * stride_t - pad
        groups.append(list(range(max(0, start), min(t_in - 1, start + kernel_t - 1) + 1)))
    return groups


def cross_rf_mask(
    child_t: int, child_h: int, child_w: int,
    parent_t: int, parent_h: int, parent_w: int,
    device=None,
) -> torch.Tensor:
    """(Nc, Np) bool, True=allowed. Spatial: child cell -> its covering parent cell.
    Temporal: leak-free causal RF (``max(RF(p)) <= tc``)."""
    rf = causal_receptive_groups(child_t)
    if len(rf) != parent_t:
        raise ValueError(f"RF groups {len(rf)} != parent_t {parent_t} (child_t={child_t})")
    rf_max = [max(g) for g in rf]
    sh, sw = child_h // parent_h, child_w // parent_w

    nc, np_ = child_t * child_h * child_w, parent_t * parent_h * parent_w
    mask = torch.zeros(nc, np_, dtype=torch.bool, device=device)
    for tc in range(child_t):
        for hc in range(child_h):
            for wc in range(child_w):
                ci = (tc * child_h + hc) * child_w + wc
                php, pwp = hc // sh, wc // sw
                for tp in range(parent_t):
                    if rf_max[tp] <= tc:                       # leak-free causal
                        pi = (tp * parent_h + php) * parent_w + pwp
                        mask[ci, pi] = True
    return mask


class CascadeMasks(nn.Module):
    """Caches the per-child cross-attention mask as a registered buffer (moves with model)."""

    def __init__(self, stage_thw: List[tuple]) -> None:
        # stage_thw fine->coarse: [(Tb,Hb,Wb),(Tm,Hm,Wm),(Tt,Ht,Wt)]
        super().__init__()
        self.n = len(stage_thw)
        for s in range(self.n - 1):  # child s, parent s+1
            ct, ch, cw = stage_thw[s]
            pt, ph, pw = stage_thw[s + 1]
            self.register_buffer(f"cross_{s}", cross_rf_mask(ct, ch, cw, pt, ph, pw), persistent=False)

    def cross(self, child_stage: int) -> torch.Tensor:
        return getattr(self, f"cross_{child_stage}")
