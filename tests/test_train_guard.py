"""The bf16 trainer has no GradScaler, so a non-finite GRADIENT (the loss can stay finite)
would be written into every weight by opt.step(). These tests lock the guard contract:
clip_grad_norm_ does NOT stop NaN, AdamW.step() corrupts params, and grad_norm() (which the
trainer checks with math.isfinite) is the predicate that catches it before the step."""

import math

import torch

from hvp.train import grad_norm


def _model_with_nan_grad():
    m = torch.nn.Linear(4, 4)
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    next(m.parameters()).grad[0, 0] = float("nan")
    return m


def test_grad_norm_is_nonfinite_under_nan_grad():
    m = _model_with_nan_grad()
    assert not math.isfinite(grad_norm(list(m.parameters())))


def test_clip_does_not_remove_nan_and_step_corrupts():
    m = _model_with_nan_grad()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    assert torch.isnan(next(m.parameters()).grad).any()        # clip did NOT sanitize it
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    opt.step()
    assert torch.isnan(next(m.parameters())).any()             # the unguarded step corrupts weights


def test_guard_predicate_skips_before_corruption():
    m = _model_with_nan_grad()
    snapshot = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    gn = grad_norm(list(m.parameters()))
    if not math.isfinite(gn):                                  # the trainer's guard
        opt.zero_grad(set_to_none=True)
    else:
        opt.step()
    for p, s in zip(m.parameters(), snapshot):
        assert torch.equal(p, s)                               # weights untouched
