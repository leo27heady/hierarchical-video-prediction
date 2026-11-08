import numpy as np
import torch

from hvp_rev.adaptive_context import (
    CONTEXT_TOKENS,
    SelectionConfig,
    cell_scores,
    select_fine_cells,
    select_spacetime_cells,
    selection_diagnostics,
)
from hvp_rev.rhvd import WIN, WindowDiT, WindowRegression


def test_selected_context_preserves_1152_token_budget_and_backward():
    torch.manual_seed(0)
    net = WindowDiT(
        n_ctx_c=48, n_ctx_f=48, mode="reg", d=48, depth=1, heads=3,
        fine_cells_per_frame=8,
    )
    assert net.expected_seq == CONTEXT_TOKENS + WIN * 16 + WIN * 64 == 1472
    batch = 1
    ctx_c = torch.randn(batch, 48, 16, 4, 4)
    ctx_f = torch.randn(batch, 48, 16, 8, 8)
    ids = torch.arange(8).view(1, 1, 8).expand(batch, 48, 8)
    tgt_c = torch.randn(batch, WIN, 16, 4, 4)
    tgt_f = torch.randn(batch, WIN, 16, 8, 8)
    actions = torch.zeros(batch, WIN, dtype=torch.long)
    loss = WindowRegression(net).loss(
        ctx_c, ctx_f, tgt_c, tgt_f, actions, ctx_f_ids=ids
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in net.parameters())


def test_cell_selector_is_deterministic_and_uses_true_cell_ids():
    frames = torch.zeros(1, 3, 64, 64)
    frames[:, :, :8, :8] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
    frames[:, :, 8:16, 8:16] = torch.rand(1, 3, 8, 8)
    cfg = SelectionConfig(mode="variance", k=8, ema_alpha=0.0, seed=4)
    first = select_fine_cells(frames, cfg)
    second = select_fine_cells(frames, cfg)
    assert torch.equal(first, second)
    assert first.shape == (1, 3, 8)
    assert 9 in first[0, 0].tolist()
    assert torch.all(first[..., 1:] >= first[..., :-1])


def test_entropy_motion_random_and_diagnostics_are_finite():
    rng = np.random.RandomState(2)
    frames = (rng.rand(2, 4, 64, 64) * 255).astype(np.uint8)
    for mode in ("entropy", "motion", "hybrid", "random"):
        cfg = SelectionConfig(mode=mode, k=8, ema_alpha=0.5, seed=3)
        scores = cell_scores(frames, cfg)
        ids = select_fine_cells(frames, cfg)
        report = selection_diagnostics(frames, ids)
        assert scores.shape == (2, 4, 64)
        assert torch.isfinite(scores).all()
        assert all(np.isfinite(value) for value in report.values())
        assert all(0.0 <= value <= 1.0 for value in report.values())


def test_recent_reserved_layout_has_exact_budget_and_dense_two_frame_tail():
    frames = torch.rand(2, 48, 64, 64)
    cfg = SelectionConfig(mode="variance", layout="recent2", ema_alpha=0.5)
    frame_ids, cell_ids = select_spacetime_cells(frames, cfg)
    assert frame_ids.shape == cell_ids.shape == (2, 384)
    for batch in range(2):
        assert int((frame_ids[batch] == 46).sum()) == 64
        assert int((frame_ids[batch] == 47).sum()) == 64
        for frame in range(46):
            assert int((frame_ids[batch] == frame).sum()) >= 5
    net = WindowDiT(
        n_ctx_c=48, n_ctx_f=48, mode="reg", d=48, depth=1, heads=3,
        fine_token_budget=384,
    )
    assert net.expected_seq == 1472


def test_conservative_recent_four_layout_retains_budget():
    frames = torch.rand(1, 48, 64, 64)
    cfg = SelectionConfig(
        mode="variance", layout="recent4", ema_alpha=0.5,
        dense_recent_frames=4, min_older_cells=2,
    )
    frame_ids, cell_ids = select_spacetime_cells(frames, cfg)
    assert frame_ids.shape == cell_ids.shape == (1, 384)
    for frame in range(44, 48):
        assert int((frame_ids[0] == frame).sum()) == 64
    for frame in range(44):
        assert int((frame_ids[0] == frame).sum()) >= 2


def test_global_recent_four_tokens_run_through_predictor_and_backward():
    torch.manual_seed(4)
    batch = 1
    frames = torch.rand(batch, 48, 64, 64)
    cfg = SelectionConfig(
        mode="variance", layout="recent4", ema_alpha=0.5,
        dense_recent_frames=4, min_older_cells=2,
    )
    frame_ids, cell_ids = select_spacetime_cells(frames, cfg)
    net = WindowDiT(
        n_ctx_c=48, n_ctx_f=48, mode="reg", d=48, depth=1, heads=3,
        fine_token_budget=384,
    )
    ctx_c = torch.randn(batch, 48, 16, 4, 4)
    ctx_f = torch.randn(batch, 48, 16, 8, 8)
    tgt_c = torch.randn(batch, WIN, 16, 4, 4)
    tgt_f = torch.randn(batch, WIN, 16, 8, 8)
    actions = torch.zeros(batch, WIN, dtype=torch.long)
    loss = WindowRegression(net).loss(
        ctx_c, ctx_f, tgt_c, tgt_f, actions,
        ctx_f_ids=cell_ids, ctx_f_frame_ids=frame_ids,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert net.tok.patchify.weight.grad is not None
