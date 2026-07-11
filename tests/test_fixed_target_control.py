import types

import numpy as np
import torch

from hvp_rev.faithful_nets import LAT
from hvp_rev.fixed_target_control import (
    FixedTargetWindowDiT,
    fixed_target_window_batch,
    update_strided_context_buffer,
)
from hvp_rev.rhvd import WindowRegression


def _data(seed=0):
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.RandomState(seed)
    data = types.SimpleNamespace()
    data.N, data.T = 5, 60
    data.train_idx, data.val_idx = np.arange(4), np.arange(4, 5)
    data.zc = torch.randn(5, 60, LAT, 4, 4, generator=generator)
    data.zf = torch.randn(5, 60, LAT, 8, 8, generator=generator)
    data.acts = rng.randint(-1, 2, (5, 60)).astype(np.int64)
    return data


def test_fixed_target_batch_separates_context_and_target_stride():
    batch = fixed_target_window_batch(
        _data(), 2, n_ctx_c=24, n_ctx_f=2, context_stride_c=2, seed=1,
    )
    assert batch["ctx_c"].shape[1] == 24
    assert batch["tgt_c"].shape[1] == 4
    assert batch["tgt_f"].shape[1] == 4


def test_fixed_target_model_has_matched_four_slot_interface():
    torch.manual_seed(0)
    net = FixedTargetWindowDiT(
        n_ctx_c=24, n_ctx_f=2, stride_c=2, mode="reg", d=48, depth=1, heads=4,
    )
    assert net.base == 48
    assert net.n_tgt_c == 4
    assert net.expected_seq == 512 + 4 * 16 + 4 * 64
    batch = fixed_target_window_batch(
        _data(), 1, n_ctx_c=24, n_ctx_f=2, context_stride_c=2, seed=2,
    )
    loss = WindowRegression(net).loss(**batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_rollout_buffer_appends_only_stride_anchors():
    old = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    pred = torch.arange(100, 104, dtype=torch.float32).view(1, 4, 1)
    new = update_strided_context_buffer(old, pred, context_stride_c=2, keep_slots=6)
    assert new.flatten().tolist() == [2, 3, 4, 5, 101, 103]
