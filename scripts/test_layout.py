"""Layout unit test for the H-grid strided arms (HGRID_DESIGN.md v2 §2 GO-gate item a).
Checks, on CPU with dummy tensors: sequence lengths, frame-id anchors (coarse slots at block
ends 3,7,...,191; hybrid hist slots at 1,5,...,185 + recent 186..191), target group structure,
and that window_batch slices agree with the model's frame-id convention on a synthetic dataset
where zc[t] encodes t (so anchor identity is directly readable from the tensor values).

  venv/Scripts/python.exe scripts/test_layout.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, ".")
import hvp_rev.rhvd as R                                   # noqa: E402

LAT, T = 16, 480


def fake_data():
    """zc[ci,t] filled with value t, zf likewise -> slice anchors readable from values."""
    zc = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1).expand(1, T, LAT, 4, 4).clone()
    zf = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1).expand(1, T, LAT, 8, 8).clone()
    return SimpleNamespace(zc=zc, zf=zf, acts=np.zeros((1, T), np.int64),
                           clips_u8=np.zeros((1, T, 64, 64), np.uint8),
                           train_idx=np.array([0]), val_idx=np.array([0]), T=T)


def ids(model, ctx_c_frames, ctx_f_frames):
    ctx_c = torch.zeros(1, ctx_c_frames, LAT, 4, 4)
    ctx_f = torch.zeros(1, ctx_f_frames, LAT, 8, 8)
    xs, lvs, rls, fms = model._ctx_tokens(ctx_c, ctx_f)
    fm = torch.cat(fms, 1)[0]
    # one frame id per GROUP (16 tokens/coarse frame, 64/fine)
    out, pos = [], 0
    for x, f in zip(xs, fms):
        out.append(int(f[0, 0]))
    return out


def main():
    # ---- d1: 48 coarse slots @ stride 4 + 6 fine ----
    m = R.WindowDiT(n_ctx_c=48, n_ctx_f=6, mode="diff", stride_c=4, n_hist_f=0)
    assert m.base == 192 and m.n_tgt_c == 1, (m.base, m.n_tgt_c)
    assert m.expected_seq == 48 * 16 + 6 * 64 + 1 * 16 + 4 * 64 == 1424, m.expected_seq
    got = ids(m, 48, 6)
    assert got[:48] == list(range(3, 192, 4)), f"coarse anchors wrong: {got[:5]}..{got[45:48]}"
    assert got[48:] == [186, 187, 188, 189, 190, 191], f"fine ids wrong: {got[48:]}"
    yc, yf = m(torch.zeros(1, 1, LAT, 4, 4), torch.zeros(1, 4, LAT, 8, 8),
               torch.zeros(1, 48, LAT, 4, 4), torch.zeros(1, 6, LAT, 8, 8),
               torch.zeros(1), torch.zeros(1, 4, dtype=torch.long))
    assert yc.shape == (1, 1, LAT, 4, 4) and yf.shape == (1, 4, LAT, 8, 8)
    print("d1 layout: seq 1424, coarse anchors 3..191 step 4, fine 186..191, 1 coarse target  OK")

    # ---- hybrid control: 47 full-res slots @4 + 6 recent ----
    h = R.WindowDiT(n_ctx_c=0, n_ctx_f=6, mode="diff", stride_c=1, n_hist_f=47)
    assert h.base == 192 and h.n_tgt_c == 4, (h.base, h.n_tgt_c)
    assert h.expected_seq == 53 * 64 + 4 * 16 + 4 * 64 == 3712, h.expected_seq
    got = ids(h, 0, 53)
    assert got[:47] == list(range(1, 186, 4)), f"hist anchors wrong: {got[:5]}..{got[44:47]}"
    assert got[47:] == [186, 187, 188, 189, 190, 191], f"recent ids wrong: {got[47:]}"
    print("hybrid layout: seq 3712 (== d0-long), hist anchors 1..185 step 4, recent 186..191  OK")

    # ---- window_batch slices agree with the model convention (value == frame index) ----
    data = fake_data()
    rng_state = np.random.get_state(); np.random.seed(0)
    bt = R.window_batch(data, 1, 48, 6, train=True, seed=5, stride_c=4, n_hist_f=0)
    np.random.set_state(rng_state)
    cc, cf, tc = bt["ctx_c"][0, :, 0, 0, 0], bt["ctx_f"][0, :, 0, 0, 0], bt["tgt_c"][0, :, 0, 0, 0]
    tj = int(cf[-1].item()) + 1                            # last fine ctx frame = tj-1
    assert torch.allclose(cc.cpu(), torch.arange(tj - 189, tj, 4).float()), "d1 ctx_c slice wrong"
    assert tc.numel() == 1 and int(tc.item()) == tj + 3, "d1 coarse target anchor wrong"
    bt = R.window_batch(data, 1, 0, 6, train=True, seed=7, stride_c=1, n_hist_f=47)
    cf = bt["ctx_f"][0, :, 0, 0, 0]
    tj = int(cf[-1].item()) + 1
    assert torch.allclose(cf[:47].cpu(), torch.arange(tj - 191, tj - 6, 4).float()), "hybrid hist slice wrong"
    assert torch.allclose(cf[47:].cpu(), torch.arange(tj - 6, tj).float()), "hybrid recent slice wrong"
    print("window_batch slices match model frame-id convention (synthetic value==index)  OK")
    print("LAYOUT TEST PASS")


if __name__ == "__main__":
    main()
