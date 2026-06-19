import torch

from hvp.attention.masks import temporal_causal_mask
from hvp.attention.rope import Rotary3D, default_split


def test_default_split_even_and_sums():
    for hd in (8, 12, 16, 24, 32, 48):
        dims = default_split(hd)
        assert sum(dims) == hd
        assert all(d % 2 == 0 for d in dims)


def test_rope_preserves_norm():
    rope = Rotary3D(16)
    q = torch.randn(2, 4, 3 * 4 * 4, 16)
    k = torch.randn_like(q)
    qr, kr = rope.apply(q, k, 3, 4, 4)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-4)


def test_rope_relative_shift_invariance():
    """q·k under RoPE depends only on the relative temporal offset (same H,W cell)."""
    rope = Rotary3D(16)
    T = 5
    qv, kv = torch.randn(16), torch.randn(16)
    q = torch.zeros(1, 1, T, 16)
    k = torch.zeros(1, 1, T, 16)
    q[0, 0, 1] = qv; q[0, 0, 3] = qv
    k[0, 0, 0] = kv; k[0, 0, 2] = kv
    qr, kr = rope.apply(q, k, T, 1, 1)
    d10 = (qr[0, 0, 1] * kr[0, 0, 0]).sum()   # offset +1
    d32 = (qr[0, 0, 3] * kr[0, 0, 2]).sum()   # offset +1
    assert torch.allclose(d10, d32, atol=1e-4)


def test_causal_mask_pattern():
    m = temporal_causal_mask(3, 1, 1)
    assert m.tolist() == [[True, False, False], [True, True, False], [True, True, True]]
