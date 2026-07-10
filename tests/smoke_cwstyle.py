"""CPU smoke for the CW-style-table program arms (B1 hier-s2-min / B2 strided-flat / B3 hier3)
+ bit-compat guards for legacy arms. Run WITHOUT touching the GPU (experiments may be live):

    CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m tests.smoke_cwstyle

Checks:
  1. hier3.py AE smoke (frozen trunk, shapes, 2-level encode bit-equal to base).
  2. New arms: expected_seq == pre-registered budgets + targets; loss+backward finite.
  3. Legacy arms (hier-long, d1-56-diff): seq/target counts unchanged; NO top params in the
     state dict (checkpoint compatibility); strided stride-4 rule still yields ONE slot.
  4. Full rollout_eval on synthetic data for each new arm: oracle=True (buffer/stride phasing)
     and nfe=1 sampled (model path incl. 3-tuple sample unpack + blind option).
"""
import os
import sys
import types

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1"), \
    "run with CUDA_VISIBLE_DEVICES=-1 so the smoke never touches the GPU"

import numpy as np
import torch

from hvp_rev import hier3
from hvp_rev.hier_convae import HierConvAE, LAT
from hvp_rev.rhvd import (ARMS, WIN, EMA, WindowDiT, WindowDiffusion, WindowRegression,
                          arm_cfg, ctx_reach, rollout_eval, window_batch)

assert not torch.cuda.is_available(), "CUDA visible -- refuse to smoke while experiments run"


def check(name, cond, msg=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {msg}")
    if not cond:
        sys.exit(1)


# ---------- 1. hier3 AE ----------
print("== hier3 AE ==")
hier3._smoke()

# ---------- 2/3. arm construction, budgets, compat ----------
print("== arm construction ==")
BUDGET = {  # pre-registered ctx-token budgets
    "hier-s2-min": 24 * 16 + 2 * 64,            # 512
    "strided-flat": (11 + 6) * 64,              # 1088
    "hier3": 48 * 4 + 24 * 16 + 9 * 64,         # 1152
    "hier-s4-min": 12 * 16 + 2 * 64,            # 320 [E-320]
    "flat-320": 5 * 64,                         # 320
}
TGTS = {  # (n_tgt_t, n_tgt_c) and target tokens
    "hier-s2-min": (0, 2, 2 * 16 + WIN * 64),   # 2 coarse slots (stride 2 divides WIN=4)
    "strided-flat": (0, WIN, WIN * 16 + WIN * 64),
    "hier3": (WIN, WIN, WIN * 4 + WIN * 16 + WIN * 64),
    "hier-s4-min": (0, 1, 1 * 16 + WIN * 64),   # 1 slot (the legacy d1 stride-4 rule)
    "flat-320": (0, WIN, WIN * 16 + WIN * 64),
}
nets = {}
for arm in ("hier-s2-min", "strided-flat", "hier3", "hier-s4-min", "flat-320"):
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = arm_cfg(arm)
    net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
                    n_hist_f=n_hist_f, n_ctx_t=n_ctx_t, stride_t=stride_t)
    nets[arm] = net
    n_tgt_t, n_tgt_c, tgt_tok = TGTS[arm]
    check(f"{arm} ctx budget", net.expected_seq - tgt_tok == BUDGET[arm],
          f"(ctx {net.expected_seq - tgt_tok} == {BUDGET[arm]}; seq {net.expected_seq})")
    check(f"{arm} targets", (net.n_tgt_t, net.n_tgt_c) == (n_tgt_t, n_tgt_c),
          f"(n_tgt_t={net.n_tgt_t}, n_tgt_c={net.n_tgt_c})")
    reach_want = 5 if arm == "flat-320" else 48
    check(f"{arm} reach", net.base == ctx_reach(n_ctx_c, n_ctx_f, stride_c, n_hist_f,
                                                n_ctx_t, stride_t) == reach_want)

# legacy compat: hier-long unchanged; d1-56-diff stride-4 rule still ONE slot; no top params
print("== legacy compat ==")
for arm, seq_want, ntc_want in (("hier-long", 48 * 16 + 6 * 64 + WIN * 16 + WIN * 64, WIN),
                                ("d1-56-diff", 56 * 16 + 6 * 64 + 1 * 16 + WIN * 64, 1)):
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = arm_cfg(arm)
    net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
                    n_hist_f=n_hist_f, n_ctx_t=n_ctx_t, stride_t=stride_t)
    check(f"{arm} seq unchanged", net.expected_seq == seq_want, f"({net.expected_seq})")
    check(f"{arm} n_tgt_c unchanged", net.n_tgt_c == ntc_want, f"({net.n_tgt_c})")
    keys = set(net.state_dict().keys())
    check(f"{arm} no top keys", not any(("top" in k or "_t_" in k or k.startswith("head_t"))
                                        for k in keys))
    check(f"{arm} 2-level tokenizer", net.tok.level_embedding.num_embeddings == 2)
check("hier3 3-level tokenizer", nets["hier3"].tok.level_embedding.num_embeddings == 3)

# ---------- loss + backward on synthetic batches ----------
print("== loss/backward ==")


def fake_data(N=6, T=60, W=192, speed=4, top=False, seed=0):
    """Minimal ActionHierData stand-in with random latents (standardized scale ~1)."""
    rng = np.random.RandomState(seed)
    d = types.SimpleNamespace()
    d.N, d.T, d.W, d.speed, d.V = N, T, W, speed, 64
    d.acts = rng.randint(-1, 2, (N, T)).astype(np.int64)
    d.clips_u8 = (rng.rand(N, T, 64, 64) * 255).astype(np.uint8)
    g = torch.Generator().manual_seed(seed)
    d.zc = torch.randn(N, T, LAT, 4, 4, generator=g)
    d.zf = torch.randn(N, T, LAT, 8, 8, generator=g)
    if top:
        d.zt = torch.randn(N, T, LAT, 2, 2, generator=g)
    d.train_idx = np.arange(0, N - 2)
    d.val_idx = np.arange(N - 2, N)
    return d


for arm in ("hier-s2-min", "strided-flat", "hier3", "hier-s4-min", "flat-320"):
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = arm_cfg(arm)
    data = fake_data(top=(n_ctx_t > 0))
    bt = window_batch(data, 2, n_ctx_c, n_ctx_f, train=True, seed=1, stride_c=stride_c,
                      n_hist_f=n_hist_f, n_ctx_t=n_ctx_t, stride_t=stride_t)
    for wrap_cls in (WindowDiffusion, WindowRegression):
        wrap = wrap_cls(nets[arm])
        loss = wrap.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"],
                         ctx_t=bt.get("ctx_t"), tgt_t=bt.get("tgt_t"))
        loss.backward()
        check(f"{arm} {wrap_cls.__name__} loss", torch.isfinite(loss).item(),
              f"({float(loss):.3f})")
        nets[arm].zero_grad(set_to_none=True)

# ---------- 4. rollout_eval end-to-end (oracle + sampled + blind) on synthetic data ----------
print("== rollout_eval (synthetic, WARM=48, T=60 -> 3 windows) ==")
ae = HierConvAE()  # untrained decode is fine; smoke checks plumbing, not quality


class _Wrap:
    Ttr = 1000


for arm in ("hier-s2-min", "strided-flat", "hier3", "hier-s4-min"):
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = arm_cfg(arm)
    data = fake_data(top=(n_ctx_t > 0))
    ema = EMA(nets[arm])
    for kwargs in (dict(oracle=True), dict(oracle=False),
                   *([dict(oracle=False, blind="top")] if n_ctx_t > 0 else
                     [dict(oracle=False, blind="coarse")] if n_ctx_c > 0 else
                     [dict(oracle=False, blind="fine")])):
        tagged = dict(kwargs)
        blind = tagged.get("blind", "")
        tag = "_smoke" + (f"_blind-{blind}" if blind else "")
        path = rollout_eval(arm, _Wrap(), ema, data, ae, seed=999, n_clips=2, nfe=1,
                            chunk=2, warm=48, tag=tag, **tagged)
        with np.load(path) as z:
            n_want = 2 * ((60 - 48) // WIN) * WIN
            ok = len(z["fg"]) == n_want and np.isfinite(z["fg"]).all()
            n_rec = len(z["fg"])
        check(f"{arm} rollout {kwargs}", ok, f"({n_rec} records)")
        os.remove(path)
# cleanup smoke viz strips
import glob
for f in glob.glob("logs_rhcp/rhvd_rollout_*_s999.png"):
    os.remove(f)

print("\nSMOKE OK: all CW-style-table arms construct, train-step, and roll out on CPU;"
      " legacy arms bit-compatible.")
