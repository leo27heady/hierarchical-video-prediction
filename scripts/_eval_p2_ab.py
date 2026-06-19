"""P2.1 A/B gate: compare the rollout-trained model vs the matched control on the offline motion
gate (the authoritative metric). Does the K=1 self-rollout term reduce the TF->AR gap and the
changed-region DRIFT vs an identical control? Usage:
  python scripts/_eval_p2_ab.py logs/so3_p2_ctrl/oneshot/last.pt logs/so3_p2/oneshot/last.pt
"""
import sys

import torch
from torch.utils.data import DataLoader

from hvp.config import Config
from hvp.data import dataset_from_cfg
from hvp.eval.horizon import horizon_eval
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import build_tokenizer


def load(ck, dev):
    b = torch.load(ck, map_location=dev, weights_only=False)
    cfg = Config.from_dict(b["config"])
    tok = build_tokenizer(cfg).to(dev).eval(); tok.load_state_dict(b["model"])
    cas = HierarchicalCascade(cfg.topology, cfg.predictor).to(dev).eval(); cas.load_state_dict(b["cascade"])
    return tok, cas, cfg


@torch.no_grad()
def ev(ck, dev):
    tok, cas, cfg = load(ck, dev)
    loader = DataLoader(dataset_from_cfg(cfg, "val_long"), batch_size=10, num_workers=0)
    return horizon_eval(tok, cas, loader, cfg, dev, modes=("pixel",), max_batches=6)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctrl, p2 = ev(sys.argv[1], dev), ev(sys.argv[2], dev)
    n = ctrl["n_steps"]
    print(f"\n=== P2.1 K=1 self-rollout vs control (SO(3) val_long, {n} horizons) ===")

    def row(name, key, h=None, better="lower"):
        cv = ctrl[key][h] if h is not None else ctrl[key]
        pv = p2[key][h] if h is not None else p2[key]
        d = pv - cv
        win = (d < 0) if better == "lower" else (d > 0)
        print(f"  {name:34s} ctrl {cv:8.3f}  p2 {pv:8.3f}  d {d:+.3f} {'[win]' if win else '[ . ]'}")

    row("TF->AR gap mean (lower=better)", "tf_ar_gap_mean", better="lower")
    print("  changed_psnr/pixel (higher=better):")
    for h in [0, 2, 4, 7, 11, min(15, n - 1), n - 1]:
        if h < n:
            row(f"   @h{h}", "changed_psnr/pixel", h, better="higher")
    row("beats_rotation_changed_until (higher)", "beats_rotation_changed_until", better="higher")
    row("ar-rotation changed mean (higher)", "ar_minus_rotation_changed_mean", better="higher")
    row("ar-persist changed mean (higher)", "ar_minus_persist_changed_mean", better="higher")
    if "fvd/pixel" in ctrl and "fvd/pixel" in p2:
        row("FVD pixel (lower, noisy)", "fvd/pixel", better="lower")
    print("\n  Precondition read: P2.1 should at least not REGRESS drift; a clear Δ>0 on changed_psnr@late-h")
    print("  is the signal the lever exists (full ship gate is P2.2 K=2 with a T>=11 cache).")


if __name__ == "__main__":
    main()
