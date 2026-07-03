"""Floors + AE-reuse gate for testbed operating point B (HGRID_DESIGN.md v2 §1).

1. AE-reuse gate: frozen hier_ae_mom.pt recon MSE on new-data frames (fine level, [-1,1] scale)
   vs the 0.0020 loopmom baseline; accept <= ~2x.
2. Floors on the NEW test split, rollout protocol (WARM=192, horizons 1..288, eval clips =
   val_idx[:100] convention): persistence fg-MSE, AE round-trip ceiling (fg-MSE + fg-sharp),
   raw-GT sharpness.

  venv/Scripts/python.exe scripts/floors_ae_gate.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hvp_rev import hier_convae                                   # noqa: E402
from hvp_rev.step1_mech import fg_mse_px_frames, fg_sharp         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WARM, HMAX = 192, 288
AE_BASELINE = 0.0020


def frames_to_pm1(u8):
    return torch.from_numpy(u8.astype(np.float32) / 127.5 - 1.0).unsqueeze(1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/hgrid_train_1200x480.npy")
    ap.add_argument("--test_path", default="data/hgrid_test_300x480.npy")
    args = ap.parse_args()
    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    train = np.load(args.train_path, mmap_mode="r")
    test = np.load(args.test_path, mmap_mode="r")

    # ---- 1. AE-reuse gate: 512 random frames from the new TRAIN file ----
    rng = np.random.RandomState(0)
    ci = rng.randint(0, train.shape[0], 512)
    ti = rng.randint(0, train.shape[1], 512)
    mses_f, mses_c = [], []
    for lo in range(0, 512, 128):
        x = frames_to_pm1(np.stack([train[c, t] for c, t in zip(ci[lo:lo+128], ti[lo:lo+128])])).to(DEV)
        zc, zf = ae.encode(x)
        rf, rc = ae.decode_fine(zf), ae.decode_coarse(zc)
        mses_f.append(torch.mean((rf - x) ** 2).item())
        mses_c.append(torch.mean((rc - x) ** 2).item())
    mf, mc = float(np.mean(mses_f)), float(np.mean(mses_c))
    verdict = "PASS (reuse frozen AE)" if mf <= 2.0 * AE_BASELINE else "FAIL -> retrain AE on new train file"
    print(f"[AE gate] fine recon MSE {mf:.4f} (baseline {AE_BASELINE:.4f}, accept <= {2*AE_BASELINE:.4f}) "
          f"| coarse {mc:.4f}  -> {verdict}")

    # ---- 2. floors on the eval subset (last 150 test clips -> first 100, matching rhvd convention) ----
    n_val = min(300, test.shape[0] // 2)
    val_idx = np.arange(test.shape[0] - n_val, test.shape[0])
    cs = val_idx[:100]
    pers, ceil, ceil_sh, gt_sh, base_full = [], [], [], [], []
    pers_u8 = np.asarray(test[cs, WARM - 1])                      # persistence frame per clip
    p01_pers = torch.from_numpy(pers_u8.astype(np.float32) / 255.0).unsqueeze(1).to(DEV)
    hs = list(range(1, HMAX + 1, 8))                              # every 8th horizon (36 slices)
    for h in hs:
        gt_u8 = np.asarray(test[cs, WARM + h - 1])
        gt01 = torch.from_numpy(gt_u8.astype(np.float32) / 255.0).unsqueeze(1).to(DEV)
        # persistence floor
        pers.append(fg_mse_px_frames(p01_pers, gt01).mean())
        # AE round-trip ceiling
        x = frames_to_pm1(gt_u8).to(DEV)
        _, zf = ae.encode(x)
        rt01 = (ae.decode_fine(zf).clamp(-1, 1) + 1) * 0.5
        ceil.append(fg_mse_px_frames(rt01, gt01).mean())
        ceil_sh.append(fg_sharp(rt01, gt01))
        gt_sh.append(fg_sharp(gt01, gt01))
        base_full.append(float(((p01_pers - gt01) ** 2).mean()))
    print(f"[floors @ WARM={WARM}, h=1..{HMAX}] persistence fg-MSE {np.mean(pers):.4f} "
          f"(full-frame {np.mean(base_full):.4f})")
    print(f"[floors] AE round-trip ceiling: fg-MSE {np.mean(ceil):.4f}  fg-sharp {np.mean(ceil_sh):.4f}")
    print(f"[floors] raw-GT fg-sharp {np.mean(gt_sh):.4f}")
    print("FLOORS_DONE")


if __name__ == "__main__":
    main()
