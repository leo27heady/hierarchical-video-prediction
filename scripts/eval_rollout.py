"""P1 eval: autoregressive rollout + per-horizon drift on the long-horizon val set.

Rebuilds the exact model from the checkpoint's stored config, free-runs both feedback
modes, and reports per-horizon PSNR/SSIM/changed-region vs teacher-forced and persistence,
the TF->AR gap, the horizon where AR stops beating persistence, and FVD. Saves a drift strip
and (optionally) logs to wandb.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from hvp.config import Config, TopologyConfig
from hvp.data import dataset_from_cfg
from hvp.eval.horizon import horizon_eval
from hvp.eval.quality import LPIPSMetric
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import build_tokenizer


def load(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = Config.from_dict(blob["config"]) if "config" in blob else Config(topology=TopologyConfig(**blob["topo"]))
    tok = build_tokenizer(cfg).to(device).eval()
    tok.load_state_dict(blob["model"])
    cas = HierarchicalCascade(cfg.topology, cfg.predictor).to(device).eval()
    cas.load_state_dict(blob["cascade"])
    return tok, cas, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/oneshot/last.pt")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--strip", default="logs/rollout_strip.png")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, cas, cfg = load(args.ckpt, device)
    loader = DataLoader(dataset_from_cfg(cfg, "val_long"), batch_size=args.batch_size, num_workers=0)
    lpips_metric = None if args.no_lpips else LPIPSMetric()

    res = horizon_eval(tok, cas, loader, cfg, device, modes=("pixel", "latent"),
                       max_batches=args.max_batches, lpips_metric=lpips_metric, strip_path=args.strip)

    n = res["n_steps"]
    hs = [0, 1, 2, 4, 7, 11, min(15, n - 1), n - 1]
    hs = sorted(set(h for h in hs if h < n))
    print(f"\n=== rollout per-horizon (seed_len={res['seed_len']}, n_steps={n}) ===")
    print("  full-frame PSNR (dB):")
    print(f"  {'h':>3s} | {'AR-px':>7s} {'AR-lat':>7s} {'TF':>7s} {'persist':>7s} {'linear':>7s} {'rotation':>8s}")
    for h in hs:
        print(f"  {h:>3d} | {res['psnr/pixel'][h]:7.2f} {res['psnr/latent'][h]:7.2f} {res['psnr/tf'][h]:7.2f} "
              f"{res['psnr/persist'][h]:7.2f} {res['psnr/linear'][h]:7.2f} {res['psnr/rotation'][h]:8.2f}")
    print("\n  CHANGED-REGION PSNR (moving pixels — the honest metric):")
    print(f"  {'h':>3s} | {'AR-px':>7s} {'persist':>7s} {'linear':>7s} {'rotation':>8s}")
    for h in hs:
        print(f"  {h:>3d} | {res['changed_psnr/pixel'][h]:7.2f} {res['changed_psnr/persist'][h]:7.2f} "
              f"{res['changed_psnr/linear'][h]:7.2f} {res['changed_psnr/rotation'][h]:8.2f}")
    print(f"\n  TF->AR gap (mean over horizons): {res['tf_ar_gap_mean']:+.2f} dB")
    print("  CHANGED-REGION gate — AR(pixel) beats each MOTION baseline until horizon:")
    for b in ("persist", "linear", "rotation"):
        print(f"    vs {b:8s}: {res[f'beats_{b}_changed_until']:>2d}/{n}  "
              f"(mean AR - {b} = {res[f'ar_minus_{b}_changed_mean']:+.2f} dB)")
    if "fvd/pixel" in res:
        print(f"  FVD  AR-pixel {res['fvd/pixel']:.1f}  AR-latent {res.get('fvd/latent', float('nan')):.1f}")
    if "lpips/pixel" in res:
        print(f"  LPIPS AR-pixel: {res['lpips/pixel']:.3f}")
    print(f"  drift strip: {res.get('strip')}")

    if args.wandb:
        from hvp.wandblog import WandbLogger
        logger = WandbLogger(cfg, cfg.run.__class__(wandb=True, wandb_project="hvp-vqfree", wandb_name="rollout"))
        import wandb
        tbl = wandb.Table(columns=["horizon", "AR_pixel", "AR_latent", "TF", "persist", "AR_pixel_changed", "tf_ar_gap"])
        for h in range(n):
            tbl.add_data(h, res["psnr/pixel"][h], res["psnr/latent"][h], res["psnr/tf"][h],
                         res["psnr/persist"][h], res["changed_psnr/pixel"][h], res["tf_ar_gap"][h])
        logger.log({"rollout/per_horizon": tbl, "rollout/tf_ar_gap_mean": res["tf_ar_gap_mean"],
                    "rollout/beats_persistence_until": res["beats_persistence_until"]}, step=0)
        logger.log_image_table([(0, "drift", _imread(res["strip"]))], step=0, key="rollout/strip") if res.get("strip") else None
        logger.finish()


def _imread(path):
    from PIL import Image
    import numpy as np
    return np.array(Image.open(path))


if __name__ == "__main__":
    main()
