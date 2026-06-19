"""G0 — the decisive pre-build gate.

On a FROZEN, recon-only encoder (detached predictive target, the plan default), cache the
per-stage continuous latents on a clip cross-split, fit a tiny motion-aware (3D-RoPE
attention) next-frame predictor per stage, and measure held-out variance-explained over
the persistence baseline — especially on the MID stage and on the cells that change.

Pass rule (per plan §4): MID held-out ve_changed >= 0.05 and stage effective rank stays
above floor. Loss->0 with collapsing rank is the SIGNATURE of collapse, not success.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from hvp.config import Config, TopologyConfig
from hvp.data import dataset_from_cfg
from hvp.predictor.continuous import ContinuousStagePredictor, predictive_loss, variance_explained
from hvp.tokenizer import HierTokenizer
from hvp.train import effective_rank


def load_tokenizer(ckpt_path: str, device) -> HierTokenizer:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    topo = TopologyConfig(**blob["topo"])
    tok = HierTokenizer(topo).to(device).eval()
    tok.load_state_dict(blob["model"])
    return tok


@torch.no_grad()
def cache_stage_features(tok: HierTokenizer, loader: DataLoader, device, max_batches=None) -> List[torch.Tensor]:
    """Encode all clips; return per-stage tensors (N, C, T, H, W) on CPU."""
    n = tok.topology.n_stages()
    buckets: List[List[torch.Tensor]] = [[] for _ in range(n)]
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        z = tok.encoder(batch["video"].to(device))
        for s in range(n):
            buckets[s].append(z[s].float().cpu())
    return [torch.cat(b, 0) for b in buckets]


def fit_stage_probe(train_feat: torch.Tensor, val_feat: torch.Tensor, *, dim, n_heads, n_layers,
                    steps, batch, lr, device, head="l2") -> Dict[str, float]:
    N, C, T, H, W = train_feat.shape
    pred = ContinuousStagePredictor(C, T, H, W, dim=dim, n_heads=n_heads, n_layers=n_layers, head=head).to(device)
    opt = torch.optim.Adam(pred.parameters(), lr=lr)
    train_feat = train_feat.to(device)
    val_feat = val_feat.to(device)
    g = torch.Generator(device=device).manual_seed(0)
    pred.train()
    for step in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g, device=device)
        xb = train_feat[idx]
        out = pred(xb)
        loss = predictive_loss(out, xb, head)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    pred.eval()
    with torch.no_grad():
        # evaluate in chunks to bound memory
        chunks = []
        for s in range(0, val_feat.shape[0], 256):
            chunks.append(pred(val_feat[s:s + 256]))
        vout = torch.cat(chunks, 0)
        ve = variance_explained(vout, val_feat)
        ve["train_loss"] = float(loss.detach())
        ve["erank"] = effective_rank(val_feat)
        ve["C_dim"] = float(C)
    return ve


def run_g0(cfg: Config, ckpt: str, *, steps=600, batch=128, lr=3e-4, dim=128, n_heads=4, n_layers=3,
           max_clips=None, device=None) -> Dict[str, Dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(ckpt, device)
    common = dict(batch_size=cfg.optim.batch_size, num_workers=0)
    tr = DataLoader(dataset_from_cfg(cfg, "train", max_clips=max_clips), shuffle=False, **common)
    va = DataLoader(dataset_from_cfg(cfg, "val", max_clips=max_clips), shuffle=False, **common)
    t0 = time.time()
    train_feat = cache_stage_features(tok, tr, device)
    val_feat = cache_stage_features(tok, va, device)
    names = ["bot", "mid", "top"]
    out: Dict[str, Dict[str, float]] = {}
    for s in range(tok.topology.n_stages()):
        res = fit_stage_probe(train_feat[s], val_feat[s], dim=dim, n_heads=n_heads, n_layers=n_layers,
                              steps=steps, batch=batch, lr=lr, device=device)
        out[names[s]] = res
    out["_meta"] = {"sec": time.time() - t0, "n_train": float(train_feat[0].shape[0]),
                    "n_val": float(val_feat[0].shape[0])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/tokenizer/last.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    from hvp.config import apply_cli_overrides
    cfg = Config.from_dict(apply_cli_overrides({}, args.set))
    cfg.validate()
    res = run_g0(cfg, args.ckpt, steps=args.steps, max_clips=args.max_clips)
    print(f"\n=== G0 continuous predictability (held-out cross-split) ===")
    print(f"  clips: train={res['_meta']['n_train']:.0f} val={res['_meta']['n_val']:.0f} "
          f"  fit {res['_meta']['sec']:.1f}s")
    print(f"  {'stage':5s} {'ve_all':>8s} {'ve_changed':>11s} {'mse_pred':>9s} {'mse_persist':>12s} "
          f"{'frac_chg':>9s} {'erank':>6s} {'Cdim':>5s}")
    for st in ["bot", "mid", "top"]:
        r = res[st]
        print(f"  {st:5s} {r['ve_all']:8.3f} {r['ve_changed']:11.3f} {r['mse_pred']:9.4f} "
              f"{r['mse_persist']:12.4f} {r['frac_changed']:9.3f} {r['erank']:6.1f} {r['C_dim']:5.0f}")
    mid = res["mid"]
    verdict = "PASS" if mid["ve_changed"] >= 0.05 else "FAIL"
    print(f"\n  GATE G0 (MID ve_changed >= 0.05): {verdict}  (ve_changed={mid['ve_changed']:.3f})")


if __name__ == "__main__":
    main()
