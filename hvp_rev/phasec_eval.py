"""Phase-C factor-metric rollout eval (pre-registered METRIC MANDATE): re-runs the AR rollout of
a trained mmL-* arm on the long MMNIST test clips, decoding every predicted frame and scoring
the CW-VAE-facing factor metrics (identity retention / trajectory error / digit count) alongside
fg-MSE. Writes `logs_rhcp/phasec_{arm}_s{seed}.npz` with per-frame records: h, fg, id_match,
id_scored, traj_err, count_dev, clip.

Duplicates the rollout buffer loop of `rhvd.rollout_eval` deliberately (that function is under
eval-code freeze and does not capture frames); CRN seeding matches it so records pair with the
standard eval npz. Identity anchors: the digits of the LAST GT warm frame, classified once per
clip (anchor_used=False if detection there fails — disclosed via id_scored).

  venv/Scripts/python.exe -u -m hvp_rev.phasec_eval --arm mmL-hier --seed 0 \
      --train_path data/mmnistL_train_3000x100.npy --test_path data/mmnistL_test_500x348.npy \
      --ae_path logs_rhcp/hier_ae_mmnistL.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hvp_rev import hier_convae
from hvp_rev.mmnist_metrics import detect_digits, classify, frame_metrics, load_classifier
from hvp_rev.rhvd import (ARMS, EMA, WIN, WindowDiT, WindowDiffusion, WindowRegression,
                          arm_cfg, ctx_reach)
from hvp_rev.step1_mech import ActionHierData, decode01, fg_mse_px_frames

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_invalid_legacy", action="store_true",
                    help="for forensic reproduction only; never use for scientific claims")
    ap.add_argument("--arm", required=True, choices=[a for a in ARMS if a.startswith("mmL-")])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--n_clips", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--warm", type=int, default=48)
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--ae_path", required=True)
    args = ap.parse_args()

    if not args.allow_invalid_legacy:
        raise SystemExit(
            "Legacy Phase C evaluator is scientifically blocked: order-based identity, "
            "many-to-one trajectory matching and survivorship coverage are invalid. "
            "Use the persistent-object Phase C v2 protocol."
        )

    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = arm_cfg(args.arm)
    assert n_ctx_t == 0 and n_hist_f == 0 and stride_c == 1, "Phase-C arms are plain 2-level"
    ae = hier_convae.load(args.ae_path)
    data = ActionHierData(ae, args.train_path, args.train_path.replace(".npy", "_actions.npy"),
                          n_ctx=6, k=4)
    test = ActionHierData(ae, args.test_path, args.test_path.replace(".npy", "_actions.npy"),
                          n_ctx=6, k=4, n_val=250, stats=data.stats)
    net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
    wrap = WindowDiffusion(net) if mode == "diff" else WindowRegression(net)
    ema = EMA(net)
    ckpt = f"logs_rhcp/rhvd_{args.arm}_s{args.seed}_t{args.steps}.pt"
    st = torch.load(ckpt, map_location=DEV, weights_only=False)
    ema.shadow.load_state_dict(st["ema"])
    print(f"[phasec_eval] loaded {ckpt} @ step {st['step']}", flush=True)
    wrap_eval = WindowDiffusion(ema.shadow, wrap.Ttr) if mode == "diff" else WindowRegression(ema.shadow)
    clf = load_classifier()

    n_frames = ctx_reach(n_ctx_c, n_ctx_f)
    WARM, T = args.warm, test.T
    assert WARM >= n_frames
    n_steps_roll = (T - WARM) // WIN
    clips = test.val_idx[:args.n_clips]
    recs = {k: [] for k in ("h", "fg", "id_match", "id_scored", "traj_err", "count_dev", "clip")}
    for c0 in range(0, len(clips), args.chunk):
        cs = clips[c0:c0 + args.chunk]
        B = len(cs)
        buf_c = (test.zc[cs, WARM - n_frames:WARM].clone().to(DEV) if n_ctx_c > 0
                 else torch.zeros(B, 1, 16, 4, 4, device=DEV))
        buf_f = test.zf[cs, WARM - n_frames:WARM].clone().to(DEV)
        # identity anchors from the LAST GT warm frame
        anchors = []
        for ci in cs:
            g = test.clips_u8[ci, WARM - 1].astype(np.float32) / 255.0
            det = detect_digits(g)
            anchors.append((classify(clf, det), len(det) > 0))
        for srn in range(n_steps_roll):
            tj = WARM + srn * WIN
            torch.manual_seed(100000 + 1000 * (c0 // args.chunk) + srn)   # CRN parity with rhvd
            acts = torch.from_numpy(test.acts[cs, tj:tj + WIN].astype(np.int64)).to(DEV)
            ctx_c = buf_c[:, -n_ctx_c:] if n_ctx_c > 0 else buf_c[:, :0]
            ctx_f = buf_f[:, -n_ctx_f:]
            yc, yf = wrap_eval.sample(ctx_c, ctx_f, acts, n_steps=args.nfe)
            if n_ctx_c > 0:
                buf_c = torch.cat([buf_c, yc], 1)[:, -n_frames:]
            buf_f = torch.cat([buf_f, yf], 1)[:, -n_frames:]
            for j in range(WIN):
                tf = tj + j
                p01 = decode01(ae, yf[:, j])
                gt01 = torch.from_numpy(
                    test.clips_u8[cs, tf].astype(np.float32) / 255.0).unsqueeze(1).to(DEV)
                fg = fg_mse_px_frames(p01, gt01)
                pnp = p01.squeeze(1).cpu().numpy()
                gnp = gt01.squeeze(1).cpu().numpy()
                for bi, ci in enumerate(cs):
                    m = frame_metrics(clf, pnp[bi], gnp[bi], *anchors[bi])
                    recs["h"].append(tf - WARM + 1)
                    recs["fg"].append(float(fg[bi]))
                    recs["clip"].append(int(ci))
                    for k in ("id_match", "id_scored", "traj_err", "count_dev"):
                        recs[k].append(float(m[k]))
        print(f"  [phasec {args.arm}] clips {c0}..{c0 + B - 1} done", flush=True)
    out = {k: np.asarray(v) for k, v in recs.items()}
    path = f"logs_rhcp/phasec_{args.arm}_s{args.seed}.npz"
    np.savez(path, **out)
    hs = out["h"]
    for lo, hi in ((1, 60), (61, 150), (151, 300)):
        m = (hs >= lo) & (hs <= hi)
        sc = out["id_scored"][m] > 0
        idm = np.nanmean(out["id_match"][m][sc]) if sc.any() else float("nan")
        tr = np.nanmean(out["traj_err"][m])
        print(f"  h{lo}-{hi}: fg {out['fg'][m].mean():.4f}  id_match {idm:.3f} "
              f"(scored {sc.mean()*100:.0f}%)  traj {tr:.1f}px  count_dev {out['count_dev'][m].mean():.2f}")
    print(f"[phasec_eval] saved {path}: {len(out['fg'])} records")


if __name__ == "__main__":
    main()
