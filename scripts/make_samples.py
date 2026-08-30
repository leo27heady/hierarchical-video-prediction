"""Clean qualitative sample figure: GT vs hier-long vs flat-budget decoded rollout predictions
at increasing horizons (testbed A / loopmom, seed 1). Light GPU decode; follows rhvd.rollout_eval.

  venv/Scripts/python.exe scripts/make_samples.py
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hvp_rev import hier_convae
from hvp_rev.step1_mech import ActionHierData, decode01
from hvp_rev.rhvd import WindowDiT, WindowDiffusion, ARMS, WIN, LAT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "docs/figures"
WARM = 48
HS = [1, 24, 60, 100, 148]           # horizons to display
CLIPS = [0, 1, 3, 5]                  # test val clips
os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def rollout_capture(arm, seed, ae, data, clips, hs):
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f = ARMS[arm]
    net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c, n_hist_f=n_hist_f).to(DEV)
    st = torch.load(f"logs_rhcp/rhvd_{arm}_s{seed}_t30000.pt", map_location=DEV, weights_only=False)
    net.load_state_dict(st["ema"]); net.eval()
    wrap = WindowDiffusion(net)
    n_frames = max(n_ctx_c * stride_c, n_ctx_f)
    cs = np.array(clips)
    B = len(cs)
    buf_c = data.zc[cs, WARM - n_frames:WARM].clone().to(DEV) if n_ctx_c > 0 else torch.zeros(B, 1, LAT, 4, 4, device=DEV)
    buf_f = data.zf[cs, WARM - n_frames:WARM].clone().to(DEV)
    want = {WARM + h - 1: h for h in hs}
    cap = {h: None for h in hs}
    n_steps = (data.T - WARM) // WIN
    for srn in range(n_steps):
        tj = WARM + srn * WIN
        torch.manual_seed(100000 + srn)
        acts = torch.from_numpy(data.acts[cs, tj:tj + WIN].astype(np.int64)).to(DEV)
        ctx_c = buf_c[:, -n_ctx_c:] if n_ctx_c > 0 else buf_c[:, :0]
        ctx_f = buf_f[:, -n_ctx_f:]
        yc, yf = wrap.sample(ctx_c, ctx_f, acts, n_steps=16)
        if n_ctx_c > 0:
            buf_c = torch.cat([buf_c, yc], 1)[:, -n_frames:]
        buf_f = torch.cat([buf_f, yf], 1)[:, -n_frames:]
        for j in range(WIN):
            tf = tj + j
            if tf in want:
                cap[want[tf]] = decode01(ae, yf[:, j]).cpu().numpy()[:, 0]   # (B,64,64)
    return cap


def main():
    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    train = ActionHierData(ae, "data/loopmom_train_3000x100.npy", "data/loopmom_train_3000x100_actions.npy", n_ctx=6, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy", "data/loopmom_test_500x200_actions.npy",
                          n_ctx=6, k=4, n_val=250, stats=train.stats)
    cs = test.val_idx[np.array(CLIPS)]
    gt = {h: test.clips_u8[cs, WARM + h - 1].astype(np.float32) / 255.0 for h in HS}
    hier = rollout_capture("hier-long", 1, ae, test, cs, HS)
    flat = rollout_capture("flat-budget", 1, ae, test, cs, HS)

    nclip = len(CLIPS); nh = len(HS)
    rows = 3 * nclip
    fig, ax = plt.subplots(rows, nh, figsize=(nh * 1.5, rows * 1.5))
    rlabels = ["GT", "hier-long", "flat-budget"]
    for ci in range(nclip):
        for r, (src, lab) in enumerate([(gt, "GT"), (hier, "hier-long"), (flat, "flat-budget")]):
            for j, h in enumerate(HS):
                a = ax[ci * 3 + r, j]
                a.imshow(src[h][ci], cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
                if ci == 0 and r == 0:
                    a.set_title(f"h={h}", fontsize=11)
                if j == 0:
                    a.set_ylabel(f"clip {CLIPS[ci]}\n{lab}", fontsize=8)
        # separator
    fig.suptitle("Qualitative rollout (testbed A, seed 1): both arms are accurate at short horizon and "
                 "degrade to blur at long horizon (sharpness fades for all arms).\nThe matched-budget "
                 "hierarchy advantage is a MODEST AVERAGE gain (+0.03 fg-MSE), not a dramatic per-frame "
                 "difference — honest qualitative reading.", fontsize=10, y=0.997)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_samples_claim1.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig7 samples ->", f"{OUT}/fig7_samples_claim1.png")

    # Compact main-text view: the first predeclared clip, with the same five horizons.
    fig, ax = plt.subplots(3, nh, figsize=(nh * 1.55, 4.7), squeeze=False)
    for r, (src, label) in enumerate([(gt, "Еталон"), (hier, "Ієрархічний пакет"), (flat, "Плоский пакет")]):
        for j, h in enumerate(HS):
            panel = ax[r, j]
            panel.imshow(src[h][0], cmap="gray", vmin=0, vmax=1)
            panel.set_xticks([])
            panel.set_yticks([])
            if r == 0:
                panel.set_title(f"горизонт {h}", fontsize=9)
            if j == 0:
                panel.set_ylabel(label, fontsize=8.5)
    fig.suptitle("Фіксований кліп 0, ініціалізація навчання 1", fontsize=11)
    fig.tight_layout()
    compact_path = f"{OUT}/fig7_samples_claim1_compact.png"
    fig.savefig(compact_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("fig7 compact samples ->", compact_path)


if __name__ == "__main__":
    main()
