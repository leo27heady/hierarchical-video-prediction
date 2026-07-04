"""Claim-1 evidence figure: degradation-vs-horizon curves for the loopmom rollout campaign.

Panels: (a) fg-MSE vs horizon (all records, empty-GT-fg excluded), (b) HARD re-entry bin only
(novel-fg-at-6 > 0.1) — the memory signal, (c) foreground sharpness vs horizon (batch-mean
series; the npz `sharp` field is chunk-level, so no CI bands there by design).
Bands: 95% clip-cluster bootstrap CIs. Persistence floor from the base arm's pers_fg.

  venv/Scripts/python.exe scripts/plot_horizon_curves.py
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from hvp_rev.rhvd import _fg_empty_mask                            # noqa: E402

TEST, WARM = "data/loopmom_test_500x200.npy", 48
ARMS = [  # (npz arm, label, ctx tokens, color, ls)
    ("hier-long",   "hier-long (48c+6f, 1152 tok, diff)",  "#d62728", "-"),
    ("flat-budget", "flat-budget (18f, 1152 tok, diff)",   "#1f77b4", "-"),
    ("flat-long",   "flat-long (48f, 3072 tok, diff)",     "#7f7f7f", "--"),
    ("reg-long",    "reg-long (48c+6f, 1152 tok, reg)",    "#ff7f0e", "-"),
    ("reg-flat",    "reg-flat (18f, 1152 tok, reg)",       "#2ca02c", "-"),
    ("hier-short",  "hier-short (6c+6f, 480 tok, diff)",   "#9467bd", ":"),
]
HBINS = [(1 + 8 * i, 8 + 8 * i) for i in range(19)]                # 8-frame bins to h=152


def cluster_mean_ci(vals, clips, n_boot=4000, seed=0):
    rng = np.random.RandomState(seed)
    uc = np.unique(clips)
    s = np.array([vals[clips == c].sum() for c in uc])
    n = np.array([(clips == c).sum() for c in uc], np.float64)
    pick = rng.randint(0, len(uc), (n_boot, len(uc)))
    m = s[pick].sum(1) / n[pick].sum(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def curve(z, mask_extra=None):
    empty = _fg_empty_mask(z["clip"], z["h"], TEST, WARM)
    base = ~empty if mask_extra is None else (~empty & mask_extra)
    xs, ys, los, his = [], [], [], []
    for lo, hi in HBINS:
        m = base & (z["h"] >= lo) & (z["h"] <= hi)
        if m.sum() < 20:
            continue
        mu, cl, ch = cluster_mean_ci(z["fg"][m], z["clip"][m])
        xs.append((lo + hi) / 2); ys.append(mu); los.append(cl); his.append(ch)
    return np.array(xs), np.array(ys), np.array(los), np.array(his)


def sharp_curve(z):
    xs, ys = [], []
    for lo, hi in HBINS:
        m = (z["h"] >= lo) & (z["h"] <= hi)
        if m.sum() < 20:
            continue
        xs.append((lo + hi) / 2); ys.append(float(z["sharp"][m].mean()))
    return np.array(xs), np.array(ys)


def main():
    data = {a: np.load(f"logs_rhcp/rhvd_eval_{a}_s0.npz") for a, *_ in ARMS}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    zb = data["hier-long"]

    for ax, extra, title in (
            (axes[0], None, "(a) fg-MSE vs horizon — all frames"),
            (axes[1], zb["hard"] > 0.1, "(b) HARD re-entry frames only (novel-fg-at-6 > 0.1)")):
        for a, lab, col, ls in ARMS:
            z = data[a]
            me = None if extra is None else (z["hard"] > 0.1)
            xs, ys, lo, hi = curve(z, me)
            ax.plot(xs, ys, ls, color=col, label=lab, lw=1.8)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
        # persistence floor (paired records; same for every arm)
        empty = _fg_empty_mask(zb["clip"], zb["h"], TEST, WARM)
        basem = ~empty if extra is None else (~empty & (zb["hard"] > 0.1))
        xs, ys = [], []
        for lo_, hi_ in HBINS:
            m = basem & (zb["h"] >= lo_) & (zb["h"] <= hi_)
            if m.sum() >= 20:
                xs.append((lo_ + hi_) / 2); ys.append(float(zb["pers_fg"][m].mean()))
        ax.plot(xs, ys, "-.", color="k", label="persistence floor", lw=1.2, alpha=0.7)
        ax.set_xlabel("rollout horizon h (frames after 48-frame GT warmup)")
        ax.set_ylabel("foreground MSE")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)

    for a, lab, col, ls in ARMS:
        xs, ys = sharp_curve(data[a])
        axes[2].plot(xs, ys, ls, color=col, label=lab, lw=1.8)
    axes[2].axhline(0.2412, color="k", ls=":", lw=1.0, alpha=0.6)
    axes[2].text(4, 0.244, "AE round-trip ceiling", fontsize=8)
    axes[2].set_xlabel("rollout horizon h")
    axes[2].set_ylabel("fg sharpness (batch-mean series)")
    axes[2].set_title("(c) sharpness retention vs horizon", fontsize=10)
    axes[2].grid(alpha=0.25)
    axes[0].legend(fontsize=7.2, loc="lower right")

    fig.suptitle("Loopmom rollout campaign (testbed A, seed 0): degradation over 152 AR frames — "
                 "matched-budget hierarchy (red vs blue) slows degradation exactly on the memory-"
                 "dependent frames", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = "logs_rhcp/fig_horizon_curves_loopmom.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
