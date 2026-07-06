"""Publication figures for the article + dissertation, generated from the audited npz/logs.
CPU-only (matplotlib); no GPU, no re-training. Writes to docs/figures/.

  venv/Scripts/python.exe scripts/make_figures.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
EV = "logs_rhcp/rhvd_eval_{}_s{}.npz"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def load(arm, s=0):
    return np.load(EV.format(arm, s))


def cci(delta, clips, nb=2000, seed=0):
    rng = np.random.RandomState(seed); uc = np.unique(clips)
    s = np.array([delta[clips == c].sum() for c in uc]); n = np.array([(clips == c).sum() for c in uc], float)
    pick = rng.randint(0, len(uc), (nb, len(uc))); m = s[pick].sum(1) / n[pick].sum(1)
    return delta.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def hcurve(z, field="fg", hmax=152, w=4):
    h = z["h"]; v = z[field]; xs = np.arange(1, hmax + 1)
    ys = np.array([v[(h >= max(1, x - w)) & (h <= x + w)].mean() if ((h >= max(1, x - w)) & (h <= x + w)).any() else np.nan for x in xs])
    return xs, ys


# ============================================================ FIG 1 — Claim 1 horizon curves
def fig1():
    arms = [("flat-long", "flat-long (48f, 3072 tok)", "#1a9850", "-"),
            ("reg-long", "reg-long (48c+6f, reg)", "#7b3294", "--"),
            ("hier-long", "hier-long (48c+6f, THE system)", "#d73027", "-"),
            ("reg-flat", "reg-flat (18f, reg)", "#c2a5cf", "--"),
            ("flat-budget", "flat-budget (18f, matched ctrl)", "#4575b4", "-"),
            ("hier-short", "hier-short (6c+6f)", "#999999", ":")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for name, lab, col, ls in arms:
        z = load(name)
        for j, (fld, hm) in enumerate([("fg", 152), ("fg", 152), ("sharp", 152)]):
            if j == 1:  # HARD only
                m = z["hard"] > 0.1; zz = {"h": z["h"][m], "fg": z["fg"][m]}
                xs, ys = hcurve(zz, "fg")
            else:
                xs, ys = hcurve(z, fld)
            ax[j].plot(xs, ys, color=col, ls=ls, lw=1.8, label=lab if j == 0 else None)
    pf = load("hier-long")["pers_fg"].mean()
    ax[0].axhline(pf, color="k", ls="-.", lw=1, label="persistence floor")
    ax[1].axhline(load("hier-long")["pers_fg"][load("hier-long")["hard"] > 0.1].mean(), color="k", ls="-.", lw=1)
    ax[2].axhline(0.2412, color="k", ls="-.", lw=1, label="AE round-trip ceiling")
    ax[0].set_title("(a) fg-MSE vs horizon — all frames"); ax[0].set_ylabel("fg-MSE")
    ax[1].set_title("(b) fg-MSE vs horizon — HARD re-entry")
    ax[2].set_title("(c) sharpness vs horizon (batch-mean)"); ax[2].set_ylabel("fg-sharpness")
    for a in ax: a.set_xlabel("rollout horizon h (frames after 48 GT warmup)")
    ax[0].legend(fontsize=7.5, loc="upper left"); ax[2].legend(fontsize=8, loc="upper right")
    fig.suptitle("Claim 1 — matched-budget hierarchy (hier-long) separates on memory-dependent horizons; "
                 "flat-long is the 2.7×-token frontier (testbed A / loopmom, seed 0)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_claim1_horizon_curves.png"); plt.close(fig)
    print("fig1 done")


# ============================================================ FIG 2 — Claim 1 two-seed win map
def fig2():
    hb = [(1, 8, "h1-8"), (9, 24, "h9-24"), (25, 60, "h25-60"), (61, 152, "h61+")]
    bb = [(-1, 0.001, "easy"), (0.001, 0.1, "mid"), (0.1, 2, "HARD")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for si, s in enumerate([0, 1]):
        hl, fb = load("hier-long", s), load("flat-budget", s)
        M = np.full((len(bb), len(hb)), np.nan); A = np.empty_like(M, dtype=object)
        for i, (blo, bhi, bn) in enumerate(bb):
            for j, (hlo, hhi, hn) in enumerate(hb):
                m = (hl["h"] >= hlo) & (hl["h"] <= hhi) & (hl["hard"] > blo) & (hl["hard"] <= bhi)
                if m.sum() < 20:
                    A[i, j] = ""; continue
                d = fb["fg"][m] - hl["fg"][m]
                mean, lo, hi = cci(d, hl["clip"][m]); M[i, j] = mean
                v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "")
                A[i, j] = f"{mean:+.3f}\n{v}"
        im = ax[si].imshow(M, cmap="RdYlGn", vmin=-0.05, vmax=0.05, aspect="auto")
        ax[si].set_xticks(range(len(hb))); ax[si].set_xticklabels([h[2] for h in hb])
        ax[si].set_yticks(range(len(bb))); ax[si].set_yticklabels([b[2] for b in bb])
        for i in range(len(bb)):
            for j in range(len(hb)):
                ax[si].text(j, i, A[i, j], ha="center", va="center", fontsize=8)
        alld = (fb["fg"] - hl["fg"]); am, alo, ahi = cci(alld, hl["clip"])
        ax[si].set_title(f"seed {s}:  ALL Δ {am:+.4f} [{alo:+.4f},{ahi:+.4f}]", fontsize=10)
        ax[si].set_xlabel("horizon");
    ax[0].set_ylabel("re-entry hardness")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ fg-MSE (flat−hier); >0 = hierarchy better")
    fig.suptitle("Claim 1 two-seed win map — hier-long vs flat-budget at matched 1152 tokens (green = hierarchy wins)", fontsize=11)
    fig.savefig(f"{OUT}/fig2_claim1_winmap.png"); plt.close(fig)
    print("fig2 done")


# ============================================================ FIG 3 — rollout matrix ordering
def fig3():
    arms = [("persistence", 0.2552, 0.0, "#000000"), ("hier-short", None, None, "#999999"),
            ("flat-budget", None, None, "#4575b4"), ("reg-flat", None, None, "#c2a5cf"),
            ("hier-long", None, None, "#d73027"), ("reg-long", None, None, "#7b3294"),
            ("flat-long", None, None, "#1a9850")]
    names, mse, shp, cols = [], [], [], []
    for n, m, sp, c in arms:
        if n == "persistence":
            names.append(n); mse.append(m); shp.append(np.nan); cols.append(c); continue
        z = load(n); names.append(n); mse.append(z["fg"].mean()); shp.append(z["sharp"].mean()); cols.append(c)
    order = np.argsort(mse)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    y = np.arange(len(names))
    ax[0].barh(y, np.array(mse)[order], color=np.array(cols)[order]); ax[0].set_yticks(y)
    ax[0].set_yticklabels(np.array(names)[order]); ax[0].invert_yaxis()
    ax[0].set_xlabel("ALL fg-MSE (152 AR frames)"); ax[0].set_title("(a) rollout error ordering")
    for i, v in enumerate(np.array(mse)[order]):
        ax[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    sh = np.array(shp)[order]
    ax[1].barh(y, sh, color=np.array(cols)[order]); ax[1].set_yticks(y)
    ax[1].set_yticklabels(np.array(names)[order]); ax[1].invert_yaxis()
    ax[1].axvline(0.2412, color="k", ls="-.", lw=1, label="AE ceiling"); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("mean fg-sharpness"); ax[1].set_title("(b) rollout sharpness")
    fig.suptitle("Rollout matrix (testbed A): hierarchy beats matched-token flat; flat-long frontier at 2.7× tokens", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_rollout_matrix.png"); plt.close(fig)
    print("fig3 done")


# ============================================================ FIG 4 — Claim 3 spatial factorial
def fig4():
    pairs = [("flat-budget", "spat18", "diffusion head", "#4575b4", "#d73027"),
             ("reg-flat", "spat18-reg", "regression head", "#4575b4", "#d73027")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (flatn, coarsen, head, c1, c2) in enumerate(pairs):
        fl, co = load(flatn), load(coarsen)
        d = fl["fg"] - co["fg"]; mean, lo, hi = cci(d, fl["clip"])
        ax[k].bar([0, 1], [fl["fg"].mean(), co["fg"].mean()], color=[c1, c2],
                  yerr=[0, 0], width=0.6)
        ax[k].set_xticks([0, 1]); ax[k].set_xticklabels([f"full-res hist\n1152 tok", f"coarse hist\n672 tok (-42%)"])
        ax[k].set_ylabel("ALL fg-MSE"); ax[k].set_ylim(0.19, 0.215)
        verdict = "coarse LOSES" if lo > 0 else ("coarse WINS" if hi < 0 else "coarse ~FREE (null)")
        ax[k].set_title(f"{head}\nΔ(full−coarse) {mean:+.4f} [{lo:+.4f},{hi:+.4f}]\n{verdict}", fontsize=9.5)
        for i, v in enumerate([fl["fg"].mean(), co["fg"].mean()]):
            ax[k].text(i, v + 0.0004, f"{v:.4f}", ha="center", fontsize=9)
    fig.suptitle("Claim 3 — spatial abstraction is lossy compression: a cost for diffusion, ~free for regression "
                 "(window fixed at 18f, testbed A)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_claim3_factorial.png"); plt.close(fig)
    print("fig4 done")


# ============================================================ FIG 5 — H-grid pair-RED crossover (testbed B)
def fig5():
    d1, d0 = load("d1-56-diff"), load("d0-diff-long")
    xs1, y1 = hcurve(d1, "fg", hmax=288, w=6); _, y0 = hcurve(d0, "fg", hmax=288, w=6)
    pers = d0["pers_fg"].mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(xs1, y0, color="#4575b4", lw=1.9, label="d0-diff-long (flat frontier, 3072 tok)")
    ax[0].plot(xs1, y1, color="#d73027", lw=1.9, label="d1-56 (hierarchy, 1360 tok)")
    ax[0].axhline(pers, color="k", ls="-.", lw=1, label="persistence floor")
    ax[0].axvspan(1, 24, color="#1a9850", alpha=0.10); ax[0].axvspan(61, 288, color="#d73027", alpha=0.06)
    ax[0].text(12, ax[0].get_ylim()[0] + 0.005, "d1 WINS\n(memory trustworthy)", fontsize=7.5, ha="center", color="#1a7d30")
    ax[0].text(180, pers + 0.01, "both > persistence (collapse)", fontsize=8, ha="center", color="#7a1a1a")
    ax[0].set_xlabel("rollout horizon h (after 224 GT warmup)"); ax[0].set_ylabel("fg-MSE")
    ax[0].set_title("(a) testbed B rollout: early win → late inversion"); ax[0].legend(fontsize=8, loc="lower right")
    # delta-vs-horizon
    hb = [(1, 8), (9, 24), (25, 60), (61, 120), (121, 192), (193, 288)]
    xs, dm, dlo, dhi = [], [], [], []
    for lo, hi in hb:
        m = (d1["h"] >= lo) & (d1["h"] <= hi)
        dd = d0["fg"][m] - d1["fg"][m]; mm, l, h = cci(dd, d1["clip"][m])
        xs.append((lo + hi) / 2); dm.append(mm); dlo.append(mm - l); dhi.append(h - mm)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].errorbar(xs, dm, yerr=[dlo, dhi], marker="o", capsize=3, color="#333")
    ax[1].fill_between([0, 30], -1, 1, color="#1a9850", alpha=0.08)
    ax[1].fill_between([50, 300], -1, 1, color="#d73027", alpha=0.06)
    ax[1].set_ylim(-0.015, 0.04); ax[1].set_xlabel("rollout horizon h")
    ax[1].set_ylabel("Δ fg-MSE (d0−d1); >0 = hierarchy better")
    ax[1].set_title("(b) hierarchy advantage inverts after h≈24")
    fig.suptitle("H-grid boundary (testbed B, capability regime): hierarchical slot memory pays only while its "
                 "self-fed contents stay trustworthy; capability claim RED (ALL null)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_hgrid_pair_crossover.png"); plt.close(fig)
    print("fig5 done")


# ============================================================ FIG 6 — MMNIST bridge (negative control)
def fig6():
    # values from logs_rhcp/mmnist_bridge_chain.log (10000-seq eval)
    rows = {"mm-flat": (180.91, 0.7625, 0.1320), "mm-hier": (181.29, 0.7607, 0.1389),
            "persistence": (286.28, 0.6528, 0.2533)}
    cols = {"mm-flat": "#4575b4", "mm-hier": "#d73027", "persistence": "#000000"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, (met, lab, hi_good) in enumerate([(0, "MSE (↓)", False), (1, "SSIM (↑)", True), (2, "LPIPS (↓)", False)]):
        names = list(rows); vals = [rows[n][met] for n in names]
        ax[j].bar(range(3), vals, color=[cols[n] for n in names])
        ax[j].set_xticks(range(3)); ax[j].set_xticklabels(names, rotation=15)
        ax[j].set_title(lab)
        for i, v in enumerate(vals): ax[j].text(i, v, f"{v:.3f}" if met else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("T7 OpenSTL/Moving-MNIST bridge (negative control): flat ≈ hier (ΔMSE 0.4 ≈ 0) — "
                 "no off-screen re-entry → no memory horizon → nothing for the coarse level to carry", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_mmnist_bridge.png"); plt.close(fig)
    print("fig6 done")


for f in (fig1, fig2, fig3, fig4, fig5, fig6):
    try:
        f()
    except Exception as e:
        print(f"{f.__name__} FAILED: {type(e).__name__}: {e}")
print("ALL FIGURES ->", OUT)
