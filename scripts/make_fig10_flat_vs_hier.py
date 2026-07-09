"""fig10_flat_vs_hier.png — flat vs hierarchical, all measured metrics (two-panel).

Panel A: grouped bars, flat vs hier ALL fg-MSE at each matched token budget, seed-mean +
min/max whiskers (2 seeds at 896/1152/3072; single seed at 512). Persistence floor drawn.
Panel B: paired Δ(flat − hier) per budget, per seed, with 95% cluster-bootstrap CIs; the
win band (Δ>0 = hier better) shaded, sign flip at both ends.

Data: rhvd_eval_*.npz (bars) + compare logs frontier_{bottom,low,top}.log + the 2026-07-04/05
headline compares (deltas + CIs). All two-seed as of 2026-07-08 except 512 (seed 0).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- measured ALL fg-MSE (diffusion head), per budget: seed values from the npz ----
BUD = [512, 896, 1152, 3072]
flat = {512: [0.2224], 896: [0.2106, 0.2102], 1152: [0.2038, 0.2044], 3072: [0.1564, 0.1580]}
hier = {512: [0.2260], 896: [0.1706, 0.1744], 1152: [0.1848, 0.1690], 3072: [0.1695, 0.1687]}
# paired Δ(flat − hier) per seed, with 95% cluster CIs (from the compare logs)
delta = {  # budget: list of (Δ, lo, hi) per seed
    512:  [(-0.0036, -0.0063, -0.0010)],
    896:  [(+0.0399, +0.0299, +0.0501), (+0.0358, +0.0267, +0.0454)],
    1152: [(+0.0189, +0.0121, +0.0264), (+0.0354, +0.0267, +0.0447)],
    3072: [(-0.0131, -0.0201, -0.0069), (-0.0107, -0.0163, -0.0058)],
}
PERSIST = 0.2552
C_FLAT, C_HIER, C_WIN, C_LOSS = "#264653", "#e76f51", "#2a9d8f", "#9b2226"


def mm(vals):
    a = np.array(vals)
    return a.mean(), a.min(), a.max()


fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})

# ---------- Panel A: grouped bars ----------
x = np.arange(len(BUD)); w = 0.38
for i, b in enumerate(BUD):
    fm, flo, fhi = mm(flat[b]); hm, hlo, hhi = mm(hier[b])
    ax[0].bar(x[i] - w/2, fm, w, color=C_FLAT, zorder=2,
              label="flat (full-res window)" if i == 0 else None)
    ax[0].bar(x[i] + w/2, hm, w, color=C_HIER, zorder=2,
              label="hierarchical (coarse + recent fine)" if i == 0 else None)
    if len(flat[b]) > 1:
        ax[0].plot([x[i]-w/2, x[i]-w/2], [flo, fhi], color="k", lw=1.1, zorder=3)
        ax[0].plot([x[i]+w/2, x[i]+w/2], [hlo, hhi], color="k", lw=1.1, zorder=3)
    winner = "hier" if hm < fm else "flat"
    ax[0].text(x[i], max(fm, hm) + 0.004,
               f"{'HIER' if winner=='hier' else 'FLAT'} wins", ha="center", fontsize=8.5,
               color=C_WIN if winner == "hier" else C_LOSS, fontweight="bold")
    ax[0].text(x[i]-w/2, fm - 0.010, f"{fm:.4f}", ha="center", fontsize=7.5, color="w", zorder=4)
    ax[0].text(x[i]+w/2, hm - 0.010, f"{hm:.4f}", ha="center", fontsize=7.5, color="w", zorder=4)
ax[0].axhline(PERSIST, color="0.4", ls=":", lw=1.0, zorder=1)
ax[0].text(len(BUD)-1, PERSIST + 0.001, "persistence 0.2552", ha="right", va="bottom",
           fontsize=8, color="0.35")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{b}\ntok" for b in BUD])
ax[0].set_ylabel("ALL fg-MSE, 152-frame rollout (lower is better)")
ax[0].set_xlabel("matched context-token budget")
ax[0].set_ylim(0.14, 0.268)
ax[0].set_title("A. Flat vs hierarchical at matched budgets\n"
                "(bars = seed-mean; whisker = seed min–max where 2 seeds)", fontsize=10)
ax[0].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax[0].grid(axis="y", alpha=0.25)

# ---------- Panel B: paired deltas ----------
ax[1].axhspan(0, 0.06, color=C_WIN, alpha=0.07, zorder=0)
ax[1].axhline(0, color="0.3", lw=1.0, zorder=1)
ax[1].text(0.02, 0.052, "Δ > 0 : hierarchy better", transform=ax[1].get_yaxis_transform(),
           color="#1d7268", fontsize=8.5, style="italic", va="top")
for i, b in enumerate(BUD):
    seeds = delta[b]
    offs = np.linspace(-0.12, 0.12, len(seeds)) if len(seeds) > 1 else [0.0]
    for j, (d, lo, hi) in enumerate(seeds):
        col = C_WIN if d > 0 else C_LOSS
        ax[1].errorbar(x[i] + offs[j], d, yerr=[[d - lo], [hi - d]], fmt="o", ms=6,
                       color=col, capsize=4, lw=1.6, zorder=3,
                       label=("hier win (CI>0)" if d > 0 else "flat win (CI<0)")
                       if (i in (1, 3) and j == 0) else None)
    dm = np.mean([d for d, _, _ in seeds])
    ax[1].text(x[i], (max(hi for _, _, hi in seeds) if dm > 0 else
                      min(lo for _, lo, _ in seeds)) + (0.004 if dm > 0 else -0.006),
               f"{'2-seed' if len(seeds) > 1 else '1-seed'}", ha="center", fontsize=7.5,
               color="0.4")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{b}\ntok" for b in BUD])
ax[1].set_ylabel("paired Δ(flat − hier), ALL  [95% cluster-bootstrap CI]")
ax[1].set_xlabel("matched context-token budget")
ax[1].set_ylim(-0.028, 0.062)
ax[1].axvline(0.5, color="0.7", ls="--", lw=0.8); ax[1].axvline(2.5, color="0.7", ls="--", lw=0.8)
ax[1].text(1.5, -0.024, "WIN BAND\n(span ≥ loop, window < loop)", ha="center", fontsize=8,
           color="#1d7268", fontweight="bold")
ax[1].text(0, -0.024, "below\nthreshold", ha="center", fontsize=7.5, color=C_LOSS)
ax[1].text(3, -0.024, "window\nsaturated", ha="center", fontsize=7.5, color=C_LOSS)
ax[1].set_title("B. Paired advantage, per seed\n"
                "(both band edges + interior now two-seed; 512 single-seed)", fontsize=10)
ax[1].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax[1].grid(axis="y", alpha=0.25)

fig.suptitle("Flat vs hierarchical at matched token budgets — hierarchy wins the band where its "
             "coarse span crosses the memory horizon\n(loopmom, diffusion head; 2 seeds at "
             "896/1152/3072 as of 2026-07-08)", fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = r"C:\Users\leoni\Documents\projects\hierarchical-video-prediction\docs\figures\fig10_flat_vs_hier.png"
fig.savefig(out, dpi=180)
print("saved", out)
