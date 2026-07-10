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
# 1792 (R-F) and 2432 (R-G) are seed-0 bracket points; the rest are two-seed.
BUD = [512, 896, 1152, 1792, 2432, 3072]
flat = {512: [0.2224, 0.2206], 896: [0.2106, 0.2102], 1152: [0.2038, 0.2044],
        1792: [0.1876], 2432: [0.1703], 3072: [0.1564, 0.1580]}
hier = {512: [0.2260, 0.2257], 896: [0.1706, 0.1744], 1152: [0.1848, 0.1690],
        1792: [0.1799], 2432: [0.1820], 3072: [0.1695, 0.1687]}
# paired Δ(flat − hier) per seed, with 95% cluster CIs (from the compare logs)
delta = {  # budget: list of (Δ, lo, hi) per seed
    512:  [(-0.0036, -0.0063, -0.0010), (-0.0051, -0.0079, -0.0024)],
    896:  [(+0.0399, +0.0299, +0.0501), (+0.0358, +0.0267, +0.0454)],
    1152: [(+0.0189, +0.0121, +0.0264), (+0.0354, +0.0267, +0.0447)],
    1792: [(+0.0077, +0.0016, +0.0142)],
    2432: [(-0.0116, -0.0162, -0.0073)],
    3072: [(-0.0131, -0.0201, -0.0069), (-0.0107, -0.0163, -0.0058)],
}
PERSIST = 0.2552
C_FLAT, C_HIER, C_WIN, C_LOSS = "#264653", "#e76f51", "#2a9d8f", "#9b2226"


def mm(vals):
    a = np.array(vals)
    return a.mean(), a.min(), a.max()


fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})

# ---------- Panel A: grouped bars + the hier PLATEAU band ----------
# From 896 tokens on, every hier arm shares the same protocol-capped 48-frame span; extra budget
# buys only ~94%-redundant recent fine frames, so the hier side is a PLATEAU, not a trend. The
# envelope [0.1666, 0.1848] = hier seed extremes ≥896 incl. the 60k regime point (hier-long-60k
# 0.1666, R-C). Regime note: the 3072 pair trained B8×60k (hatched); hier arms are
# regime-sensitive (−0.018 at 2× steps, R-C) while flat arms are not (−0.0006).
PLAT_LO, PLAT_HI = 0.1666, 0.1848
REGIME_60K = {3072}
x = np.arange(len(BUD)); w = 0.38
ax[0].fill_between([x[1] - 0.55, x[-1] + 0.55], PLAT_LO, PLAT_HI, color=C_HIER, alpha=0.12,
                   zorder=1, lw=0)
ax[0].annotate("hier plateau ≈0.17 (span capped at the 48-frame loop;\n"
               "seed+regime envelope — a plateau, not a trend)",
               xy=(1.55, PLAT_HI), xytext=(-0.45, 0.2415), fontsize=7.5, color="#a34a33",
               style="italic", ha="left", va="bottom",
               arrowprops=dict(arrowstyle="->", color="#a34a33", lw=0.9))
for i, b in enumerate(BUD):
    fm, flo, fhi = mm(flat[b]); hm, hlo, hhi = mm(hier[b])
    hatch = "//" if b in REGIME_60K else None
    ax[0].bar(x[i] - w/2, fm, w, color=C_FLAT, zorder=2, hatch=hatch,
              label="flat (full-res window)" if i == 0 else None)
    ax[0].bar(x[i] + w/2, hm, w, color=C_HIER, zorder=2, hatch=hatch,
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
ax[0].text(-0.45, PERSIST + 0.001, "persistence 0.2552", ha="left", va="bottom",
           fontsize=8, color="0.35")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{b}\ntok" for b in BUD])
ax[0].set_ylabel("ALL fg-MSE, 152-frame rollout (lower is better)")
ax[0].set_xlabel("matched context-token budget")
ax[0].set_ylim(0.14, 0.268)
ax[0].set_title("A. Flat vs hierarchical at matched budgets — flat descends through a hier "
                "PLATEAU\n(bars = seed-mean; whiskers = seed min–max; hatched = B8×60k regime, "
                "others B16×30k)", fontsize=9.5)
ax[0].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax[0].grid(axis="y", alpha=0.25)

# ---------- Panel B: paired deltas ----------
ax[1].axhspan(0, 0.06, color=C_WIN, alpha=0.07, zorder=0)
ax[1].axhline(0, color="0.3", lw=1.0, zorder=1)
ax[1].text(0.02, 0.052, "Δ > 0 : hierarchy better", transform=ax[1].get_yaxis_transform(),
           color="#1d7268", fontsize=8.5, style="italic", va="top")
_seen_lbl = set()
for i, b in enumerate(BUD):
    seeds = delta[b]
    offs = np.linspace(-0.12, 0.12, len(seeds)) if len(seeds) > 1 else [0.0]
    for j, (d, lo, hi) in enumerate(seeds):
        col = C_WIN if d > 0 else C_LOSS
        lbl = "hier win (CI>0)" if d > 0 else "flat win (CI<0)"
        show = lbl if lbl not in _seen_lbl else None
        _seen_lbl.add(lbl)
        ax[1].errorbar(x[i] + offs[j], d, yerr=[[d - lo], [hi - d]], fmt="o", ms=6,
                       color=col, capsize=4, lw=1.6, zorder=3, label=show)
    dm = np.mean([d for d, _, _ in seeds])
    ax[1].text(x[i], (max(hi for _, _, hi in seeds) if dm > 0 else
                      min(lo for _, lo, _ in seeds)) + (0.004 if dm > 0 else -0.006),
               f"{'2-seed' if len(seeds) > 1 else '1-seed'}", ha="center", fontsize=7.5,
               color="0.4")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{b}\ntok" for b in BUD])
ax[1].set_ylabel("paired Δ(flat − hier), ALL  [95% cluster-bootstrap CI]")
ax[1].set_xlabel("matched context-token budget")
ax[1].set_ylim(-0.028, 0.062)
# band spans indices 1–3 (896/1152/1792 hier-wins); edges between 512|896 and 1792|2432
ax[1].axvline(0.5, color="0.7", ls="--", lw=0.8); ax[1].axvline(3.5, color="0.7", ls="--", lw=0.8)
ax[1].text(2.0, -0.024, "WIN BAND (span ≥ loop, window < loop)", ha="center", fontsize=8,
           color="#1d7268", fontweight="bold")
ax[1].text(0, -0.024, "below\nthreshold", ha="center", fontsize=7.5, color=C_LOSS)
ax[1].text(4.5, -0.024, "window\nsaturated", ha="center", fontsize=7.5, color=C_LOSS)
ax[1].set_title("B. Paired advantage, per seed — upper edge bracketed to (1792, 2432)\n"
                "(896/1152/3072 two-seed; 1792/2432 bracket seed 0)", fontsize=9.5)
ax[1].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax[1].grid(axis="y", alpha=0.25)

fig.suptitle("Flat vs hierarchical at matched token budgets — hierarchy wins the band where its "
             "coarse span crosses the memory horizon\n(loopmom, diffusion head; 6 budgets, both "
             "band edges measured, as of 2026-07-10)", fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = r"C:\Users\leoni\Documents\projects\hierarchical-video-prediction\docs\figures\fig10_flat_vs_hier.png"
fig.savefig(out, dpi=180)
print("saved", out)
