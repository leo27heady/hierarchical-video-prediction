"""fig9_frontier.png — token-budget frontier, operating point A (loopmom, seed 0).

Data: Table 4.2 + frontier completion (logs_rhcp/frontier_{top,bottom}.log,
docs/SPATIAL_FACTORIAL.md FRONTIER MEASURED + LOWER EDGE MEASURED 2026-07-07;
logs_rhcp/frontier_low.log: hier-min 0.1706 / flat-min 0.2106, +0.0399 [+0.0299,+0.0501]).
fg-MSE ALL over 152 AR frames.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (tokens, fg-MSE ALL) — diffusion-head frontier curves
flat = [(512, 0.2224), (896, 0.2106), (1152, 0.2038), (3072, 0.1564)]
hier = [(480, 0.2260), (896, 0.1706), (1152, 0.1848), (3072, 0.1695)]
# regression-head reference points (context, dashed/open markers)
reg_pts = [
    ("spat18-reg", 672, 0.2008),
    ("reg-flat", 1152, 0.2015),
    ("reg-long", 1152, 0.1682),
]
flat_names = ["flat-short", "flat-min", "flat-budget", "flat-long"]
hier_names = ["hier-short", "hier-min", "hier-long", "hier-rich"]
PERSIST = 0.2552

fx, fy = np.array([p[0] for p in flat], float), np.array([p[1] for p in flat], float)
hx, hy = np.array([p[0] for p in hier], float), np.array([p[1] for p in hier], float)

# crossings of the two piecewise-linear curves in log-token space -> mid-budget win region
grid = np.linspace(np.log(max(fx[0], hx[0])), np.log(3072), 4000)
f_i = np.interp(grid, np.log(fx), fy)
h_i = np.interp(grid, np.log(hx), hy)
diff = f_i - h_i  # >0 where hier is better (lower error)
sign = np.sign(diff)
idx = np.nonzero(np.diff(sign) != 0)[0]
crossings = []
for i in idx:
    t0, t1, d0, d1 = grid[i], grid[i + 1], diff[i], diff[i + 1]
    crossings.append(np.exp(t0 - d0 * (t1 - t0) / (d1 - d0)))

fig, ax = plt.subplots(figsize=(7.6, 5.0))

# win band: lower edge = geometric span-threshold (48*16 coarse tok + minimal fine ~ 830;
# measured bound: flat wins at 512, hier wins at 896); upper edge = measured curve crossing.
lo_edge = 830.0
hi_edge = crossings[-1] if crossings else 3072.0
ax.axvspan(lo_edge, hi_edge, color="#2a9d8f", alpha=0.10, zorder=0)
ax.axvline(lo_edge, color="#2a9d8f", lw=0.9, ls="--", zorder=1)
ax.text(np.sqrt(lo_edge * hi_edge), 0.2455,
        "hier wins\n(span ≥ loop, window < loop)", ha="center", va="top", fontsize=9,
        color="#1d7268", style="italic")
ax.text(820, 0.2645, "span=loop threshold\n(48×16 tok, geometric)", ha="right", va="top",
        fontsize=7.5, color="#1d7268")

ax.axhline(PERSIST, color="0.45", lw=1.0, ls=":", zorder=1)
ax.text(3072, PERSIST - 0.0015, "persistence 0.2552", ha="right", va="top",
        fontsize=8, color="0.35")

ax.plot(fx, fy, "-o", color="#264653", lw=1.8, ms=6, label="flat (full-res window)", zorder=3)
ax.plot(hx, hy, "-s", color="#e76f51", lw=1.8, ms=6,
        label="hierarchical (coarse + recent fine)", zorder=3)

# regression-head reference points
rx = [p[1] for p in reg_pts]
ry = [p[2] for p in reg_pts]
ax.scatter(rx, ry, marker="D", s=34, facecolors="none", edgecolors="#6a4c93",
           linestyles="dashed", linewidths=1.2, zorder=3,
           label="regression-head points (context)")

# seed-1 replications where measured: hollow markers + spread whiskers on the hier arms
# (hier-long 0.1690 / flat-budget 0.2044; hier-min 0.1744 / flat-min 0.2102 — R-A, 2026-07-08)
ax.plot([1152, 1152], [0.1690, 0.1848], color="#e76f51", lw=0.9, alpha=0.55, zorder=2)
ax.plot([896, 896], [0.1706, 0.1744], color="#e76f51", lw=0.9, alpha=0.55, zorder=2)
ax.scatter([1152, 896], [0.1690, 0.1744], marker="s", s=36, facecolors="none",
           edgecolors="#e76f51", linewidths=1.4, zorder=3, label="seed-1 replications (hollow)")
ax.scatter([1152, 896], [0.2044, 0.2102], marker="o", s=36, facecolors="none",
           edgecolors="#264653", linewidths=1.4, zorder=3)
ax.annotate("hier-long s1\n0.1690", (1152, 0.1690), textcoords="offset points",
            xytext=(70, -26), ha="left", fontsize=7.5, color="#b3492f",
            arrowprops=dict(arrowstyle="-", color="#e76f51", lw=0.7, alpha=0.6))

# point labels
lab_kw = dict(fontsize=8.5, zorder=4)
offs_flat = [(24, -24), (-6, 10), (16, 10), (-8, -24)]
for (x, y), name, (dx, dy) in zip(flat, flat_names, offs_flat):
    ax.annotate(f"{name}\n{y:.4f}", (x, y), textcoords="offset points",
                xytext=(dx, dy), ha="center", color="#264653", **lab_kw)
offs_hier = [(-2, 10), (30, -16), (12, -26), (8, 10)]
for (x, y), name, (dx, dy) in zip(hier, hier_names, offs_hier):
    ax.annotate(f"{name}\n{y:.4f}", (x, y), textcoords="offset points",
                xytext=(dx, dy), ha="center", color="#b3492f", **lab_kw)
reg_offs = [(-8, -14), (36, 6), (38, -3)]
for (name, x, y), (dx, dy) in zip(reg_pts, reg_offs):
    ax.annotate(f"{name} {y:.4f}", (x, y), textcoords="offset points",
                xytext=(dx, dy), ha="center", fontsize=7.5, color="#6a4c93", zorder=4)

# end-inversion callouts
ax.annotate("hier wins +0.0399 / +0.0358 (2 seeds)\n(peak of the band; every cell, both seeds)",
            (896, 0.1706),
            textcoords="offset points", xytext=(44, -52), ha="left", fontsize=8,
            color="#1d7268",
            arrowprops=dict(arrowstyle="-", color="#2a9d8f", lw=0.9))
ax.annotate("flat wins\n−0.0036 [−0.0063,−0.0010]", (500, 0.2242),
            textcoords="offset points", xytext=(16, 42), ha="left", fontsize=8,
            color="0.25",
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8))
ax.annotate("flat wins\n−0.0131 [−0.0201,−0.0069]", (2900, 0.1685),
            textcoords="offset points", xytext=(-118, 36), ha="left", fontsize=8,
            color="0.25",
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8))

ax.set_xscale("log")
ax.set_xticks([512, 672, 896, 1152, 3072])
ax.set_xticklabels(["480–512", "672", "896", "1152", "3072"], fontsize=9)
ax.minorticks_off()
ax.set_xlim(430, 3600)
ax.set_ylim(0.145, 0.265)
ax.set_xlabel("context-token budget (log scale)")
ax.set_ylabel("fg-MSE, ALL (152-frame AR rollout; lower is better)")
ax.set_title("Token-budget frontier: hierarchy wins the band where its span crosses the memory horizon\n"
             "(loopmom; curves = seed 0, hollow = seed-1 replications; lower edge TWO-SEED at the geometric ~896-tok prediction)",
             fontsize=10.5)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax.grid(True, which="major", axis="y", alpha=0.25)

fig.tight_layout()
out = r"C:\Users\leoni\Documents\projects\hierarchical-video-prediction\docs\figures\fig9_frontier.png"
fig.savefig(out, dpi=180)
print("saved", out, "| crossings at tokens:", [round(c) for c in crossings])
