"""fig13_family_edges.png — the band-edge law across abstraction families ([FIG ch4-5]).

One panel: paired Δ(flat − hier) at each measured budget, drawn PER FAMILY, with each family's
token-arithmetic edge prediction marked. Message: the lower edge is arithmetic —
edge(family) = (loop / k_t) · tokens-per-frame(k_s) + fine floor — measured at its prediction
for BOTH families; the upper edge is the measured bracket (1792, 2432) for the stride-1 family.

Data: compare logs (cluster CIs). Stride-1 points two-seed except the bracket; stride-2 point
seed 0 (s1 queued per the binding rule — figure footnote carries this).
"""
import os
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if os.environ.get("HVP_ALLOW_HISTORICAL_FIGURES") != "1":
    raise RuntimeError(
        "fig13 is an archived development-era family-edge visualization and must not be used as "
        "terminal evidence. Run scripts/make_rft_result_figures.py for the registered frontier, "
        "or set HVP_ALLOW_HISTORICAL_FIGURES=1 solely for provenance reproduction."
    )

# family -> {budget: [(delta, lo, hi) per seed]}
S1 = {  # stride-1 coarse family (k_t=1, k_s=2): edge prediction 48*16 + 2*64 = 896
    512:  [(-0.0036, -0.0063, -0.0010), (-0.0051, -0.0079, -0.0024)],
    896:  [(+0.0399, +0.0299, +0.0501), (+0.0358, +0.0267, +0.0454)],
    1152: [(+0.0189, +0.0121, +0.0264), (+0.0354, +0.0267, +0.0447)],
    1792: [(+0.0077, +0.0016, +0.0142)],
    2432: [(-0.0116, -0.0162, -0.0073)],
    3072: [(-0.0131, -0.0201, -0.0069), (-0.0107, -0.0163, -0.0058)],
}
S2 = {  # stride-2 coarse family (k_t=2, k_s=2): edge prediction 24*16 + 2*64 = 512
    512:  [(+0.0330, +0.0246, +0.0419)],
}
C1, C2, GRAY = "#e76f51", "#8338ec", "0.55"

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.axhline(0, color="0.3", lw=1.0)
ax.axhspan(0, 0.055, color="#2a9d8f", alpha=0.06)
ax.text(0.985, 0.965, "Δ > 0 : hierarchy better", transform=ax.transAxes, ha="right",
        va="top", color="#1d7268", fontsize=8.5, style="italic")

for fam, col, mk, name in ((S1, C1, "o", "stride-1 coarse (edge pred. 896)"),
                           (S2, C2, "D", "stride-2 coarse (edge pred. 512)")):
    first = True
    for b, seeds in sorted(fam.items()):
        offs = np.linspace(-0.03, 0.03, len(seeds)) if len(seeds) > 1 else [0.0]
        for j, (d, lo, hi) in enumerate(seeds):
            ax.errorbar(b * (1 + offs[j]), d, yerr=[[d - lo], [hi - d]], fmt=mk, ms=6,
                        color=col, capsize=3.5, lw=1.5,
                        label=name if first else None)
            first = False
# family edge predictions (vertical marks) + the measured upper bracket
ax.axvline(896, color=C1, ls="--", lw=1.1, alpha=0.8)
ax.text(896, 0.052, "896 = 48·16+2·64\n(stride-1 prediction)", ha="center", fontsize=7.5,
        color=C1)
ax.axvline(512, color=C2, ls="--", lw=1.1, alpha=0.8)
ax.text(512, 0.052, "512 = 24·16+2·64\n(stride-2 prediction)", ha="center", fontsize=7.5,
        color=C2)
ax.axvspan(1792, 2432, color=GRAY, alpha=0.15)
ax.text(2085, -0.0245, "upper edge\nbracket (R-F/R-G)", ha="center", fontsize=7.5, color=GRAY)

ax.set_xscale("log")
ax.set_xticks([512, 896, 1152, 1792, 2432, 3072])
ax.set_xticklabels(["512", "896", "1152", "1792", "2432", "3072"])
ax.minorticks_off()
ax.set_xlabel("matched context-token budget (log scale)")
ax.set_ylabel("paired Δ(flat − hier), ALL  [95% cluster CI]")
ax.set_ylim(-0.028, 0.062)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
ax.set_title("The band-edge law across abstraction families — edges are arithmetic, not "
             "accidents\nhier wins from each family's predicted edge; at 512 tok the stride-1 "
             "family loses (two-seed) while stride-2 wins", fontsize=9.5)
fig.text(0.01, 0.01, "stride-2 point: seed 0, every-cell win; seed-1 queued per the binding "
         "rule before headline use. Same budget, different families — not a contradiction.",
         fontsize=7, color="0.4")
fig.tight_layout(rect=[0, 0.035, 1, 1])
out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig13_family_edges.png"
fig.savefig(out, dpi=180)
print("saved", out)
