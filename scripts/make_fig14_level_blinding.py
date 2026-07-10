"""fig14_level_blinding.png — level-content attribution by stream blinding ([FIG ch4-6]).

Curves: fg-MSE vs rollout horizon for hier-long unblinded / blind-coarse / blind-fine (+ hier3
blind-top if present), plus flat-budget as the "no coarse memory at all" reference. Inset text:
the dt<=48 HARD retrieval read. Binding OOD caveat (pre-registered) goes in the caption: zeroed
streams are out-of-distribution inputs — collapse magnitude = stream contribution confounded
with OOD brittleness; the probe licenses "stream removed -> capability lost" only.

Graceful: skips series whose npz is absent (run after the Phase-B blinding stage).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = os.path.join(os.path.dirname(__file__), "..", "logs_rhcp")
HBINS = [(1, 8), (9, 16), (17, 24), (25, 36), (37, 48), (49, 60), (61, 80), (81, 104),
         (105, 128), (129, 152)]
SERIES = [  # (npz basename, label, color, style)
    ("rhvd_eval_hier-long_s0.npz", "hier-long (unblinded)", "#e76f51", "-"),
    ("rhvd_eval_hier-long_s0_blind-coarse.npz", "blind-COARSE stream", "#9b2226", "--"),
    ("rhvd_eval_hier-long_s0_blind-fine.npz", "blind-FINE stream", "#8338ec", "--"),
    ("rhvd_eval_hier3_s0_blind-top.npz", "hier3, blind-TOP", "#2a9d8f", ":"),
    ("rhvd_eval_flat-budget_s0.npz", "flat-budget (no coarse at all)", "#264653", "-"),
]


def main():
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = [0.5 * (lo + hi) for lo, hi in HBINS]
    drawn, notes = 0, []
    for base, label, col, sty in SERIES:
        p = os.path.join(LOGS, base)
        if not os.path.exists(p):
            print(f"  (skip {base} — not measured yet)")
            continue
        with np.load(p) as z:
            h, fg = z["h"], z["fg"]
            curve = [float(fg[(h >= lo) & (h <= hi)].mean()) for lo, hi in HBINS]
            dt_ok = "dt" in z.files and "hard" in z.files and "seen" in z.files
            if dt_ok:
                m = (z["dt"] <= 48) & (z["hard"] > 0.1) & (z["seen"] >= 0.5)
                notes.append(f"{label}: dt≤48 HARD {float(fg[m].mean()):.3f}")
        ax.plot(xs, curve, sty, marker="o", ms=3.5, lw=1.7, color=col, label=label)
        drawn += 1
    ax.set_xlabel("rollout horizon h (frames after GT warmup)")
    ax.set_ylabel("fg-MSE (lower better)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.95, loc="lower right")
    ax.set_title("What each stream stores — blinding one context stream at every rollout step\n"
                 "(zeros in standardized latent space; OOD caveat binding: lower bound on "
                 "contribution ⊕ upper bound on brittleness)", fontsize=9.5)
    if notes:
        ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=7.5,
                color="0.35")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "figures",
                       "fig14_level_blinding.png")
    fig.savefig(out, dpi=180)
    print(f"saved {out} ({drawn} series)")


if __name__ == "__main__":
    main()
