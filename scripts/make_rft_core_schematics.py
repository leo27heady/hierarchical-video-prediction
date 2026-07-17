"""Generate the two non-result schematics used by the RFT article and dissertation.

The drawings are deterministic and derive their numerical labels from the frozen E2
configuration.  They deliberately distinguish the manipulated context package from the
reach/fidelity interpretation and label the pose-indexed E8a system as a privileged reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
E2_CONFIG = ROOT / "experiments" / "configs" / "e2_untouched_confirmation.json"


COLORS = {
    "ink": "#20252b",
    "muted": "#65717d",
    "grid": "#d8dde3",
    "flat": "#d97706",
    "coarse": "#2563a6",
    "fine": "#3a9d78",
    "target": "#8a5aa5",
    "raw": "#b6463d",
    "soft_blue": "#dceaf7",
    "soft_green": "#dff1e9",
    "soft_orange": "#fae8cd",
    "soft_purple": "#ece2f3",
    "soft_red": "#f6dfdc",
    "paper": "#ffffff",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COLORS["ink"],
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}")
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str | None = None,
    fontsize: float = 9,
    linestyle: str = "-",
) -> FancyBboxPatch:
    edge = edge or COLORS["ink"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    style: str = "-|>",
    linestyle: str = "-",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.2,
            color=color or COLORS["ink"],
            linestyle=linestyle,
            connectionstyle=connectionstyle,
        )
    )


def make_testbed(config: dict) -> None:
    world = int(config["dataset"]["world"])
    viewport = int(config["dataset"]["viewport"])
    warm = int(config["evaluation"]["warmup_frames"])
    rollout = int(config["evaluation"]["rollout_frames"])

    fig = plt.figure(figsize=(11.6, 5.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(-2, world + 2)
    ax.set_ylim(-0.22, 1.24)
    ax.axis("off")
    ax.set_title(
        "A  Procedural re-entry instrument: geometry is observed exactly",
        loc="left",
        fontweight="bold",
    )

    ax.add_patch(Rectangle((0, 0.34), world, 0.47, fc="#f2f4f6", ec=COLORS["ink"], lw=1.2))
    for x in range(0, world + 1, 32):
        ax.plot([x, x], [0.34, 0.81], color=COLORS["grid"], lw=0.8)
        if x < world:
            ax.text(x + 2, 0.29, str(x), color=COLORS["muted"], fontsize=7)

    glyphs = [(24, "7"), (91, "3"), (157, "8")]
    for x, digit in glyphs:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.43),
                18,
                0.29,
                boxstyle="round,pad=0.01",
                fc=COLORS["soft_purple"],
                ec=COLORS["target"],
                lw=1.0,
            )
        )
        ax.text(x + 9, 0.575, digit, ha="center", va="center", fontsize=17, color=COLORS["target"])

    view_positions = [3, 67, 131]
    view_labels = [r"$t-48$", r"$t-24$", r"$t$ (re-entry)"]
    view_colors = [COLORS["flat"], COLORS["coarse"], COLORS["raw"]]
    for position, label, color in zip(view_positions, view_labels, view_colors):
        ax.add_patch(
            Rectangle(
                (position, 0.37),
                viewport,
                0.41,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
            )
        )
        ax.text(
            position + viewport / 2,
            0.88,
            f"{label}\n64-px viewport",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

    _arrow(
        ax,
        (view_positions[0] + viewport / 2, 0.99),
        (view_positions[1] + viewport / 2, 0.99),
        color=COLORS["muted"],
        connectionstyle="arc3,rad=-0.18",
    )
    _arrow(
        ax,
        (view_positions[1] + viewport / 2, 0.99),
        (view_positions[2] + viewport / 2, 0.99),
        color=COLORS["muted"],
        connectionstyle="arc3,rad=-0.18",
    )
    ax.text(world - 1, 0.03, "toroidal world: right edge wraps to left", ha="right", color=COLORS["muted"])
    ax.text(
        0,
        1.15,
        f"{world}-px static world  •  {config['dataset']['digits']} persistent digits  •  "
        f"saved actions/poses  •  {warm} GT warm-up + {rollout} predicted frames",
        ha="left",
        fontsize=9,
    )

    ax2 = fig.add_subplot(grid[1])
    ax2.set_title(
        "B  Pair-derived reach analysis: test the records visible only to the longer package",
        loc="left",
        fontweight="bold",
    )
    ax2.set_xlim(-52, 4)
    ax2.set_ylim(-0.7, 3.6)
    ax2.set_xlabel("Frames before the current target (0 = prediction time)")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(
        [
            "flat-min\n14-frame reach",
            "hier-min\n48-frame reach",
            "pose mosaic K=48\n(privileged reference)",
        ]
    )
    ax2.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    for y, start, color in [
        (0, -14, COLORS["flat"]),
        (1, -48, COLORS["coarse"]),
        (2, -48, COLORS["raw"]),
    ]:
        ax2.plot([start, 0], [y, y], color=color, lw=8, solid_capstyle="butt")
        ax2.plot([0], [y], marker=">", color=color, ms=8)
    ax2.axvspan(-48, -14, color=COLORS["soft_blue"], alpha=0.9, zorder=-2)
    ax2.text(
        -31,
        3.05,
        r"registered mechanistic band: $14 < \Delta t \leq 48$",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["coarse"],
    )
    ax2.text(
        3.5,
        2,
        "exact future pose\nraw observed pixels",
        ha="right",
        va="center",
        color=COLORS["raw"],
        fontsize=8,
    )
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    _save(fig, "fig15_loopmom_observability_schematic")


def make_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Context packaging is the intervention; reach and fidelity are explanatory properties",
        loc="left",
        fontweight="bold",
    )

    _box(ax, (0.02, 0.60), 0.12, 0.17, "Input frames\n64 × 64", face="#f2f4f6")
    _box(
        ax,
        (0.18, 0.53),
        0.16,
        0.31,
        "Frozen two-level\nconv autoencoder\n\nshared encoder",
        face=COLORS["soft_blue"],
        edge=COLORS["coarse"],
    )
    ax.text(0.26, 0.49, "no predictor gradient", ha="center", color=COLORS["muted"], fontsize=8)
    _arrow(ax, (0.14, 0.685), (0.18, 0.685))

    _box(
        ax,
        (0.39, 0.73),
        0.14,
        0.12,
        "coarse latent\n16 × 4 × 4",
        face=COLORS["soft_blue"],
        edge=COLORS["coarse"],
    )
    _box(
        ax,
        (0.39, 0.53),
        0.14,
        0.12,
        "fine latent\n16 × 8 × 8",
        face=COLORS["soft_green"],
        edge=COLORS["fine"],
    )
    _arrow(ax, (0.34, 0.72), (0.39, 0.79), color=COLORS["coarse"])
    _arrow(ax, (0.34, 0.64), (0.39, 0.59), color=COLORS["fine"])

    _box(
        ax,
        (0.58, 0.55),
        0.18,
        0.28,
        "Standardize with\ntraining-only statistics\n\npackage context tokens\n+ actions + role/level/time",
        face="#f4f5f7",
    )
    _arrow(ax, (0.53, 0.79), (0.58, 0.75), color=COLORS["coarse"])
    _arrow(ax, (0.53, 0.59), (0.58, 0.63), color=COLORS["fine"])

    _box(
        ax,
        (0.80, 0.55),
        0.17,
        0.28,
        "Shared WindowDiT\njoint self-attention\n\n4-frame latent target\n16-step DDIM",
        face=COLORS["soft_purple"],
        edge=COLORS["target"],
    )
    _arrow(ax, (0.76, 0.69), (0.80, 0.69), color=COLORS["target"])
    _box(
        ax,
        (0.80, 0.32),
        0.17,
        0.11,
        "Frozen fine decoder\n→ predicted frames",
        face=COLORS["soft_green"],
        edge=COLORS["fine"],
    )
    _arrow(ax, (0.885, 0.55), (0.885, 0.43), color=COLORS["fine"])

    ax.plot([0.015, 0.985], [0.25, 0.25], color=COLORS["grid"], lw=1.0)
    ax.text(0.02, 0.205, "Three tested ways to spend context tokens", fontweight="bold")

    package_specs = [
        (
            0.02,
            0.145,
            "Flat full resolution",
            "14 × 64 = 896 tokens\nreach 14 • high fidelity",
            COLORS["soft_orange"],
            COLORS["flat"],
        ),
        (
            0.34,
            0.14,
            "Spatial package",
            "48 × 16 + 2 × 64 = 896\nreach 48 • coarse history",
            COLORS["soft_blue"],
            COLORS["coarse"],
        ),
        (
            0.66,
            0.14,
            "Temporal package",
            "11 × 64 + 6 × 64 = 1,088\nreach 48 • sparse anchors",
            COLORS["soft_green"],
            COLORS["fine"],
        ),
    ]
    for x, y, title, subtitle, face, edge in package_specs:
        _box(ax, (x, y - 0.085), 0.29, 0.11, f"{title}\n{subtitle}", face=face, edge=edge, fontsize=8.5)

    ax.text(
        0.5,
        0.002,
        "Target tokens, total sequence length, training regime, sampler work and codec seed are "
        "reported separately; equal context tokens do not imply equal total compute.",
        ha="center",
        color=COLORS["muted"],
        fontsize=8,
    )
    _save(fig, "fig16_context_packaging_architecture")


def main() -> None:
    _style()
    config = json.loads(E2_CONFIG.read_text(encoding="utf-8"))
    make_testbed(config)
    make_architecture()
    print("RFT_CORE_SCHEMATICS_DONE")


if __name__ == "__main__":
    main()
