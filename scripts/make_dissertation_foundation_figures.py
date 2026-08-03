"""Generate deterministic foundation figures for dissertation Chapters 1 and 2.

The drawings are explanatory schematics, not empirical result figures. Measured memory panels
use only the values already reported in the dissertation and their cited repository artifacts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#d1d5db"
BLUE = "#0072b2"
ORANGE = "#e69f00"
GREEN = "#009e73"
PURPLE = "#cc79a7"
RED = "#d55e00"
PALE = "#f3f4f6"
WHITE = "#ffffff"
PDF_TIMESTAMP = datetime(2026, 8, 3, tzinfo=timezone.utc)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", facecolor=WHITE, transparent=False)
    fig.savefig(
        OUT / f"{stem}.pdf",
        facecolor=WHITE,
        transparent=False,
        metadata={
            "Creator": "hierarchical-video-prediction",
            "CreationDate": PDF_TIMESTAMP,
            "ModDate": PDF_TIMESTAMP,
        },
    )
    plt.close(fig)


def clean(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    color=PALE,
    *,
    fontsize=8.5,
    edge=INK,
    text_color=INK,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
    )


def arrow(ax, a, b, *, color=INK, width=1.2, style="-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=width,
            color=color,
        )
    )


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold")


def make_control_system() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.1), constrained_layout=True)
    clean(ax)
    title(ax, "Controlled autoregressive video generation")

    box(ax, 0.04, 0.68, 0.17, 0.13, "control $u_t$\nactions / conditions", "#e8f1f8")
    box(ax, 0.04, 0.45, 0.17, 0.13, "innovation $\\xi_t$\nsampled noise", "#f7eaf3")
    box(ax, 0.30, 0.49, 0.30, 0.24,
        "transition / refinement core\n$z_t=f_\\theta(m_t,u_t,\\xi_t)$", "#e8f1f8")
    box(ax, 0.71, 0.54, 0.21, 0.14, "decoded output\n$\\hat{x}_t$", "#e7f5f0")
    box(ax, 0.34, 0.18, 0.22, 0.13, "retained context state\n$m_{t+1}=U(m_t,z_t)$", "#fff3d6")

    arrow(ax, (0.21, 0.745), (0.30, 0.65), color=BLUE)
    arrow(ax, (0.21, 0.515), (0.30, 0.57), color=PURPLE)
    arrow(ax, (0.60, 0.61), (0.71, 0.61), color=GREEN)
    arrow(ax, (0.49, 0.49), (0.46, 0.31), color=ORANGE)
    arrow(ax, (0.34, 0.245), (0.26, 0.245), color=ORANGE)
    arrow(ax, (0.26, 0.245), (0.26, 0.54), color=ORANGE)
    arrow(ax, (0.26, 0.54), (0.30, 0.54), color=ORANGE)
    ax.text(0.23, 0.34, "self-fed\nfeedback", color=ORANGE, ha="center", va="center")
    ax.text(0.65, 0.75, "one generated step", color=MUTED, ha="center")
    ax.text(
        0.5,
        0.06,
        "Conceptual system view: trajectory quality depends on control, stochastic refinement, and the retained self-fed state.",
        ha="center",
        color=MUTED,
    )
    save(fig, "fig30_control_system")


def make_consistency_landscape() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    clean(ax)
    title(ax, "Conceptual refinement toward a coherent trajectory region")

    center = np.asarray([0.73, 0.48])
    for width, height, alpha in [(0.48, 0.58, 0.10), (0.34, 0.40, 0.16), (0.20, 0.23, 0.25)]:
        ax.add_patch(
            mpl.patches.Ellipse(
                center,
                width,
                height,
                facecolor=BLUE,
                edgecolor=BLUE,
                linewidth=1.2,
                alpha=alpha,
            )
        )
    path = [(0.13, 0.78), (0.25, 0.66), (0.38, 0.64), (0.49, 0.53), (0.61, 0.50), (0.70, 0.48)]
    for a, b in zip(path[:-1], path[1:]):
        arrow(ax, a, b, color=RED, width=1.8)
    ax.scatter(*zip(*path), s=28, color=RED, edgecolor=WHITE, linewidth=0.6, zorder=4)
    ax.text(0.10, 0.84, "inconsistent candidate", color=RED, ha="left", fontweight="bold")
    ax.text(0.73, 0.48, "coherent\nregion", color=INK, ha="center", va="center", fontweight="bold")
    ax.text(0.47, 0.72, "iterative update", color=RED, ha="center")
    ax.text(0.50, 0.13, "$J$ denotes a family of consistency criteria, not one measured scalar surface.", ha="center")
    ax.text(0.50, 0.06, "Illustrative geometry only; axes and path are not empirical measurements.", ha="center", color=MUTED)
    save(fig, "fig31_consistency_landscape")


def make_system_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.9), constrained_layout=True)
    clean(ax)
    title(ax, "Representation, dynamics, and rendering are separate system interfaces")
    box(ax, 0.04, 0.43, 0.18, 0.19, "representation\nencoder $E$\n$x_t \\mapsto z_t$", "#e8f1f8")
    box(ax, 0.33, 0.35, 0.34, 0.35, "dynamics / generation core\n\ncontext-state update\nconditional refinement\ntarget-window prediction", "#f7eaf3")
    box(ax, 0.78, 0.43, 0.18, 0.19, "rendering\ndecoder $D$\n$\\hat z_t \\mapsto \\hat x_t$", "#e7f5f0")
    arrow(ax, (0.22, 0.525), (0.33, 0.525), color=BLUE)
    arrow(ax, (0.67, 0.525), (0.78, 0.525), color=GREEN)

    box(ax, 0.08, 0.77, 0.14, 0.10, "controls $u_t$", "#fff3d6")
    box(ax, 0.43, 0.79, 0.14, 0.10, "noise $\\xi_t$", "#fbe9e7")
    arrow(ax, (0.22, 0.82), (0.38, 0.69), color=ORANGE)
    arrow(ax, (0.50, 0.79), (0.50, 0.70), color=RED)

    # Two complementary arcs form one visibly closed cycle with a single direction marker.
    ax.add_patch(FancyArrowPatch(
        (0.455, 0.395), (0.545, 0.395),
        connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=11,
        linewidth=1.8, color=PURPLE,
    ))
    ax.add_patch(FancyArrowPatch(
        (0.545, 0.395), (0.455, 0.395),
        connectionstyle="arc3,rad=-0.35", arrowstyle="-",
        linewidth=1.8, color=PURPLE,
    ))
    ax.text(0.50, 0.27, "internal refinement loop", color=PURPLE, ha="center")
    ax.text(0.50, 0.10, "Conceptual interfaces; later chapters test different resource costs at different interfaces.", ha="center", color=MUTED)
    save(fig, "fig32_system_decomposition")


def make_visual_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.5), constrained_layout=True)
    clean(ax)
    title(ax, "Conceptual levels of visual state")
    layers = [
        ((0.23, 0.70), 0.54, 0.13, "MACRO  — scene layout / global composition", BLUE),
        ((0.16, 0.43), 0.68, 0.14, "MESO  — objects / contours / geometry", GREEN),
        ((0.09, 0.15), 0.82, 0.15, "MICRO  — texture / lighting / pixel detail", ORANGE),
    ]
    for (x, y), w, h, label, color in layers:
        box(ax, x, y, w, h, label, color, text_color=WHITE)
    arrow(ax, (0.27, 0.43), (0.34, 0.30), color=ORANGE)
    arrow(ax, (0.34, 0.70), (0.27, 0.57), color=ORANGE)
    ax.text(0.12, 0.50, "detail\nconditioning", color=ORANGE, ha="center", va="center")
    arrow(ax, (0.73, 0.30), (0.66, 0.43), color=BLUE)
    arrow(ax, (0.66, 0.57), (0.73, 0.70), color=BLUE)
    ax.text(0.89, 0.50, "coarser\nsummary", color=BLUE, ha="center", va="center")
    ax.text(0.50, 0.06, "Conceptual taxonomy: the validated predictor uses coarse and fine levels, not an explicit semantic macro label.", ha="center", color=MUTED)
    save(fig, "fig33_visual_hierarchy")


def make_temporal_grid() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.2), constrained_layout=True)
    ax.set_xlim(-0.4, 8.5)
    ax.set_ylim(-0.4, 5.1)
    title(ax, "Physical time and refinement pseudo-time form a two-dimensional process")
    for t in range(8):
        for k in range(4):
            ax.scatter(t, k, s=52, color=BLUE if k < 3 else GREEN, zorder=3)
            if k < 3:
                arrow(ax, (t, k + 0.08), (t, k + 0.82), color=BLUE, width=0.8)
        if t < 7:
            arrow(ax, (t + 0.08, 3), (t + 0.82, 3), color=ORANGE, width=1.0)
    ax.add_patch(Rectangle((3.55, -0.22), 3.9, 3.6, fill=False, edgecolor=RED, linewidth=1.4, linestyle="--"))
    ax.text(5.5, -0.05, "sliding context window $w$", ha="center", color=RED)
    ax.set_xlabel("physical frame time $t$")
    ax.set_ylabel("refinement step $k$")
    ax.set_xticks(range(8))
    ax.set_yticks(range(4))
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "fig34_temporal_grid")


def make_fixed_tokenization() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax in axes:
        clean(ax)
    axes[0].set_title("A  Fixed grid", loc="left", fontweight="bold")
    axes[1].set_title("B  Information density", loc="left", fontweight="bold")
    for ax in axes:
        ax.add_patch(Rectangle((0.10, 0.12), 0.78, 0.70, facecolor="#eaf0f4", edgecolor=INK))
        ax.add_patch(Circle((0.62, 0.46), 0.13, facecolor=ORANGE, edgecolor=INK))
    for i in range(1, 6):
        axes[0].plot([0.10 + i * 0.78 / 6] * 2, [0.12, 0.82], color=INK, linewidth=0.6)
        axes[0].plot([0.10, 0.88], [0.12 + i * 0.70 / 6] * 2, color=INK, linewidth=0.6)
    axes[0].text(0.49, 0.05, "36 equal-cost tokens", ha="center", color=MUTED)
    for r, alpha in zip((0.21, 0.17, 0.13), (0.20, 0.35, 0.60)):
        axes[1].add_patch(Circle((0.62, 0.46), r, fill=False, edgecolor=RED, linewidth=2, alpha=alpha))
    axes[1].text(0.49, 0.05, "complexity concentrated at the moving contour", ha="center", color=MUTED)
    fig.suptitle("Uniform patching spends the same capacity on background and structure", fontweight="bold")
    save(fig, "fig35_fixed_grid_tokenization")


def make_diffusion_pair() -> None:
    rng = np.random.default_rng(7)
    base = np.zeros((48, 48))
    yy, xx = np.mgrid[:48, :48]
    base[((xx - 25) ** 2 / 115 + (yy - 24) ** 2 / 210) < 1] = 1
    frames = []
    for amount in np.linspace(0, 1, 6):
        noise = rng.normal(0, 1, base.shape)
        frames.append(np.clip((1 - amount) * base + amount * (noise * 0.25 + 0.5), 0, 1))
    for reverse, stem, heading in [
        (False, "fig36_forward_diffusion", "Forward process $q$: data to noise"),
        (True, "fig37_reverse_diffusion", "Learned reverse process $p_\\theta$: noise to data"),
    ]:
        seq = list(reversed(frames)) if reverse else frames
        fig, axes = plt.subplots(1, 6, figsize=(11.5, 2.8), constrained_layout=True)
        for i, (ax, image) in enumerate(zip(axes, seq)):
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            if reverse:
                labels = ["$x_T$", "$x_{t_4}$", "$x_{t_3}$", "$x_{t_2}$", "$x_{t_1}$", "$\\hat{x}_0$"]
            else:
                labels = ["$x_0$", "$x_{t_1}$", "$x_{t_2}$", "$x_{t_3}$", "$x_{t_4}$", "$x_T$"]
            ax.set_title(labels[i])
            ax.axis("off")
            if i < len(seq) - 1:
                ax.text(
                    1.10,
                    0.50,
                    "$\\rightarrow$",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=BLUE if reverse else RED,
                    clip_on=False,
                )
        fig.suptitle(heading, fontsize=12, fontweight="bold")
        fig.text(
            0.5,
            0.02,
            "Illustrative states and schedule positions; the strip is explanatory, not a model sample or measured trajectory.",
            ha="center",
            color=MUTED,
            fontsize=8.5,
        )
        save(fig, stem)


def make_ldm() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.0), constrained_layout=True)
    clean(ax)
    title(ax, "Latent diffusion separates spatial compression from iterative denoising")
    box(ax, 0.03, 0.50, 0.12, 0.14, "image\n$x$", "#e8f1f8")
    box(ax, 0.20, 0.50, 0.14, 0.14, "encoder\n$E$", BLUE)
    box(ax, 0.40, 0.50, 0.14, 0.14, "clean latent\n$z_0$", PALE)
    box(ax, 0.64, 0.50, 0.14, 0.14, "noisy latent\n$z_t$", "#fbe9e7")
    box(ax, 0.84, 0.50, 0.12, 0.14, "noise\n$z_T$", "#f7eaf3")
    arrow(ax, (0.15, 0.57), (0.20, 0.57), color=BLUE)
    arrow(ax, (0.34, 0.57), (0.40, 0.57), color=BLUE)
    arrow(ax, (0.54, 0.60), (0.64, 0.60), color=RED)
    arrow(ax, (0.78, 0.60), (0.84, 0.60), color=RED)
    ax.text(0.69, 0.70, "forward corruption $q$", color=RED, ha="center")

    arrow(ax, (0.84, 0.49), (0.78, 0.49), color=PURPLE)
    arrow(ax, (0.64, 0.49), (0.54, 0.49), color=PURPLE)
    ax.text(0.69, 0.40, "learned denoising $p_\\theta$", color=PURPLE, ha="center")
    box(ax, 0.40, 0.22, 0.14, 0.12, "decoder\n$D$", GREEN)
    box(ax, 0.20, 0.22, 0.14, 0.12, "reconstruction\n$\\hat{x}$", "#e7f5f0")
    arrow(ax, (0.47, 0.50), (0.47, 0.34), color=GREEN)
    arrow(ax, (0.40, 0.28), (0.34, 0.28), color=GREEN)
    box(ax, 0.67, 0.20, 0.27, 0.10, "conditioning $c$: context / controls / time", WHITE, fontsize=8.0, edge=PURPLE)
    ax.add_patch(
        FancyArrowPatch(
            (0.80, 0.30),
            (0.72, 0.49),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            linestyle="--",
            color=PURPLE,
        )
    )
    ax.text(0.50, 0.08, "Spatial compression lowers denoising cost; retained temporal context remains a separate resource axis.", ha="center", color=MUTED)
    save(fig, "fig38_latent_diffusion")


def make_long_context_barrier() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), constrained_layout=True)
    clean(axes[0])
    axes[0].set_title("A  Flat-context scaling (conceptual)", loc="left", fontweight="bold")
    box(axes[0], 0.08, 0.64, 0.34, 0.16, "dense fine history\n$B_{ctx}=64H$", "#fff3d6")
    box(axes[0], 0.58, 0.64, 0.34, 0.16, "attention work\n$O(L^2)$", "#fbe9e7")
    arrow(axes[0], (0.42, 0.72), (0.58, 0.72), color=RED)
    box(axes[0], 0.08, 0.31, 0.34, 0.18, "coarse + recent fine\n$B_{ctx}=16H_c+64w_f$", "#e8f1f8")
    box(axes[0], 0.58, 0.31, 0.34, 0.18, "same nominal reach\nwith fewer context tokens", "#e7f5f0")
    arrow(axes[0], (0.42, 0.40), (0.58, 0.40), color=BLUE)
    axes[0].text(0.50, 0.13, "No hardware ceiling or crossover is fitted in this schematic.", ha="center", color=MUTED)

    clean(axes[1])
    axes[1].set_title("B  Re-entry geometry (conceptual)", loc="left", fontweight="bold")
    axes[1].plot([0.08, 0.92], [0.64, 0.64], color=INK, linewidth=1.1)
    events = [(0.13, "observed", GREEN), (0.43, "leaves view", ORANGE), (0.86, "re-enters", RED)]
    for x, label, color in events:
        axes[1].scatter(x, 0.64, s=80, color=color, edgecolor=INK, linewidth=0.5, zorder=3)
        axes[1].text(x, 0.73, label, ha="center")
    axes[1].add_patch(Rectangle((0.57, 0.52), 0.31, 0.08, facecolor=ORANGE, alpha=0.75, edgecolor=INK))
    axes[1].text(0.725, 0.46, "short dense window: earlier view absent", ha="center", color=ORANGE)
    axes[1].add_patch(Rectangle((0.10, 0.28), 0.78, 0.08, facecolor=BLUE, alpha=0.75, edgecolor=INK))
    axes[1].text(0.49, 0.20, "long coarse stream: earlier view addressable", ha="center", color=BLUE)
    axes[1].text(0.50, 0.08, "Addressable history is not a guarantee of successful retrieval.", ha="center", color=MUTED)
    save(fig, "fig39_flat_long_context_barrier")


def make_representation_choice() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.3), constrained_layout=True)
    clean(ax)
    title(ax, "Representation interfaces reviewed; only the continuous hierarchy is evaluated here")
    box(ax, 0.04, 0.57, 0.19, 0.17, "discrete VQ tokens\nindex assignment + codebook", "#efe7dc")
    box(ax, 0.04, 0.25, 0.19, 0.17, "continuous AE latents\nend-to-end differentiable", "#e5edf6")
    box(ax, 0.34, 0.57, 0.25, 0.17, "clean categorical likelihood\nlanguage-model interface", "#f3efe8")
    box(ax, 0.34, 0.25, 0.25, 0.17, "two independently decoded levels\nfrozen codec for package tests", "#e8eef4")
    box(ax, 0.71, 0.40, 0.24, 0.20, "Chapter 3–4 scope\ncontinuous two-level codec\n+ controlled context packaging", GREEN)
    arrow(ax, (0.23, 0.655), (0.34, 0.655), color=ORANGE)
    arrow(ax, (0.23, 0.335), (0.34, 0.335), color=BLUE)
    arrow(ax, (0.59, 0.335), (0.71, 0.46), color=BLUE)
    ax.plot([0.59, 0.71], [0.655, 0.54], color=MUTED, linestyle="--")
    ax.text(0.65, 0.68, "reviewed,\nnot compared", ha="center", color=MUTED)
    ax.text(
        0.50,
        0.08,
        "No controlled VQ-versus-continuous experiment is claimed; representation generality is tested only by the alternate-codec study.",
        ha="center",
        color=MUTED,
    )
    save(fig, "fig56_representation_scope")


def make_package_compare() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    clean(ax)
    title(ax, "Executed context packages: exact token cost and nominal reach")
    rows = [
        ("dense flat", 3072, 48, "full detail"),
        ("budget-matched flat", 1152, 18, "local fidelity"),
        ("spatial hierarchy", 1152, 48, "coarse reach + recent detail"),
        ("temporal sparse", 1088, 48, "sparse full-resolution anchors"),
    ]
    y_positions = [0.76, 0.59, 0.42, 0.25]
    x0, timeline_w, bar_h = 0.18, 0.45, 0.09
    for (name, tokens, reach, note), y in zip(rows, y_positions):
        ax.text(0.02, y + 0.045, name, va="center", fontweight="bold", fontsize=8.5)
        ax.text(0.66, y + 0.045, f"{tokens:,} tokens", va="center", fontweight="bold")
        ax.text(0.77, y + 0.045, f"reach {reach}", va="center")
        ax.text(0.88, y + 0.045, note, va="center", color=MUTED, fontsize=8)

    # The horizontal geometry represents the 48-frame addressable span.
    ax.add_patch(Rectangle((x0, y_positions[0]), timeline_w, bar_h, facecolor=ORANGE, edgecolor=INK, linewidth=0.8, alpha=0.82))
    ax.text(x0 + timeline_w / 2, y_positions[0] + bar_h / 2, "48 fine", ha="center", va="center", color=WHITE, fontweight="bold", fontsize=8)

    flat18_x = x0 + timeline_w * (30 / 48)
    flat18_w = timeline_w * (18 / 48)
    ax.add_patch(Rectangle((flat18_x, y_positions[1]), flat18_w, bar_h, facecolor=ORANGE, edgecolor=INK, linewidth=0.8, alpha=0.82))
    ax.text(flat18_x + flat18_w / 2, y_positions[1] + bar_h / 2, "18 fine", ha="center", va="center", color=WHITE, fontweight="bold", fontsize=8)

    ax.add_patch(Rectangle((x0, y_positions[2]), timeline_w, bar_h, facecolor=BLUE, edgecolor=INK, linewidth=0.8, alpha=0.82))
    recent_w = timeline_w * (6 / 48)
    ax.add_patch(Rectangle((x0 + timeline_w - recent_w, y_positions[2]), recent_w, bar_h, facecolor=ORANGE, edgecolor=INK, linewidth=0.6, alpha=0.95))
    ax.text(x0 + timeline_w * 0.43, y_positions[2] + bar_h / 2, "48 coarse", ha="center", va="center", color=WHITE, fontweight="bold", fontsize=8)
    ax.text(x0 + timeline_w - recent_w / 2, y_positions[2] + bar_h / 2, "+6f", ha="center", va="center", color=WHITE, fontweight="bold", fontsize=7.5)

    anchor_x = np.linspace(x0 + 0.006, x0 + timeline_w - recent_w - 0.010, 11)
    for anchor in anchor_x:
        ax.add_patch(Rectangle((anchor - 0.005, y_positions[3]), 0.010, bar_h, facecolor=GREEN, edgecolor=INK, linewidth=0.4, alpha=0.88))
    ax.add_patch(Rectangle((x0 + timeline_w - recent_w, y_positions[3]), recent_w, bar_h, facecolor=GREEN, edgecolor=INK, linewidth=0.8, alpha=0.95))
    ax.text(x0 + timeline_w * 0.34, y_positions[3] + bar_h / 2, "11 anchors @ stride 4", ha="center", va="center", color=INK, fontweight="bold", fontsize=7.4,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.82, "pad": 1.2})
    ax.text(x0 + timeline_w - recent_w / 2, y_positions[3] + bar_h / 2, "+6f", ha="center", va="center", color=WHITE, fontweight="bold", fontsize=7.5)
    ax.text(0.18, 0.90, "older history", color=MUTED)
    ax.text(0.63, 0.90, "recent history", color=MUTED, ha="right")
    ax.text(0.50, 0.09, "The temporal package is near-matched (1,088 vs 1,152), not token-identical.", ha="center", color=MUTED)
    ax.text(0.50, 0.04, "All rows show context tokens only; the target interface is accounted separately.", ha="center", color=MUTED)
    save(fig, "fig40_context_package_comparison")


def make_conditioning_schematics(*, include_image_to_image: bool = True) -> None:
    # Image-to-image: a simple sequential interface is appropriate.
    if include_image_to_image:
        fig, ax = plt.subplots(figsize=(11.4, 3.8), constrained_layout=True)
        clean(ax)
        title(ax, "Image-to-image conditioning")
        stages = ["reference image", "partial noising\nstrength $s$", "conditional\ndenoising", "output"]
        colors = (BLUE, PURPLE, ORANGE, GREEN)
        xs = (0.03, 0.28, 0.53, 0.78)
        for x, stage, color in zip(xs, stages, colors):
            box(ax, x, 0.43, 0.17, 0.19, stage, color)
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(ax, (a + 0.17, 0.525), (b, 0.525))
        ax.text(0.50, 0.16, "Higher $s$ permits more variation and generally weakens identity retention.", ha="center", color=MUTED)
        save(fig, "fig41_image_to_image")

    # ControlNet: the structural branch injects residuals into a frozen
    # diffusion backbone; it is not a serial preprocessor for that backbone.
    fig, ax = plt.subplots(figsize=(11.4, 4.5), constrained_layout=True)
    clean(ax)
    title(ax, "ControlNet-class structural conditioning")
    box(ax, 0.04, 0.62, 0.18, 0.15, "noisy latent\n+ text condition", "#f7eaf3")
    box(ax, 0.35, 0.56, 0.34, 0.24, "frozen diffusion backbone\nblock 1  ···  block $L$", PALE)
    box(ax, 0.80, 0.62, 0.15, 0.15, "denoised\noutput", "#e7f5f0")
    arrow(ax, (0.22, 0.695), (0.35, 0.695), color=PURPLE)
    arrow(ax, (0.69, 0.695), (0.80, 0.695), color=GREEN)
    box(ax, 0.04, 0.25, 0.18, 0.15, "edge / depth / pose\nstructural signal", "#e8f1f8")
    box(ax, 0.35, 0.21, 0.34, 0.22, "trainable control branch\nzero-initialized adapters\nresidual injections ↑", BLUE, text_color=WHITE)
    arrow(ax, (0.22, 0.325), (0.35, 0.325), color=BLUE)
    for x in (0.43, 0.52, 0.61):
        arrow(ax, (x, 0.43), (x, 0.56), color=BLUE)
    ax.text(0.50, 0.08, "Conceptual routing; exact injection locations depend on the implementation.", ha="center", color=MUTED)
    save(fig, "fig42_controlnet")

    # Shared-attention style conditioning: reference and target branches enter
    # attention in parallel, while target content follows its own denoising path.
    fig, ax = plt.subplots(figsize=(11.4, 4.5), constrained_layout=True)
    clean(ax)
    title(ax, "Reference-assisted shared attention")
    box(ax, 0.04, 0.64, 0.19, 0.15, "reference image\nstyle features", "#e8f1f8")
    box(ax, 0.04, 0.30, 0.19, 0.15, "target latent\n+ target prompt", "#fff3d6")
    box(ax, 0.37, 0.47, 0.25, 0.20, "shared-attention block\nreference K,V + target Q", PURPLE, text_color=WHITE)
    box(ax, 0.75, 0.47, 0.20, 0.20, "target denoising\ntrajectory / output", "#e7f5f0")
    arrow(ax, (0.23, 0.715), (0.37, 0.61), color=BLUE)
    arrow(ax, (0.23, 0.375), (0.37, 0.53), color=ORANGE)
    arrow(ax, (0.62, 0.57), (0.75, 0.57), color=GREEN)
    ax.text(0.58, 0.25, "reference and target enter attention in parallel; the output remains target-specific", color=ORANGE, ha="center")
    ax.text(0.50, 0.08, "Conceptual routing: shared attention transfers reference features without replacing target content inputs.", ha="center", color=MUTED)
    save(fig, "fig43_shared_attention")


def make_trajectory_consistency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax in axes:
        clean(ax)
    axes[0].set_title("A  Which past can each package address?", loc="left", fontweight="bold")
    stages = [(0.08, "visible", GREEN), (0.40, "off-screen", MUTED), (0.75, "re-entry", RED)]
    for i, (x, label, color) in enumerate(stages):
        box(axes[0], x, 0.63, 0.17, 0.13, label, color, text_color=WHITE)
        if i < 2:
            arrow(axes[0], (x + 0.17, 0.695), (stages[i + 1][0], 0.695))
    axes[0].add_patch(Rectangle((0.55, 0.42), 0.36, 0.07, facecolor=ORANGE, edgecolor=INK, alpha=0.80))
    axes[0].text(0.73, 0.36, "short dense window", color=ORANGE, ha="center")
    axes[0].add_patch(Rectangle((0.09, 0.22), 0.82, 0.07, facecolor=BLUE, edgecolor=INK, alpha=0.80))
    axes[0].text(0.50, 0.15, "long-reach package", color=BLUE, ha="center")

    axes[1].set_title("B  Expected diagnostic pattern", loc="left", fontweight="bold")
    box(axes[1], 0.06, 0.63, 0.25, 0.14, "earlier evidence\naddressable", "#e8f1f8")
    box(axes[1], 0.06, 0.27, 0.25, 0.14, "earlier evidence\nnot addressable", "#fff3d6")
    box(axes[1], 0.57, 0.63, 0.34, 0.14, "support can be reconstructed\nif retrieval succeeds", "#e7f5f0")
    box(axes[1], 0.57, 0.27, 0.34, 0.14, "prediction must rely on\nrecent evidence / prior", "#fbe9e7")
    arrow(axes[1], (0.31, 0.70), (0.57, 0.70), color=BLUE)
    arrow(axes[1], (0.31, 0.34), (0.57, 0.34), color=ORANGE)
    axes[1].text(0.50, 0.08, "Conceptual prediction only; empirical effects are reported separately with intervals.", ha="center", color=MUTED)
    save(fig, "fig44_trajectory_consistency")


def make_blt_and_vblt(*, include_vblt: bool = True) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 4.8), constrained_layout=True, gridspec_kw={"height_ratios": [1.0, 0.72]})
    entropy = np.array([0.15] * 10 + [0.82, 0.92, 0.78, 0.66] + [0.20] * 8 + [0.94, 0.84, 0.70] + [0.18] * 9)
    x = np.arange(len(entropy))
    axes[0].bar(x, entropy, color=[RED if e > 0.5 else BLUE for e in entropy], width=0.88)
    axes[0].axhline(0.50, color=MUTED, linestyle="--", linewidth=1.0, label="illustrative split threshold")
    axes[0].set_xlim(-0.8, len(entropy) - 0.2)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("local surprisal")
    axes[0].set_xticks([])
    axes[0].set_title("A  Illustrative local surprisal profile", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)

    axes[1].set_xlim(-0.8, len(entropy) - 0.2)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    patch_spans = [(0, 9, BLUE), (10, 10, RED), (11, 11, RED), (12, 13, RED), (14, 21, BLUE), (22, 22, RED), (23, 23, RED), (24, 24, RED), (25, 33, BLUE)]
    for start, end, color in patch_spans:
        axes[1].add_patch(Rectangle((start - 0.45, 0.42), end - start + 0.90, 0.30, facecolor=color, edgecolor=WHITE, linewidth=1.5, alpha=0.90))
    axes[1].text(4.5, 0.57, "long patch", ha="center", va="center", color=WHITE, fontweight="bold")
    axes[1].text(17.5, 0.57, "long patch", ha="center", va="center", color=WHITE, fontweight="bold")
    axes[1].text(29, 0.57, "long patch", ha="center", va="center", color=WHITE, fontweight="bold")
    axes[1].text(11.7, 0.19, "short patches near surprising bytes", ha="center", color=RED)
    axes[1].text(0, 0.83, "B  Resulting variable-length patches", ha="left", fontweight="bold")
    fig.suptitle("Entropy-guided byte patching allocates more tokens to locally surprising regions", fontweight="bold")
    fig.text(0.50, 0.01, "Conceptual profile; it explains the allocation rule and is not an empirical entropy map.", ha="center", color=MUTED, fontsize=8.5)
    save(fig, "fig45_blt_variable_patching")

    if not include_vblt:
        return

    fig, ax = plt.subplots(figsize=(11.5, 4.2), constrained_layout=True)
    clean(ax)
    title(ax, "Vision Byte Latent Transformer: local adaptation around a global causal core")
    box(ax, 0.03, 0.43, 0.14, 0.16, "raw video\nbytes", "#e6edf4")
    box(ax, 0.23, 0.36, 0.20, 0.30, "local encoder\n\nentropy-guided\nvariable patches", BLUE)
    box(ax, 0.49, 0.36, 0.20, 0.30, "global latent\ntransformer\n\ntime-causal mask", PURPLE)
    box(ax, 0.75, 0.36, 0.20, 0.30, "local decoder\n\n256-way byte\nclassification", GREEN)
    for a, b in [(0.17, 0.23), (0.43, 0.49), (0.69, 0.75)]:
        arrow(ax, (a, 0.51), (b, 0.51))
    ax.text(0.50, 0.13, "local uncertainty controls capacity; the global model handles long-range dependence", ha="center", color=MUTED)
    save(fig, "fig46_vblt_architecture")


def make_rope_nd() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), constrained_layout=True)
    specs = [
        ("horizontal $x$", "$\\theta_x=x\\omega_i$", RED, 0.35),
        ("vertical $y$", "$\\theta_y=y\\omega_i$", GREEN, 0.60),
        ("time $t$", "$\\theta_t=t\\omega_i$", BLUE, 0.88),
    ]
    for index, (ax, (axis_name, formula, color, angle)) in enumerate(zip(axes, specs)):
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.35)
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1.0, facecolor=WHITE, edgecolor=GRID, linewidth=1.2))
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.axvline(0, color=GRID, linewidth=0.8)
        arrow(ax, (0, 0), (np.cos(angle), np.sin(angle)), color=color, width=2.0)
        arc = mpl.patches.Arc((0, 0), 0.85, 0.85, theta1=0, theta2=np.degrees(angle), color=color, linewidth=1.5)
        ax.add_patch(arc)
        ax.text(0.52 * np.cos(angle / 2), 0.52 * np.sin(angle / 2), formula, color=color, ha="center", va="center", fontsize=9)
        ax.set_title(f"{chr(65 + index)}  {axis_name} subspace", loc="left", fontweight="bold")
    fig.suptitle("RoPE-ND uses independent rotary subspaces for each coordinate axis", fontweight="bold")
    fig.text(0.50, 0.01, "$R_x(x) \\oplus R_y(y) \\oplus R_t(t)$: relative displacement is encoded independently along each axis.", ha="center", color=MUTED, fontsize=8.5)
    save(fig, "fig47_rope_nd")


def make_reversible_algorithms() -> None:
    # Figure 2.9: tape-free forward lane and the state that is actually retained.
    fig, ax = plt.subplots(figsize=(11.7, 4.5), constrained_layout=True)
    clean(ax)
    title(ax, "Tape-free forward evolution")
    xs = [0.04, 0.25, 0.46, 0.67, 0.84]
    stages = ["$V_0$", "$\\Phi_1(\\xi_1)$", "$\\Phi_2(\\xi_2)$", "$\\cdots$", "$V_T$"]
    for i, (x, stage) in enumerate(zip(xs, stages)):
        box(
            ax,
            x,
            0.61,
            0.12,
            0.14,
            stage,
            BLUE if i in (1, 2) else PALE,
            text_color=WHITE if i in (1, 2) else INK,
        )
        if i + 1 < len(stages):
            arrow(ax, (x + 0.12, 0.68), (xs[i + 1], 0.68), color=BLUE)
    ax.text(0.03, 0.84, "forward lane", color=BLUE, fontweight="bold")
    for x in xs[1:-1]:
        ax.plot([x + 0.06, x + 0.06], [0.58, 0.50], color=RED, linestyle="--", linewidth=1.0)
        ax.text(x + 0.06, 0.46, "discard", color=RED, ha="center", fontsize=7.5)
    box(ax, 0.22, 0.20, 0.56, 0.14, "retained: terminal state $V_T$ + parameters + replay information $\\xi_{1:T}$", "#e7f5f0")
    arrow(ax, (0.90, 0.61), (0.70, 0.34), color=GREEN)
    ax.text(0.50, 0.08, "Intermediate activation tapes are discarded; required stochastic replay information is retained or reproducibly regenerated.", ha="center", color=MUTED)
    save(fig, "fig48_activation_free_forward")

    # Figure 2.10: reverse state reconstruction (right-to-left) plus one local
    # forward autograd graph.  The two directions are deliberately separated.
    fig, ax = plt.subplots(figsize=(11.7, 5.2), constrained_layout=True)
    clean(ax)
    title(ax, "Reverse replay reconstructs one local differentiation graph at a time")
    ax.text(0.03, 0.80, "state reconstruction", color=PURPLE, fontweight="bold")
    box(ax, 0.78, 0.70, 0.15, 0.14, "$V_t,\\nabla V_t$", "#f7eaf3")
    box(ax, 0.48, 0.70, 0.18, 0.14, "inverse step\n$\\Phi_t^{-1}(\\cdot;\\xi_t)$", PURPLE, text_color=WHITE)
    box(ax, 0.20, 0.70, 0.15, 0.14, "$V_{t-1}$", PALE)
    arrow(ax, (0.78, 0.77), (0.66, 0.77), color=PURPLE)
    arrow(ax, (0.48, 0.77), (0.35, 0.77), color=PURPLE)

    ax.text(0.03, 0.49, "local autograd replay", color=BLUE, fontweight="bold")
    box(ax, 0.20, 0.38, 0.15, 0.14, "$V_{t-1}$\n(requires grad)", PALE)
    box(ax, 0.48, 0.38, 0.18, 0.14, "re-evaluate\n$V_t=\\Phi_t(V_{t-1};\\xi_t)$", BLUE, text_color=WHITE)
    box(ax, 0.78, 0.38, 0.15, 0.14, "local VJP\n+ accumulate", "#e7f5f0")
    arrow(ax, (0.35, 0.45), (0.48, 0.45), color=BLUE)
    arrow(ax, (0.66, 0.45), (0.78, 0.45), color=GREEN)
    arrow(ax, (0.855, 0.38), (0.855, 0.25), color=RED)
    ax.text(0.855, 0.21, "free local graph", color=RED, ha="center")
    ax.text(0.50, 0.10, "Activation tape/local graph: $O(1)$ in $T$.  Stored replay signal: up to $O(TD_{\\xi})$ unless regenerated from a reproducible RNG state.", ha="center", color=MUTED)
    save(fig, "fig49_reverse_local_replay")

    # Figure 2.11: make physical direction and backward direction explicit.
    fig, ax = plt.subplots(figsize=(11.7, 5.0), constrained_layout=True)
    clean(ax)
    title(ax, "Two-phase time-agnostic backpropagation")
    ax.text(0.03, 0.78, "1  forward without activation tape", color=BLUE, fontweight="bold")
    forward_x = [0.08, 0.30, 0.52, 0.74]
    for i, x in enumerate(forward_x):
        box(
            ax,
            x,
            0.66,
            0.13,
            0.13,
            f"$V_{i}$" if i < 3 else "$V_T$",
            PALE if i in (0, 3) else BLUE,
            text_color=INK if i in (0, 3) else WHITE,
        )
        if i + 1 < len(forward_x):
            arrow(ax, (x + 0.13, 0.725), (forward_x[i + 1], 0.725), color=BLUE)
    box(ax, 0.82, 0.45, 0.14, 0.12, "terminal loss\n$\\mathcal{L}(V_T)$", "#fbe9e7")
    arrow(ax, (0.805, 0.66), (0.87, 0.57), color=RED)

    ax.text(0.03, 0.39, "2  reverse reconstruction + local gradients", color=PURPLE, fontweight="bold")
    reverse_x = [0.76, 0.52, 0.28, 0.04]
    reverse_labels = ["$\\nabla V_T$", "replay $t$", "replay $t-1$", "$\\nabla V_0$"]
    for i, (x, label) in enumerate(zip(reverse_x, reverse_labels)):
        box(
            ax,
            x,
            0.25,
            0.13,
            0.13,
            label,
            PURPLE if i in (1, 2) else PALE,
            text_color=WHITE if i in (1, 2) else INK,
        )
        if i + 1 < len(reverse_x):
            arrow(
                ax,
                (x - 0.01, 0.315),
                (reverse_x[i + 1] + 0.145, 0.315),
                color=PURPLE,
                width=1.6,
            )
    arrow(ax, (0.87, 0.45), (0.805, 0.38), color=RED)
    box(ax, 0.40, 0.05, 0.22, 0.11, "accumulated parameter gradients", "#e7f5f0")
    arrow(ax, (0.52, 0.25), (0.51, 0.16), color=GREEN)
    ax.text(0.97, 0.89, "exactness requires transition inversion and replay of realized stochastic inputs", ha="right", color=MUTED, fontsize=8.2)
    save(fig, "fig50_time_agnostic_backprop")


def make_memory_frontier() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    k = np.array([40, 80, 120, 160])
    bptt = np.array([12.2, np.nan, np.nan, np.nan])
    checkpoint = np.array([1.47, 1.63, 1.82, 1.97])
    reversible = np.array([1.10, 1.13, 1.16, 1.18])
    ax.plot(k, checkpoint, "o-", color=ORANGE, label="gradient checkpointing")
    ax.plot(k, reversible, "o-", color=BLUE, label="reversible replay")
    ax.scatter(k[0], bptt[0], color=RED, s=55, label="BPTT")
    ax.annotate("host-RAM spill", (80, 15.2), color=RED, ha="center")
    ax.annotate("OOM", (120, 15.2), color=RED, ha="center", fontweight="bold")
    ax.axhline(16, color=INK, linestyle="--", label="16 GB device")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 22)
    ax.set_xticks(k)
    ax.set_xlabel("rollout length $K$")
    ax.set_ylabel("peak GPU memory (GB, log scale)")
    ax.set_title("Measured memory frontier of the stochastic recurrent core", loc="left", fontweight="bold")
    ax.grid(color=GRID, linewidth=0.6, which="both")
    ax.legend(frameon=False, ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "fig51_reversible_memory_frontier")


def make_reversibility_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(11.7, 5.4), constrained_layout=True)
    clean(ax)
    title(ax, "Evidence-scoped applicability of reversible computation")
    rows = [
        ("network depth", "RevNet-style layer inversion", "layer activations", GREEN,
         "literature-established; used in A2"),
        ("trajectory time", "temporal chaining + stochastic replay", "BPTT activation tape", BLUE,
         "measured: BPTT-matched gradients\nwithin reported tolerance; 1.09–1.18 GB"),
        ("inner optimization", "reversible candidate-state replay", "small trajectory state", RED,
         "tested: saves 14 MB (0.26%) vs checkpointing;\nno practical advantage"),
    ]
    for i, (axis_name, mechanism, target, color, verdict) in enumerate(rows):
        y = 0.72 - i * 0.25
        box(ax, 0.03, y, 0.17, 0.14, axis_name, PALE)
        arrow(ax, (0.20, y + 0.07), (0.27, y + 0.07))
        box(ax, 0.27, y, 0.27, 0.14, mechanism, color, text_color=WHITE)
        arrow(ax, (0.54, y + 0.07), (0.59, y + 0.07))
        box(ax, 0.59, y, 0.17, 0.14, target, PALE)
        ax.text(0.79, y + 0.07, verdict, ha="left", va="center", color=color, fontsize=7.6, fontweight="bold")
    ax.text(
        0.5,
        0.07,
        "The verdicts are restricted to the cited architecture and measured operating regions; they are not universal guarantees.",
        ha="center",
        color=MUTED,
    )
    save(fig, "fig52_reversibility_axes")


def make_dissertation_systems_map() -> None:
    """Map the four article strands without implying one executable pipeline."""
    fig, ax = plt.subplots(figsize=(12.4, 7.3), constrained_layout=True)
    clean(ax)
    title(ax, "Dissertation systems map: four evidence strands under one resource question")

    box(ax, 0.25, 0.85, 0.50, 0.09,
        "Long-horizon video generation under finite representation, state, and training memory",
        PALE, fontsize=9.2)

    lane_x = [0.025, 0.275, 0.525, 0.775]
    lane_w = 0.20
    lanes = [
        ("A1  representation", "V-BLT + RoPE-ND", "allocate representation\nwhere uncertainty is high", BLUE,
         "representation / token budget"),
        ("A2  activation memory", "TempVerseFormer", "reconstruct temporal states\ninstead of storing tapes", GREEN,
         "$M_{activations}$"),
        ("A3  evaluation criteria", "consistency framework", "separate structural,\ninformational, temporal criteria", PURPLE,
         "measurement language"),
        ("A4  retained context", "hierarchical predictor", "compress distant history;\nretain recent detail", ORANGE,
         "$M_{context/state}$"),
    ]
    for x, (heading, method, contribution, color, resource) in zip(lane_x, lanes):
        box(ax, x, 0.64, lane_w, 0.12, heading, color, fontsize=8.6, text_color=WHITE)
        box(ax, x, 0.46, lane_w, 0.11, method, WHITE, fontsize=8.5, edge=color)
        box(ax, x, 0.26, lane_w, 0.13, contribution, PALE, fontsize=8.1, edge=color)
        box(ax, x, 0.10, lane_w, 0.08, resource, WHITE, fontsize=8.2, edge=color)
        arrow(ax, (x + lane_w / 2, 0.64), (x + lane_w / 2, 0.57), color=color)
        arrow(ax, (x + lane_w / 2, 0.46), (x + lane_w / 2, 0.39), color=color)
        arrow(ax, (x + lane_w / 2, 0.26), (x + lane_w / 2, 0.18), color=color)

    # The A3 criteria inform interpretation of the other strands, but are not
    # wired into them as an executable model component.
    for target_x in (lane_x[0] + lane_w / 2, lane_x[1] + lane_w / 2, lane_x[3] + lane_w / 2):
        ax.add_patch(
            FancyArrowPatch(
                (lane_x[2] + lane_w / 2, 0.43),
                (target_x, 0.43),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                linestyle="--",
                color=MUTED,
                connectionstyle="arc3,rad=0.06",
            )
        )

    # Shared problem-to-strand links are methodological: each article addresses
    # a different term of the same resource decomposition.
    for target_x in (x + lane_w / 2 for x in lane_x):
        ax.add_patch(
            FancyArrowPatch(
                (0.50, 0.85),
                (target_x, 0.76),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                linestyle="--",
                color=MUTED,
                connectionstyle="arc3,rad=0.05",
            )
        )

    ax.plot([0.03, 0.09], [0.050, 0.050], color=INK, linewidth=1.6)
    ax.text(0.10, 0.050, "implemented or directly measured dependency", va="center", fontsize=8.2)
    ax.plot([0.54, 0.60], [0.050, 0.050], color=MUTED, linewidth=1.4, linestyle="--")
    ax.text(0.61, 0.050, "conceptual / methodological transfer", va="center", fontsize=8.2)
    ax.text(0.50, 0.010, "Separate evidence boxes: not one integrated executable pipeline.", ha="center", va="center", color=RED, fontsize=8.2)
    save(fig, "fig61_dissertation_systems_map")


def make_two_resource_synthesis() -> None:
    """Place the separately measured context and activation axes side by side."""
    result_path = ROOT / "artifacts" / "confirmation_e2" / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    contrasts = result["frontier"]["contrasts"]
    plotted = [
        (896, "b896-s0", "o"), (896, "b896-s1", "s"),
        (1152, "b1152-s0", "o"), (1152, "b1152-s1", "s"), (1152, "b1152-s2", "^"),
        (1792, "b1792-s0", "o"),
        (2432, "b2432-30k-s0", "D"), (2432, "b2432-60k-s0", "P"),
        (3072, "b3072-s0", "o"), (3072, "b3072-s1", "s"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    ax = axes[0]
    offsets = {"o": 0.975, "s": 1.025, "^": 1.0, "D": 0.975, "P": 1.025}
    for budget, key, marker in plotted:
        row = contrasts[key]
        color = BLUE if row["lo"] > 0 else ORANGE if row["hi"] < 0 else MUTED
        ax.errorbar(
            budget * offsets[marker], row["mean"],
            yerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            color=color, marker=marker, markersize=5.2, capsize=2.5,
            linewidth=1.2, linestyle="none",
        )
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([896, 1152, 1792, 2432, 3072])
    ax.set_xticklabels(["896", "1,152", "1,792", "2,432", "3,072"])
    ax.set_xlabel("context-token budget")
    ax.set_ylabel("flat error − hierarchy error")
    ax.set_title("A  Context/state axis (A4)", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.text(0.97, 0.97, "blue: hierarchy confirmed\ngrey: unresolved\norange: flat confirmed",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2)
    shape_handles = [
        Line2D([], [], color=INK, marker="o", linestyle="none", markersize=5.2, label="seed 0"),
        Line2D([], [], color=INK, marker="s", linestyle="none", markersize=5.2, label="seed 1"),
        Line2D([], [], color=INK, marker="^", linestyle="none", markersize=5.2, label="seed 2"),
        Line2D([], [], color=INK, marker="D", linestyle="none", markersize=5.2, label="B8×30k (s0)"),
        Line2D([], [], color=INK, marker="P", linestyle="none", markersize=5.2, label="B8×60k (s0)"),
    ]
    ax.legend(
        handles=shape_handles, loc="upper right", bbox_to_anchor=(0.98, 0.80),
        ncol=2, frameon=False, fontsize=7.4, handletextpad=0.35, columnspacing=0.8,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    horizons = np.asarray([40, 80, 120, 160])
    checkpoint = np.asarray([1.471, 1.637, 1.801, 1.967])
    reversible = np.asarray([1.091, 1.125, 1.152, 1.176])
    ax.plot(horizons, checkpoint, "s--", color=ORANGE, linewidth=1.6, label="checkpointing")
    ax.plot(horizons, reversible, "o-", color=BLUE, linewidth=1.7, label="reversible replay")
    ax.scatter([40], [12.162], marker="^", s=55, color=RED, label="BPTT (device-resident)")
    ax.scatter([80], [23.305], marker="D", s=48, facecolor=WHITE, edgecolor=RED, linewidth=1.4, label="BPTT (host spill)")
    ax.scatter([120, 160], [16, 16], marker="x", s=52, color=RED, linewidth=1.6)
    ax.text(120, 16.7, "OOM", color=RED, ha="center", fontsize=8)
    ax.text(160, 16.7, "OOM", color=RED, ha="center", fontsize=8)
    ax.axhline(16, color=MUTED, linestyle=":", linewidth=1.1, label="16 GB device")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 25)
    ax.set_xlabel("rollout length $K$")
    ax.set_ylabel("peak training memory (GB)")
    ax.set_title("B  Activation-memory axis (A2)", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle("Two independently measured resource axes", fontweight="bold")
    save(fig, "fig62_two_resource_synthesis")


def make_executed_configuration_matrix() -> None:
    """Summarize the frozen programme, including explicitly unexecuted cells."""
    columns = ["Study / role", "Dataset", "Package / contrast", "Ctx tokens", "Head", "Train seeds", "Codec", "Sampler / NFE", "Evidence state"]
    rows = [
        ["E2 lower reach", "LoopMom holdout", "hier-min / flat-min", "896", "diff", "0–1", "original", "DDIM 16 / s0", "untouched confirmed"],
        ["E2 interior", "LoopMom holdout", "hier-long / flat-budget", "1,152", "diff", "0–2", "original", "DDIM 16 / s0", "untouched confirmed"],
        ["E2 middle", "LoopMom holdout", "hier-c1 / flat-c1", "1,792", "diff", "0", "original", "DDIM 16 / s0", "unresolved"],
        ["E2 upper", "LoopMom holdout", "hier-c2 / flat-c2", "2,432", "diff", "0; 30k/60k", "original", "DDIM 16 / s0", "flat confirmed"],
        ["E2 dense reach", "LoopMom holdout", "hier-rich / flat-long", "3,072", "diff", "0–1", "original", "DDIM 16 / s0", "flat confirmed"],
        ["B1 whole configuration", "LoopMom holdout", "hier-s2-min / flat-short", "512", "diff", "0–1", "original", "DDIM 16 / s0", "configuration confirmed"],
        ["E4b shape × head", "LoopMom development", "temporal 1,088 / spatial 1,152", "near-match", "diff + reg", "0–4", "original", "DDIM 16 / s0", "seed robustness"],
        ["E7a codec", "LoopMom protected", "four registered contrasts", "896–2,432", "diff", "0", "alternate", "DDIM 16 / s0", "codec robustness"],
        ["E7c inference", "LoopMom protected", "four registered contrasts", "896–2,432", "diff", "0", "original", "NFE 8/16/32; s0–2", "sampler robustness"],
        ["E5a reliability", "long-loop hgrid3 set", "long-reach / dense comparator", "registered pair", "diff", "0–2", "original", "registered sampler", "not replicated"],
        ["E8a explicit access", "LoopMom protected", "raw K14/K38/K48 + neural refs", "not matched", "raw + diff refs", "seed-0 refs", "raw / original", "exact pose / DDIM 16", "privileged reference"],
        ["E6 external boundary", "Long Moving-MNIST", "hierarchical / flat", "1,152", "diff", "0–2", "original", "registered sampler", "mixed boundary"],
        ["B1 fixed-target", "LoopMom", "stride-2 / flat-short", "512", "diff", "—", "—", "—", "not executed; claim scoped"],
        ["E9 extension", "natural video", "post-freeze extension", "—", "—", "—", "—", "—", "skipped; no incremental utility"],
    ]

    fig, ax = plt.subplots(figsize=(15.2, 8.7), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Executed experimental coverage and explicit non-executed cells", loc="left", fontweight="bold", pad=14)
    widths = [0.105, 0.105, 0.20, 0.075, 0.065, 0.08, 0.075, 0.13, 0.165]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", colWidths=widths, bbox=[0.0, 0.07, 1.0, 0.86])
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    table.scale(1.0, 1.42)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            status = rows[row - 1][-1]
            if "not executed" in status or "skipped" in status:
                cell.set_facecolor("#fbe9e7")
            elif "unresolved" in status or "not replicated" in status:
                cell.set_facecolor("#f3f4f6")
            else:
                cell.set_facecolor(WHITE if row % 2 else "#f8fafc")
            if col == 8:
                cell.get_text().set_fontweight("bold")
                if "confirmed" in status:
                    cell.get_text().set_color(BLUE if "flat" not in status else ORANGE)
                elif "not executed" in status or "skipped" in status:
                    cell.get_text().set_color(RED)
                else:
                    cell.get_text().set_color(MUTED)
    ax.text(0.0, 0.015,
            "Only executed cells support empirical claims. Dashes and red-tinted rows are structural gaps, not zero effects; no Cartesian completion is implied.",
            transform=ax.transAxes, ha="left", color=MUTED, fontsize=8.5)
    save(fig, "fig63_executed_configuration_matrix")


def main() -> None:
    style()
    make_control_system()
    make_consistency_landscape()
    make_system_decomposition()
    make_visual_hierarchy()
    make_temporal_grid()
    make_fixed_tokenization()
    make_diffusion_pair()
    make_ldm()
    make_long_context_barrier()
    make_representation_choice()
    make_package_compare()
    make_conditioning_schematics()
    make_trajectory_consistency()
    make_blt_and_vblt()
    make_rope_nd()
    make_reversible_algorithms()
    make_memory_frontier()
    make_reversibility_taxonomy()
    make_dissertation_systems_map()
    make_two_resource_synthesis()
    make_executed_configuration_matrix()
    print("DISSERTATION_FOUNDATION_FIGURES_DONE")


if __name__ == "__main__":
    main()
