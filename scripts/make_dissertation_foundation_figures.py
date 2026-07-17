"""Generate deterministic foundation figures for dissertation Chapters 1 and 2.

The drawings are explanatory schematics, not empirical result figures. Measured memory panels
use only the values already reported in the dissertation and their cited repository artifacts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

INK = "#20252b"
MUTED = "#68737e"
GRID = "#d8dde3"
BLUE = "#6f86b6"
ORANGE = "#d9965b"
GREEN = "#70a97f"
PURPLE = "#8a6aaa"
RED = "#b9564f"
PALE = "#eef2f5"


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
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def clean(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, w, h, text, color=PALE, *, fontsize=8.5, edge=INK) -> None:
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
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


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
    fig, ax = plt.subplots(figsize=(11.8, 4.8), constrained_layout=True)
    clean(ax)
    title(ax, "Video generation as an open dynamic control system")
    box(ax, 0.03, 0.57, 0.16, 0.13, "entropy source\n$\\xi_t$ (noise)", "#ebe6f2")
    box(ax, 0.03, 0.31, 0.16, 0.13, "control $u_t$\ncontext + actions", "#e5edf6")
    box(ax, 0.30, 0.40, 0.31, 0.25, "state-transition core\n$x_t=f(x_{t-1},u_t,\\theta,\\xi_t)$\nlatent state + iterative refinement", "#e7eef4")
    box(ax, 0.72, 0.46, 0.20, 0.14, "generated video\n$\\hat{x}_{1:T}$", "#e4efe8")
    for y in (0.635, 0.375):
        arrow(ax, (0.19, y), (0.30, 0.52))
    arrow(ax, (0.61, 0.525), (0.72, 0.525))
    arrow(ax, (0.82, 0.46), (0.50, 0.24), color=ORANGE)
    arrow(ax, (0.50, 0.24), (0.38, 0.40), color=ORANGE)
    ax.text(0.67, 0.22, "autoregressive feedback", color=ORANGE, ha="center")
    arrow(ax, (0.48, 0.83), (0.48, 0.66), color=RED)
    ax.text(0.48, 0.87, "model / data uncertainty", color=RED, ha="center")
    ax.text(
        0.5,
        0.06,
        "Quality is a trajectory property: control, representation and accumulated self-fed state interact.",
        ha="center",
        color=MUTED,
    )
    save(fig, "fig30_control_system")


def make_consistency_landscape() -> None:
    fig = plt.figure(figsize=(8.7, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    x = np.linspace(-2.8, 2.8, 90)
    y = np.linspace(-2.5, 2.5, 80)
    xx, yy = np.meshgrid(x, y)
    z = 0.11 * (xx**2 + yy**2) + 0.32 * np.sin(1.5 * xx) * np.cos(1.2 * yy)
    ax.plot_surface(xx, yy, z, cmap="Blues_r", alpha=0.82, linewidth=0)
    path_x = np.array([-2.25, -1.65, -0.95, -0.25, 0.35, 0.72])
    path_y = np.array([1.75, 1.30, 0.75, 0.25, -0.18, -0.45])
    path_z = 0.11 * (path_x**2 + path_y**2) + 0.32 * np.sin(1.5 * path_x) * np.cos(1.2 * path_y) + 0.06
    ax.plot(path_x, path_y, path_z, color=RED, marker="o", linewidth=2.2)
    ax.text(path_x[0], path_y[0], path_z[0] + 0.18, "noise / high $J$")
    ax.text(path_x[-1], path_y[-1], path_z[-1] + 0.12, "structured / low $J$")
    ax.set_title("Multi-step generation minimizes a consistency criterion $J(x,u)$", loc="left", fontweight="bold")
    ax.set_xlabel("state coordinate 1")
    ax.set_ylabel("state coordinate 2")
    ax.set_zlabel("$J$")
    ax.view_init(30, -58)
    save(fig, "fig31_consistency_landscape")


def make_system_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.5), constrained_layout=True)
    clean(ax)
    title(ax, "Structural decomposition of the video-generation system")
    box(ax, 0.04, 0.48, 0.19, 0.18, "encoder\npixels $\\rightarrow$ latent", "#e6edf4")
    box(ax, 0.34, 0.40, 0.31, 0.34, "iterative generator / predictor\n\nstate update\nconditioning and noise\ninternal refinement loop", "#ebe6f2")
    box(ax, 0.76, 0.48, 0.19, 0.18, "decoder\nlatent $\\rightarrow$ pixels", "#e5efe8")
    arrow(ax, (0.23, 0.57), (0.34, 0.57))
    arrow(ax, (0.65, 0.57), (0.76, 0.57))
    arrow(ax, (0.56, 0.40), (0.42, 0.40), color=PURPLE, style="<|-")
    box(ax, 0.07, 0.79, 0.13, 0.10, "controls", "#f0e8dc")
    arrow(ax, (0.14, 0.79), (0.40, 0.72), color=ORANGE)
    box(ax, 0.43, 0.82, 0.13, 0.10, "noise", "#f0e6ec")
    arrow(ax, (0.50, 0.82), (0.50, 0.74), color=RED)
    ax.text(0.50, 0.17, "representation", ha="center", color=MUTED)
    ax.plot([0.13, 0.50, 0.86], [0.25, 0.25, 0.25], "o-", color=MUTED)
    save(fig, "fig32_system_decomposition")


def make_visual_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    clean(ax)
    title(ax, "Hierarchical decomposition of visual state")
    layers = [
        ((0.40, 0.72), 0.20, 0.14, "MACRO\nsemantics + composition", BLUE),
        ((0.27, 0.44), 0.46, 0.16, "MESO\nobjects + contours + geometry", GREEN),
        ((0.12, 0.13), 0.76, 0.18, "MICRO\ntextures + lighting + pixels", ORANGE),
    ]
    for (x, y), w, h, label, color in layers:
        box(ax, x, y, w, h, label, color)
    arrow(ax, (0.23, 0.34), (0.42, 0.44), color=MUTED)
    arrow(ax, (0.58, 0.44), (0.76, 0.34), color=MUTED)
    ax.text(0.92, 0.80, "higher abstraction\nlower spatial rate", ha="right", color=MUTED)
    ax.text(0.92, 0.06, "higher detail\nhigher spatial rate", ha="right", color=MUTED)
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
        (False, "fig36_forward_diffusion", "Forward diffusion: structure to noise"),
        (True, "fig37_reverse_diffusion", "Reverse diffusion: noise to structure"),
    ]:
        seq = list(reversed(frames)) if reverse else frames
        fig, axes = plt.subplots(1, 6, figsize=(11.5, 2.3), constrained_layout=True)
        for i, (ax, image) in enumerate(zip(axes, seq)):
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"$t={i}$")
            ax.axis("off")
        fig.suptitle(heading, fontsize=12, fontweight="bold")
        save(fig, stem)


def make_ldm() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.2), constrained_layout=True)
    clean(ax)
    title(ax, "Latent diffusion moves iterative refinement into a compact state space")
    box(ax, 0.03, 0.43, 0.15, 0.16, "image $x$", "#e6edf4")
    box(ax, 0.25, 0.43, 0.15, 0.16, "encoder", BLUE)
    box(ax, 0.48, 0.36, 0.24, 0.30, "latent $z_t$\n\nDiT / U-Net\n$T$ denoising steps", "#eae4f1")
    box(ax, 0.79, 0.43, 0.15, 0.16, "decoder", GREEN)
    arrow(ax, (0.18, 0.51), (0.25, 0.51))
    arrow(ax, (0.40, 0.51), (0.48, 0.51))
    arrow(ax, (0.72, 0.51), (0.79, 0.51))
    arrow(ax, (0.64, 0.70), (0.54, 0.70), color=PURPLE, style="<|-")
    ax.text(0.60, 0.75, "iterative latent loop", ha="center", color=PURPLE)
    ax.text(0.50, 0.13, "spatial compression lowers per-step cost; temporal memory remains a separate design problem", ha="center", color=MUTED)
    save(fig, "fig38_latent_diffusion")


def make_long_context_barrier() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.4), constrained_layout=True)
    length = np.linspace(1, 100, 100)
    axes[0].plot(length, 0.007 * length, label="KV memory $\\propto L$", color=BLUE)
    axes[0].plot(length, 0.00013 * length**2, label="attention $\\propto L^2$", color=RED)
    axes[0].axhline(0.56, color=INK, linestyle="--", label="hardware ceiling")
    axes[0].axvline(np.sqrt(0.56 / 0.00013), color=MUTED, linestyle=":")
    axes[0].set_xlabel("flat context length $L$")
    axes[0].set_ylabel("normalized resource")
    axes[0].set_title("A  Resource barrier", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    clean(axes[1])
    axes[1].set_title("B  Re-entry beyond the flat window", loc="left", fontweight="bold")
    axes[1].plot([0.08, 0.92], [0.58, 0.58], color=INK)
    for x, label, color in [(0.15, "seen", GREEN), (0.43, "exit", ORANGE), (0.84, "re-entry ?", RED)]:
        axes[1].scatter(x, 0.58, s=90, color=color, zorder=3)
        axes[1].text(x, 0.68, label, ha="center")
    axes[1].add_patch(Rectangle((0.60, 0.47), 0.28, 0.22, fill=False, edgecolor=RED, linewidth=1.5))
    axes[1].text(0.74, 0.42, "flat window $w$", ha="center", color=RED)
    axes[1].add_patch(Rectangle((0.12, 0.24), 0.76, 0.07, facecolor=BLUE, alpha=0.55, edgecolor=INK))
    axes[1].text(0.50, 0.16, "low-rate coarse memory still reaches the earlier view", ha="center", color=BLUE)
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
    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    clean(ax)
    title(ax, "Equal token capacity can be packaged for reach or for local fidelity")
    labels = [
        ("flat contiguous", [(0.12, 0.74, 0.58, 0.10, ORANGE)]),
        ("spatial hierarchy", [(0.12, 0.48, 0.58, 0.07, BLUE), (0.57, 0.57, 0.13, 0.09, ORANGE)]),
        ("temporal stride", [(0.12 + i * 0.058, 0.25, 0.026, 0.10, GREEN) for i in range(10)]),
    ]
    for row, (label, segments) in enumerate(labels):
        y_label = 0.79 - row * 0.25
        ax.text(0.02, y_label, label, va="center", fontweight="bold")
        for x, y, w, h, color in segments:
            ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=INK, linewidth=0.7))
    ax.text(0.12, 0.89, "older", color=MUTED)
    ax.text(0.70, 0.89, "recent", ha="right", color=MUTED)
    box(ax, 0.79, 0.31, 0.17, 0.34, "shared predictor\njoint attention\nfixed target interface", "#eae4f1")
    for y in (0.79, 0.54, 0.29):
        arrow(ax, (0.71, y), (0.79, 0.50), color=PURPLE)
    ax.text(0.45, 0.08, "same capacity, different reach × fidelity allocation", ha="center", color=MUTED)
    save(fig, "fig40_context_package_comparison")


def make_conditioning_schematics() -> None:
    specs = [
        (
            "fig41_image_to_image",
            "Image-to-image conditioning trades variation against identity",
            ["reference image", "partial noising\nstrength $s$", "conditional\ndenoising", "output"],
            "higher $s$ → more variation, weaker identity retention",
        ),
        (
            "fig42_controlnet",
            "ControlNet constrains generation with a structural control branch",
            ["structural signal\nedge / depth", "trainable\ncontrol branch", "frozen diffusion\nbackbone", "output"],
            "boundary/geometry control narrows the admissible state trajectory",
        ),
        (
            "fig43_shared_attention",
            "Shared attention transfers style while target content remains distinct",
            ["reference\nstyle features", "shared K,V\nattention", "target prompts\n1…N", "styled outputs\n1…N"],
            "shared style representation; separate content trajectories",
        ),
    ]
    for stem, heading, stages, note in specs:
        fig, ax = plt.subplots(figsize=(11.4, 3.8), constrained_layout=True)
        clean(ax)
        title(ax, heading)
        colors = (BLUE, PURPLE, ORANGE, GREEN)
        xs = (0.03, 0.28, 0.53, 0.78)
        for x, stage, color in zip(xs, stages, colors):
            box(ax, x, 0.43, 0.17, 0.19, stage, color)
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(ax, (a + 0.17, 0.525), (b, 0.525))
        ax.text(0.50, 0.16, note, ha="center", color=MUTED)
        save(fig, stem)


def make_trajectory_consistency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    h = np.arange(1, 101)
    flat = 0.03 + 0.0005 * h + 0.05 / (1 + np.exp(-(h - 48) / 6))
    hier = 0.03 + 0.0005 * h + 0.018 / (1 + np.exp(-(h - 48) / 6))
    axes[0].plot(h, flat, color=ORANGE, label="flat short window")
    axes[0].plot(h, hier, color=BLUE, label="long-reach package")
    axes[0].axvline(48, color=INK, linestyle="--", label="window boundary")
    axes[0].set_xlabel("rollout horizon $h$")
    axes[0].set_ylabel("schematic re-entry error")
    axes[0].set_title("A  Error separates on memory-relevant records", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    clean(axes[1])
    axes[1].set_title("B  Re-entry event", loc="left", fontweight="bold")
    for i, (x, label) in enumerate([(0.10, "visible"), (0.38, "occluded"), (0.70, "re-entry")]):
        box(axes[1], x, 0.55, 0.18, 0.14, label, PALE)
        if i < 2:
            arrow(axes[1], (x + 0.18, 0.62), (x + 0.28, 0.62))
    axes[1].text(0.79, 0.38, "long reach:\nrestores earlier structure", ha="center", color=BLUE)
    axes[1].text(0.79, 0.19, "short window:\nguess / blur", ha="center", color=ORANGE)
    save(fig, "fig44_trajectory_consistency")


def make_blt_and_vblt() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.1), constrained_layout=True)
    clean(ax)
    title(ax, "Entropy-driven byte patching allocates more tokens to surprising regions")
    rng = np.random.default_rng(13)
    entropy = np.array([0.15] * 14 + [0.8, 0.9, 0.75, 0.65] + [0.2] * 10 + [0.95, 0.85, 0.7] + [0.18] * 11)
    bytes_ = rng.integers(0, 255, len(entropy))
    ax.bar(np.arange(len(bytes_)), entropy, color=[RED if e > 0.5 else BLUE for e in entropy], width=0.9)
    ax.set_xlim(-1, len(bytes_))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("byte position")
    ax.set_ylabel("local entropy")
    ax.text(7, 0.94, "large latent patch", ha="center", color=BLUE)
    ax.text(15.5, 0.98, "small patches", ha="center", color=RED)
    ax.text(33, 0.98, "small patches", ha="center", color=RED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "fig45_blt_variable_patching")

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
    fig = plt.figure(figsize=(9.2, 5.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    t = np.arange(6)
    x = np.arange(5)
    y = np.arange(4)
    tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
    colors = mpl.colormaps["viridis"](tt.ravel() / max(t))
    ax.scatter(xx, yy, tt, c=colors, s=32)
    ax.quiver(0, 0, 0, 4, 0, 0, color=RED, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 3, 0, color=GREEN, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 0, 5, color=BLUE, arrow_length_ratio=0.08)
    ax.set_xlabel("horizontal $x$")
    ax.set_ylabel("vertical $y$")
    ax.set_zlabel("time $t$")
    ax.set_title("RoPE-ND assigns independent rotations to spatial and temporal coordinates", loc="left", fontweight="bold")
    save(fig, "fig47_rope_nd")


def make_reversible_algorithms() -> None:
    specs = [
        (
            "fig48_activation_free_forward",
            "Activation-free forward evolution",
            ["encode $V_0$", "reversible\n$\\Phi_1$", "reversible\n$\\Phi_2$", "…", "final $V_T$"],
            "retain terminal state, parameters and random seeds; discard intermediate activations",
        ),
        (
            "fig49_reverse_local_replay",
            "Reverse replay reconstructs one local graph at a time",
            ["$V_t,\\nabla V_t$", "invert\n$\\Phi_t^{-1}$", "rebuild local\ngraph", "accumulate\ngradients", "free + $t-1$"],
            "memory does not scale with trajectory length; recomputation increases wall time",
        ),
        (
            "fig50_time_agnostic_backprop",
            "Two-phase time-agnostic backpropagation",
            ["forward\nwithout tape", "terminal\nloss", "reverse\nreconstruction", "local\ngradients", "parameter\nupdate"],
            "exactness requires reversible transitions and replay of the realized stochastic noise",
        ),
    ]
    for stem, heading, stages, note in specs:
        fig, ax = plt.subplots(figsize=(11.7, 3.8), constrained_layout=True)
        clean(ax)
        title(ax, heading)
        xs = np.linspace(0.02, 0.82, len(stages))
        for i, (x, stage) in enumerate(zip(xs, stages)):
            box(ax, float(x), 0.43, 0.15, 0.18, stage, (BLUE, PURPLE, GREEN, ORANGE, PALE)[i])
            if i + 1 < len(stages):
                arrow(ax, (float(x) + 0.15, 0.52), (float(xs[i + 1]), 0.52))
        ax.text(0.50, 0.16, note, ha="center", color=MUTED)
        save(fig, stem)


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
    fig, ax = plt.subplots(figsize=(11.7, 5.0), constrained_layout=True)
    clean(ax)
    title(ax, "Reversibility helps along depth and time, but not the dominant inner-optimization state")
    rows = [
        ("network depth", "RevNet-style layer inversion", "activation memory", GREEN, "pays"),
        ("trajectory time", "temporal chaining + replayed noise", "BPTT trajectory memory", BLUE, "pays"),
        ("inner optimization", "candidate update trajectory", "optimizer / model base state", RED, "does not pay here"),
    ]
    for i, (axis_name, mechanism, target, color, verdict) in enumerate(rows):
        y = 0.72 - i * 0.25
        box(ax, 0.03, y, 0.17, 0.14, axis_name, PALE)
        arrow(ax, (0.20, y + 0.07), (0.27, y + 0.07))
        box(ax, 0.27, y, 0.29, 0.14, mechanism, color)
        arrow(ax, (0.56, y + 0.07), (0.63, y + 0.07))
        box(ax, 0.63, y, 0.22, 0.14, target, PALE)
        ax.text(0.94, y + 0.07, verdict, ha="right", va="center", color=color, fontweight="bold")
    ax.text(
        0.5,
        0.07,
        "Measured inner loop: 5.282–5.296 GB for truncate/checkpoint/reversible; reversible saves 14 MB (0.26%) at ≈2× time.",
        ha="center",
        color=MUTED,
    )
    save(fig, "fig52_reversibility_axes")


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
    print("DISSERTATION_FOUNDATION_FIGURES_DONE")


if __name__ == "__main__":
    main()
