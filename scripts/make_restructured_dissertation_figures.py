"""Generate dissertation-native figures and extract unchanged published assets.

The native schematics use a common white-background visual grammar. Published
images are copied or losslessly reconstructed from official PDF image objects
and are never redrawn by this script. The script writes a SHA-256 provenance
ledger beside the figures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "restructured"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1F4E79"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
INK = "#1F2937"
MUTED = "#5F6B7A"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#E8F5F2"
PALE_ORANGE = "#FCEFEA"
PALE_GRAY = "#F4F6F8"

SCHEMA_VERSION = "1.0"
GENERATOR = "scripts/make_restructured_dissertation_figures.py"
NATIVE_METADATA = {
    "ch1_fig1_1_system_boundary": (
        "1.1",
        "author-generated-conceptual",
        "Partially observed controlled-process formulation in Chapter 1",
        "system boundary separating latent world evolution, observations, retained context, prediction and feedback",
    ),
    "ch1_fig1_2_hidden_viewport": (
        "1.2",
        "author-generated-conceptual",
        "Hidden-state and viewport argument in Chapter 1",
        "storyboard distinguishing persistent world state from the currently visible crop",
    ),
    "ch1_fig1_3_teacher_self_fed": (
        "1.3",
        "author-generated-conceptual",
        "Teacher-conditioning and self-fed rollout equations in Chapter 1",
        "causal comparison of observed-input training and generated-input rollout",
    ),
    "ch1_fig1_4_resource_locations": (
        "1.4",
        "author-generated-conceptual",
        "Systems memory decomposition in Chapter 1",
        "location map for retained context, activations, optimizer state and transient workspace",
    ),
    "ch1_fig1_6b_video_prediction_example": (
        "1.6b",
        "author-generated-conceptual",
        "Static-to-video transition in Chapter 1",
        "conceptual observed-and-predicted sequence illustrating motion, disappearance and memory-dependent re-entry",
    ),
    "ch1_fig1_6_consistency_taxonomy": (
        "1.7",
        "author-generated-conceptual",
        "Consistency taxonomy synthesized from the Chapter 1 literature review",
        "overlapping appearance, identity, geometry and concept criteria",
    ),
    "ch1_fig1_7_static_to_video": (
        "1.12",
        "author-generated-conceptual",
        "Static-to-video consistency synthesis in Chapter 1",
        "mapping from static-image invariants to temporal obligations",
    ),
    "ch1_fig1_8_failure_montage": (
        "1.9",
        "author-generated-conceptual",
        "Failure taxonomy in Chapter 1",
        "conceptual examples of distinct long-horizon failure modes; not model outputs",
    ),
    "ch1_fig1_8_trajectory_consistency": (
        "1.10",
        "author-generated-conceptual",
        "Temporal smoothness, trajectory and re-entry criteria in Chapter 1",
        "operational separation of local smoothness, global trajectory correctness and delayed identity recovery",
    ),
    "ch1_fig1_9_cross_level_consistency": (
        "1.11",
        "author-generated-conceptual",
        "Cross-level consistency formulation in Chapter 1",
        "contrast between compatible and contradictory coarse/fine predictions",
    ),
    "ch1_fig1_11_control_interfaces": (
        "1.20",
        "author-generated-conceptual",
        "Consistency-control interface synthesis in Chapter 1",
        "five interfaces through which consistency information enters a generator",
    ),
    "ch1_fig1_12_support_identity_bridge": (
        "1.21",
        "author-generated-conceptual",
        "Support-versus-identity boundary in Chapters 1 and 4",
        "independent support and identity outcomes",
    ),
    "ch1_fig1_15_activation_tape": (
        "1.22",
        "author-generated-conceptual",
        "Ordinary temporal backpropagation formulation in Chapter 1",
        "activation-tape growth over a differentiated temporal unroll",
    ),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "text.color": INK,
        "axes.edgecolor": INK,
    }
)


def canvas(width: float = 12, height: float = 6.5):
    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, *, fc=PALE_BLUE, ec=BLUE, lw=1.8, fs=10, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def arrow(ax, start, end, *, color=INK, lw=1.6, style="-|>", dashed=False, curve=0.0):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)
    return patch


def title(ax, text, subtitle=None):
    ax.text(0.5, 0.965, text, ha="center", va="top", fontsize=15, weight="bold", color=INK)
    if subtitle:
        ax.text(0.5, 0.915, subtitle, ha="center", va="top", fontsize=9.5, color=MUTED)


def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def viewport(ax, x, y, w, h, *, offset=0.0, label="observed viewport"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PALE_GRAY, edgecolor=INK, linewidth=1.4))
    for gx in np.linspace(x + 0.05, x + w - 0.05, 6):
        ax.plot([gx, gx], [y, y + h], color="#D5DAE0", lw=0.6)
    for gy in np.linspace(y + 0.04, y + h - 0.04, 4):
        ax.plot([x, x + w], [gy, gy], color="#D5DAE0", lw=0.6)
    cx = x + w * (0.25 + offset)
    cy = y + h * 0.53
    tri = Polygon([[cx, cy + 0.07], [cx - 0.055, cy - 0.055], [cx + 0.06, cy - 0.04]],
                  closed=True, facecolor=ORANGE, edgecolor="#A33B20", linewidth=1.2)
    ax.add_patch(tri)
    ax.text(x + w / 2, y - 0.035, label, ha="center", va="top", fontsize=8.5, color=MUTED)


def fig1_1_system_boundary():
    fig, ax = canvas(13, 6.8)
    title(
        ax,
        "Довгогоризонтне прогнозування є задачею зі зворотним зв’язком і частковим спостереженням",
        "Стан світу змінюється поза предиктором і поточним полем огляду",
    )

    box(ax, 0.05, 0.55, 0.17, 0.15, "прихований стан світу\n$s_t$", fc=PALE_TEAL, ec=TEAL, weight="bold", fs=10)
    box(ax, 0.31, 0.55, 0.17, 0.15, "спостереження / поле огляду\n$y_t = O(s_t)$", fc=PALE_BLUE, ec=BLUE, fs=9.4)
    box(ax, 0.57, 0.55, 0.16, 0.15, "формування контексту\n$m_t=C_t(y_{\\leq t},u_{\\leq t})$", fc=PALE_BLUE, ec=BLUE, fs=9.0)
    box(ax, 0.82, 0.55, 0.14, 0.15, "предиктор\n$p_\\theta(y_{t+1:t+H}|m_t)$", fc=PALE_ORANGE, ec=ORANGE, fs=9.2)
    arrow(ax, (0.22, 0.625), (0.31, 0.625), color=TEAL)
    arrow(ax, (0.48, 0.625), (0.57, 0.625), color=BLUE)
    arrow(ax, (0.73, 0.625), (0.82, 0.625), color=BLUE)

    box(ax, 0.08, 0.24, 0.19, 0.13, "перехід стану світу\n$s_{t+1}=F(s_t,u_t,\\xi_t)$", fc="white", ec=TEAL, fs=9.1)
    box(ax, 0.39, 0.24, 0.18, 0.13, "керувальний сигнал\n$u_t$", fc="white", ec=GOLD, fs=9.5)
    box(ax, 0.71, 0.24, 0.19, 0.13, "згенерований зворотний зв’язок\n$\\hat y_{t+1}$", fc="white", ec=ORANGE, fs=8.7)
    arrow(ax, (0.145, 0.54), (0.16, 0.38), color=TEAL)
    arrow(ax, (0.18, 0.38), (0.18, 0.54), color=TEAL, curve=0.45)
    arrow(ax, (0.48, 0.38), (0.23, 0.30), color=GOLD, curve=0.08)
    arrow(ax, (0.48, 0.38), (0.65, 0.54), color=GOLD, curve=-0.08)
    arrow(ax, (0.89, 0.54), (0.81, 0.38), color=ORANGE)
    arrow(ax, (0.71, 0.305), (0.65, 0.54), color=ORANGE, curve=-0.18)

    ax.text(0.17, 0.17, "поза моделлю", ha="center", color=TEAL, weight="bold")
    ax.text(0.67, 0.17, "інтерфейс збереженого стану та прогнозування", ha="center", color=BLUE, weight="bold")
    ax.plot([0.285, 0.285], [0.15, 0.77], color=MUTED, lw=1.2, linestyle="--")
    ax.text(
        0.50,
        0.075,
        "Під час самоподачі згенеровані спостереження змінюють подальший контекст, але не замінюють перехід прихованого стану світу.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "ch1_fig1_1_system_boundary")


def fig1_2_hidden_viewport():
    fig, ax = canvas(12, 5.8)
    title(ax, "The world state persists outside the observed viewport",
          "A later prediction can depend on appearance that is absent from every recent frame")
    ax.add_patch(Rectangle((0.05, 0.18), 0.90, 0.62, facecolor="#F8FAFC", edgecolor=INK, linewidth=1.6))
    ax.text(0.075, 0.76, "latent cyclic world state", fontsize=10, weight="bold")
    positions = [0.14, 0.35, 0.58, 0.82]
    labels = ["t = 0: visible", "t = 8: leaving", "t = 20: hidden", "t = 40: re-entry"]
    for i, (x, lab) in enumerate(zip(positions, labels)):
        ax.add_patch(Rectangle((x - 0.08, 0.32), 0.16, 0.28, fill=False, edgecolor=BLUE, linewidth=2))
        cx = [x - 0.02, x + 0.07, x + 0.12, x - 0.055][i]
        tri = Polygon([[cx, 0.53], [cx - 0.025, 0.43], [cx + 0.035, 0.44]], closed=True,
                      facecolor=ORANGE, edgecolor="#A33B20")
        ax.add_patch(tri)
        ax.text(x, 0.275, lab, ha="center", fontsize=9)
        if i < 3:
            arrow(ax, (x + 0.09, 0.46), (positions[i + 1] - 0.09, 0.46), color=TEAL)
    ax.text(0.5, 0.12, "The observation window loses the object; retained state determines whether its earlier identity is still usable.",
            ha="center", fontsize=10, color=INK)
    save(fig, "ch1_fig1_2_hidden_viewport")


def fig1_3_teacher_self_fed():
    fig, ax = canvas(12, 6.2)
    title(ax, "Teacher-conditioned training and self-fed evaluation",
          "Solid blue arrows carry observations; orange arrows carry generated feedback")
    ax.text(0.04, 0.83, "Teacher-conditioned", weight="bold", color=BLUE, fontsize=11)
    ax.text(0.04, 0.41, "Self-fed rollout", weight="bold", color=ORANGE, fontsize=11)
    xs = [0.22, 0.42, 0.62, 0.82]
    for i, x in enumerate(xs):
        box(ax, x - 0.06, 0.68, 0.12, 0.12, f"y{ i }", fc=PALE_BLUE, ec=BLUE, weight="bold")
        box(ax, x - 0.06, 0.26, 0.12, 0.12, "y0" if i == 0 else rf"$\hat{{y}}_{i}$",
            fc=PALE_BLUE if i == 0 else PALE_ORANGE, ec=BLUE if i == 0 else ORANGE, weight="bold")
        if i < len(xs) - 1:
            arrow(ax, (x + 0.07, 0.74), (xs[i + 1] - 0.07, 0.74), color=BLUE)
            arrow(ax, (x + 0.07, 0.32), (xs[i + 1] - 0.07, 0.32), color=ORANGE)
    ax.text(0.90, 0.74, "observed\ninputs", ha="left", va="center", color=MUTED, fontsize=8.5)
    ax.text(0.90, 0.32, "generated\ninputs", ha="left", va="center", color=ORANGE, fontsize=8.5)
    ax.text(0.50, 0.09, "A local deviation becomes both an output error and a downstream context shift.",
            ha="center", fontsize=10)
    save(fig, "ch1_fig1_3_teacher_self_fed")


def fig1_4_resource_locations():
    fig, ax = canvas(12, 6.5)
    title(ax, "Ресурсні складники розміщені на різних шляхах системи")
    box(ax, 0.05, 0.55, 0.14, 0.13, "збережений\nконтекст / стан", fc=PALE_TEAL, ec=TEAL, weight="bold", fs=9.2)
    box(ax, 0.25, 0.55, 0.15, 0.13, "кодер /\nкодек", fc=PALE_BLUE, ec=BLUE)
    box(ax, 0.47, 0.55, 0.15, 0.13, "параметри\nпредиктора", fc=PALE_BLUE, ec=BLUE)
    box(ax, 0.69, 0.55, 0.13, 0.13, "вибірка /\nрозгортання", fc=PALE_ORANGE, ec=ORANGE)
    box(ax, 0.86, 0.55, 0.10, 0.13, "вихід", fc=PALE_GRAY, ec=INK)
    for a, b in [((0.19, 0.615), (0.25, 0.615)), ((0.40, 0.615), (0.47, 0.615)),
                 ((0.62, 0.615), (0.69, 0.615)), ((0.82, 0.615), (0.86, 0.615))]:
        arrow(ax, a, b)
    arrow(ax, (0.91, 0.54), (0.12, 0.47), color=ORANGE, curve=-0.22)
    ax.text(0.52, 0.40, "згенерований зворотний зв’язок оновлює контекст", ha="center", color=ORANGE, fontsize=9)
    ax.plot([0.25, 0.82], [0.28, 0.28], color=BLUE, lw=3)
    ax.text(0.535, 0.315, "диференційований шлях навчання", ha="center", color=BLUE, weight="bold")
    for x, label, color in [(0.32, "стрічка\nактивацій", ORANGE), (0.53, "стан\nоптимізатора", TEAL), (0.74, "тимчасова\nробоча пам’ять", MUTED)]:
        box(ax, x - 0.07, 0.13, 0.14, 0.11, label, fc="white", ec=color)
        arrow(ax, (x, 0.25), (x, 0.28), color=color)
    ax.text(0.5, 0.055, "Реверсивність змінює зберігання активацій, а ієрархія – збережений контекст. Жодна операція не випливає з іншої.",
            ha="center", fontsize=9.5)
    save(fig, "ch1_fig1_4_resource_locations")


def fig1_6b_video_prediction_example():
    fig, ax = canvas(13, 5.4)
    title(
        ax,
        "Video prediction adds continuity, trajectory and delayed recovery",
        "Conceptual sequence only: panels are not selected model outputs",
    )

    frame_y, frame_h, frame_w = 0.29, 0.43, 0.125
    xs = [0.035, 0.185, 0.335, 0.535, 0.685, 0.835]
    labels = ["$t-2$", "$t-1$", "$t$", "$t+1$", "$t+2$", "$t+3$"]
    object_x = [0.18, 0.43, 0.72, 0.91, None, 0.18]

    for j, (x, label, ox) in enumerate(zip(xs, labels, object_x)):
        ax.add_patch(Rectangle((x, frame_y), frame_w, frame_h, facecolor=PALE_GRAY,
                               edgecolor=BLUE if j < 3 else ORANGE, linewidth=1.6))
        ax.plot([x, x + frame_w], [frame_y + 0.10, frame_y + 0.10], color="#CBD5E1", lw=1.0)
        ax.text(x + frame_w / 2, frame_y - 0.035, label, ha="center", va="top",
                fontsize=10, weight="bold")
        if ox is not None:
            cx = x + frame_w * ox
            cy = frame_y + 0.22
            tri = Polygon(
                [[cx, cy + 0.065], [cx - 0.04, cy - 0.045], [cx + 0.045, cy - 0.04]],
                closed=True, facecolor=ORANGE, edgecolor=INK, linewidth=1.3,
            )
            ax.add_patch(tri)
            ax.add_patch(Circle((cx + 0.002, cy + 0.008), 0.010, facecolor=GOLD,
                                edgecolor=INK, linewidth=0.8))
        else:
            ax.text(x + frame_w / 2, frame_y + 0.23, "outside\nviewport", ha="center",
                    va="center", fontsize=9, color=MUTED)

    ax.plot([0.485, 0.485], [0.22, 0.79], color=MUTED, lw=1.2, linestyle="--")
    ax.text(0.26, 0.80, "observed context", ha="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(0.73, 0.80, "predicted continuation", ha="center", fontsize=11, color=ORANGE, weight="bold")
    arrow(ax, (0.10, 0.23), (0.90, 0.23), color=MUTED, lw=1.2)
    ax.text(
        0.50,
        0.10,
        "A locally plausible next frame is insufficient: the sequence must preserve motion and recover the same object after it leaves view.",
        ha="center", va="center", fontsize=10.2, color=INK,
    )
    save(fig, "ch1_fig1_6b_video_prediction_example")


def fig1_6_consistency_taxonomy():
    fig, ax = canvas(11, 7)
    title(ax, "Consistency requirements overlap")
    circles = [
        (0.36, 0.58, 0.23, PALE_BLUE, BLUE, "appearance\nstyle, texture, light"),
        (0.59, 0.58, 0.23, PALE_TEAL, TEAL, "identity\ninstance, class"),
        (0.42, 0.36, 0.23, "#FFF7DF", "#B7791F", "geometry\nshape, layout"),
        (0.64, 0.34, 0.23, PALE_ORANGE, ORANGE, "concept\nrelation, event"),
    ]
    for x, y, r, fc, ec, txt in circles:
        ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=2, alpha=0.78))
        ax.text(x, y, txt, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0.51, 0.49, "task-defined\ninvariant", ha="center", va="center", fontsize=11, weight="bold", color=INK)
    ax.text(0.5, 0.08, "A generated sequence may satisfy one dimension and fail another; no circle is a universal class boundary.",
            ha="center", fontsize=10)
    save(fig, "ch1_fig1_6_consistency_taxonomy")


def fig1_7_static_to_video():
    fig, ax = canvas(12, 6.5)
    title(ax, "Static-image criteria acquire temporal obligations in video")
    left = [("appearance", 0.74), ("identity", 0.58), ("geometry", 0.42), ("concept", 0.26)]
    right = [("appearance over time", 0.78), ("re-entry identity", 0.64), ("support + continuity", 0.50),
             ("trajectory", 0.36), ("event semantics", 0.22), ("cross-level agreement", 0.08)]
    ax.text(0.21, 0.88, "related static images", ha="center", weight="bold", color=BLUE)
    ax.text(0.79, 0.88, "self-fed video sequence", ha="center", weight="bold", color=TEAL)
    for txt, y in left:
        box(ax, 0.08, y - 0.05, 0.25, 0.10, txt, fc=PALE_BLUE, ec=BLUE)
    for txt, y in right:
        box(ax, 0.66, y - 0.045, 0.27, 0.09, txt, fc=PALE_TEAL, ec=TEAL)
    mapping = [(0.74, 0.78), (0.58, 0.64), (0.42, 0.50), (0.42, 0.36), (0.26, 0.22), (0.26, 0.08)]
    for y1, y2 in mapping:
        arrow(ax, (0.34, y1), (0.65, y2), color=MUTED, lw=1.2)
    ax.text(0.50, 0.82, "+ order, feedback, disappearance, delay", ha="center", va="center",
            color=ORANGE, weight="bold", fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0}, zorder=10)
    save(fig, "ch1_fig1_7_static_to_video")


def fig1_8_failure_montage():
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), facecolor="white")
    labels = ["appearance drift", "identity swap", "support loss", "trajectory error", "extra foreground"]
    colors = [ORANGE, TEAL, BLUE, GOLD, ORANGE]
    for j, ax in enumerate(axes):
        ax.set_facecolor("white")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0.04, 0.18), 0.92, 0.66, facecolor=PALE_GRAY, edgecolor="#CBD2D9"))
        xs = [0.18, 0.39, 0.60, 0.81]
        for i, x in enumerate(xs):
            if j == 2 and i >= 2:
                continue
            y = 0.48 + (0.06 * i if j == 3 else 0)
            if j == 1 and i >= 2:
                shape = Circle((x, y), 0.07, facecolor=colors[j], edgecolor=INK)
            else:
                c = colors[j] if not (j == 0 and i >= 2) else "#9B6B9E"
                shape = Polygon([[x, y + 0.09], [x - 0.07, y - 0.07], [x + 0.07, y - 0.06]],
                                closed=True, facecolor=c, edgecolor=INK)
            ax.add_patch(shape)
        if j == 4:
            ax.add_patch(Circle((0.72, 0.73), 0.05, facecolor=ORANGE, edgecolor=INK))
        ax.set_title(labels[j], fontsize=10, weight="bold", color=INK)
    fig.suptitle("Plausible frames can fail for different reasons", fontsize=15, weight="bold", y=1.02)
    fig.text(0.5, 0.04, "Conceptual examples; they are not selected model outputs.", ha="center", color=MUTED, fontsize=9)
    save(fig, "ch1_fig1_8_failure_montage")


def fig1_8_trajectory_consistency():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), facecolor="white")
    headings = [
        "Local temporal smoothness",
        "Global trajectory correctness",
        "Delayed re-entry identity",
    ]
    for ax, heading in zip(axes, headings):
        ax.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0.05, 0.18), 0.90, 0.66, facecolor=PALE_GRAY, edgecolor="#CBD2D9"))
        ax.set_title(heading, fontsize=10.5, weight="bold", color=INK, pad=8)

    # Smooth adjacent displacements can follow the wrong global path.
    gt_x = np.linspace(0.14, 0.86, 40)
    gt_y = 0.48 + 0.12 * np.sin(np.linspace(0, np.pi, 40))
    pred_y = 0.38 + 0.10 * np.sin(np.linspace(0, np.pi, 40))
    axes[0].plot(gt_x, gt_y, color=TEAL, lw=2.3, linestyle="--", label="target trajectory")
    axes[0].plot(gt_x, pred_y, color=ORANGE, lw=2.6, label="smooth prediction")
    for idx in (4, 14, 25, 35):
        axes[0].add_patch(Circle((gt_x[idx], pred_y[idx]), 0.026, facecolor=ORANGE, edgecolor=INK, lw=0.8))
    axes[0].text(0.5, 0.10, "smooth: pass   trajectory: fail", ha="center", fontsize=9, color=ORANGE, weight="bold")

    # A locally uneven path may nevertheless end close to the correct trajectory.
    target_x = np.linspace(0.14, 0.86, 50)
    target_y = 0.42 + 0.22 * target_x
    pred_x = np.array([0.14, 0.26, 0.33, 0.52, 0.57, 0.74, 0.86])
    pred_y = 0.42 + 0.22 * pred_x + np.array([0.00, 0.05, -0.03, 0.04, -0.02, 0.02, 0.00])
    axes[1].plot(target_x, target_y, color=TEAL, lw=2.3, linestyle="--")
    axes[1].plot(pred_x, pred_y, color=ORANGE, lw=2.2, marker="o", markersize=4)
    axes[1].text(0.5, 0.10, "smooth: imperfect   trajectory: pass", ha="center", fontsize=9, color=TEAL, weight="bold")

    # The object is absent during the central interval and identity is tested on return.
    axes[2].add_patch(Rectangle((0.12, 0.32), 0.22, 0.38, facecolor="white", edgecolor=BLUE, lw=1.6))
    axes[2].add_patch(Rectangle((0.66, 0.32), 0.22, 0.38, facecolor="white", edgecolor=BLUE, lw=1.6))
    axes[2].add_patch(Polygon([[0.23, 0.62], [0.16, 0.42], [0.31, 0.43]], closed=True,
                              facecolor=TEAL, edgecolor=INK, lw=1.0))
    axes[2].add_patch(Circle((0.77, 0.51), 0.09, facecolor=ORANGE, edgecolor=INK, lw=1.0))
    arrow(axes[2], (0.35, 0.51), (0.64, 0.51), color=MUTED, dashed=True)
    axes[2].text(0.50, 0.58, "object unobserved", ha="center", fontsize=8.5, color=MUTED)
    axes[2].text(0.5, 0.10, "support: pass   identity: fail", ha="center", fontsize=9, color=ORANGE, weight="bold")

    handles = [
        Line2D([0], [0], color=TEAL, linestyle="--", lw=2.2, label="target"),
        Line2D([0], [0], color=ORANGE, lw=2.2, label="prediction"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Local smoothness, trajectory, and re-entry identity answer different questions",
                 fontsize=15, weight="bold", y=0.99)
    fig.text(0.5, 0.015, "Conceptual diagnostic examples; no panel is an empirical model result.",
             ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.78, bottom=0.12, wspace=0.12)
    save(fig, "ch1_fig1_8_trajectory_consistency")


def fig1_9_cross_level_consistency():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor="white")
    specs = [
        ("Compatible levels", TEAL, False),
        ("Contradictory levels", ORANGE, True),
    ]
    for ax, (heading, colour, contradict) in zip(axes, specs):
        ax.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(heading, fontsize=12, weight="bold", color=colour, pad=10)
        rows = [(0.70, "coarse / macro"), (0.44, "middle / meso"), (0.18, "fine / micro")]
        for idx, (y, label) in enumerate(rows):
            ax.add_patch(FancyBboxPatch((0.13, y - 0.08), 0.74, 0.16,
                                        boxstyle="round,pad=0.01,rounding_size=0.015",
                                        facecolor="white", edgecolor=colour, linewidth=1.8))
            ax.text(0.02, y, label, ha="left", va="center", fontsize=8.5, color=MUTED)
            # Target support is dashed; filled prediction changes only in contradictory panel.
            center = (0.50, y)
            ax.add_patch(Rectangle((center[0] - 0.16, center[1] - 0.045), 0.32, 0.09,
                                   fill=False, edgecolor=TEAL, linewidth=1.5, linestyle="--"))
            if contradict and idx == 1:
                pred_center = (0.66, y)
                shape = Circle(pred_center, 0.055, facecolor=ORANGE, edgecolor=INK, lw=0.9)
            elif contradict and idx == 2:
                pred_center = (0.50, y)
                shape = Circle(pred_center, 0.055, facecolor=ORANGE, edgecolor=INK, lw=0.9)
            else:
                pred_center = center
                shape = Rectangle((pred_center[0] - 0.11, pred_center[1] - 0.032), 0.22, 0.064,
                                  facecolor=TEAL, edgecolor=INK, lw=0.8)
            ax.add_patch(shape)
            if idx < 2:
                arrow(ax, (0.50, y - 0.09), (0.50, rows[idx + 1][0] + 0.09),
                      color=colour, dashed=contradict)
        verdict = "support and identity agree across levels" if not contradict else "location or identity changes between levels"
        ax.text(0.5, 0.035, verdict, ha="center", fontsize=9, weight="bold", color=colour)

    fig.suptitle("Multiple resolutions form a useful hierarchy only when their predictions agree",
                 fontsize=15, weight="bold", y=0.99)
    fig.text(0.5, 0.005, "Dashed teal rectangle: target support. Filled mark: level-specific prediction.",
             ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.84, bottom=0.10, wspace=0.10)
    save(fig, "ch1_fig1_9_cross_level_consistency")


def fig1_11_control_interfaces():
    fig, ax = canvas(13, 6.5)
    title(ax, "Consistency can enter a generator through different interfaces")
    items = [
        ("reference input", "image-to-image", BLUE, PALE_BLUE),
        ("weights", "DreamBooth / LoRA", TEAL, PALE_TEAL),
        ("structure", "edge / pose / depth", ORANGE, PALE_ORANGE),
        ("latent or noise", "inversion", "#7B5EA7", "#F0EAF7"),
        ("attention", "shared features", "#B7791F", "#FFF7DF"),
    ]
    xs = np.linspace(0.12, 0.88, len(items))
    box(ax, 0.40, 0.18, 0.20, 0.15, "generative\nmodel", fc=PALE_GRAY, ec=INK, weight="bold", fs=11)
    for x, (head, sub, color, fc) in zip(xs, items):
        box(ax, x - 0.075, 0.66, 0.15, 0.14, f"{head}\n{sub}", fc=fc, ec=color, fs=9)
        arrow(ax, (x, 0.65), (0.50, 0.34), color=color, curve=(x - 0.5) * 0.22)
    arrow(ax, (0.50, 0.17), (0.50, 0.08), color=INK)
    ax.text(0.50, 0.045, "controlled output family", ha="center", weight="bold")
    ax.text(0.5, 0.86, "The interface determines what information is available and what cost is paid.",
            ha="center", fontsize=10, color=MUTED)
    save(fig, "ch1_fig1_11_control_interfaces")


def fig1_12_support_identity_bridge():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    specs = [
        ("Correct support, wrong identity", True, False),
        ("Correct identity, broken support", False, True),
    ]
    for ax, (lab, support_ok, identity_ok) in zip(axes, specs):
        ax.set_facecolor("white"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0.08, 0.18), 0.84, 0.65, facecolor=PALE_GRAY, edgecolor=INK))
        gt = Polygon([[0.48, 0.72], [0.34, 0.36], [0.66, 0.38]], closed=True,
                     fill=False, edgecolor=TEAL, linewidth=3, linestyle="--")
        ax.add_patch(gt)
        if support_ok:
            pred = Circle((0.50, 0.51), 0.17, facecolor=ORANGE, edgecolor=INK, alpha=0.85)
        else:
            pred = Polygon([[0.65, 0.66], [0.56, 0.39], [0.77, 0.41]], closed=True,
                           facecolor=ORANGE, edgecolor=INK)
        ax.add_patch(pred)
        ax.text(0.5, 0.09, lab, ha="center", weight="bold", fontsize=10)
        ax.text(0.5, 0.88, f"support: {'pass' if support_ok else 'fail'}   |   identity: {'pass' if identity_ok else 'fail'}",
                ha="center", color=TEAL if support_ok and identity_ok else ORANGE, fontsize=9)
    fig.suptitle("Where an object is and what it is are separate criteria", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, "Dashed teal contour: target support. Filled orange shape: prediction.", ha="center", color=MUTED, fontsize=9)
    save(fig, "ch1_fig1_12_support_identity_bridge")


def fig1_15_activation_tape():
    fig, ax = canvas(12, 6.2)
    title(ax, "Ordinary BPTT stores a growing temporal activation tape")
    xs = np.linspace(0.10, 0.90, 7)
    for i, x in enumerate(xs):
        box(ax, x - 0.045, 0.58, 0.09, 0.11, f"s{i}", fc=PALE_BLUE, ec=BLUE, weight="bold")
        if i < len(xs) - 1:
            arrow(ax, (x + 0.05, 0.635), (xs[i + 1] - 0.05, 0.635), color=BLUE)
        box(ax, x - 0.045, 0.34, 0.09, 0.09, f"a{i}", fc=PALE_ORANGE, ec=ORANGE)
        arrow(ax, (x, 0.57), (x, 0.44), color=ORANGE)
    ax.plot([0.08, 0.92], [0.27, 0.27], color=ORANGE, lw=3)
    ax.text(0.50, 0.22, "stored activations grow with differentiated horizon K", ha="center", color=ORANGE, weight="bold")
    arrow(ax, (0.90, 0.16), (0.10, 0.16), color=TEAL, lw=2)
    ax.text(0.50, 0.10, "backward traversal", ha="center", color=TEAL, weight="bold")
    save(fig, "ch1_fig1_15_activation_tape")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dimensions(path: Path) -> dict:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        with Image.open(path) as image:
            return {"pixel_dimensions": [int(image.width), int(image.height)]}
    if path.suffix.lower() == ".pdf":
        page = PdfReader(path).pages[0]
        return {
            "page_dimensions_points": [
                round(float(page.mediabox.width), 3),
                round(float(page.mediabox.height), 3),
            ]
        }
    raise ValueError(f"unsupported provenance media type: {path}")


def native_record(path: Path, stem: str) -> dict:
    figure_id, asset_class, data_source, role = NATIVE_METADATA[stem]
    generator_path = ROOT / GENERATOR
    record = {
        "schema_version": SCHEMA_VERSION,
        "figure_id": figure_id,
        "file": path.relative_to(ROOT).as_posix(),
        "asset_class": asset_class,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "generator": GENERATOR,
        "generator_sha256": sha256(generator_path),
        "data_or_concept_source": data_source,
        "source_files": ["docs/dissertation/restructured_english/02_CHAPTER_1.md"],
        "modification_status": "new dissertation-native deterministic rendering",
        "rights_status": "author-generated dissertation artwork",
        "permission_status": "no external artwork permission required",
        "role": role,
        "caption_attribution": f"Chapter 1 Figure {figure_id}; caption identifies it as dissertation-native.",
        "original_figure": figure_id,
        "source_page": None,
        "extraction_method": None,
    }
    record.update(dimensions(path))
    return record


def extract_published_2024():
    source = OUT / "sources" / "Kulyk_Mokin_2024_official.pdf"
    if not source.exists():
        raise FileNotFoundError(
            f"Official 2024 article PDF is required for published-figure extraction: {source}"
        )
    document = fitz.open(source)
    selected = [
        (4, 3, "published_2024_fig1_sdxl", "1.6", "Fig. 1", 2, "illustrative SDXL outputs"),
        (6, 5, "published_2024_fig2_forward_diffusion", "1.13", "Fig. 2", 2, "forward-diffusion mechanism"),
        (11, 10, "published_2024_fig3_reverse_diffusion", "1.14", "Fig. 3", 3, "reverse-diffusion illustration"),
        (17, 16, "published_2024_fig5_visual_perception", "1.8", "Fig. 5", 4, "local-view and integrated-scene analogy"),
        (27, 26, "published_2024_fig7_controlnet", "1.17", "Fig. 7", 7, "geometry-conditioning interface"),
        (34, None, "published_2024_fig9_stylealigned", "1.19", "Fig. 9", 9, "shared-attention style examples"),
    ]
    records = []
    for xref, smask_xref, stem, figure_id, original_figure, page, role in selected:
        base_stream = document.xref_stream_raw(xref)
        mask_stream = document.xref_stream_raw(smask_xref) if smask_xref else None
        if smask_xref is None:
            extracted = document.extract_image(xref)
            suffix = ".jpg" if extracted["ext"].lower() in {"jpg", "jpeg", "jpx"} else f".{extracted['ext'].lower()}"
            target = OUT / f"{stem}{suffix}"
            target.write_bytes(extracted["image"])
            extraction_method = "direct official-PDF DCT image-object extraction; encoded bytes unchanged"
            modification_status = "official published image object reused unchanged; no crop, redraw, or recompression"
        else:
            base = fitz.Pixmap(document, xref)
            mask = fitz.Pixmap(document, smask_xref)
            combined = fitz.Pixmap(base, mask)
            target = OUT / f"{stem}.png"
            combined.save(target)
            extraction_method = "lossless official-PDF XObject plus soft-mask reconstruction to PNG"
            modification_status = "official published appearance retained unchanged; lossless container conversion only; no crop or redraw"
        record = {
            "schema_version": SCHEMA_VERSION,
            "figure_id": figure_id,
            "file": target.relative_to(ROOT).as_posix(),
            "asset_class": "published-reused-unchanged",
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "source_paper": (
                "Kulyk and Mokin, Methods for Ensuring Consistent Generation in Diffusion Models, "
                "Visnyk of Vinnytsia Politechnical Institute 175(4), 2024, "
                "DOI 10.31649/1997-9266-2024-175-4-75-85"
            ),
            "source_asset": str(source),
            "source_pdf_sha256": sha256(source),
            "original_figure": original_figure,
            "source_page": page,
            "source_xref": xref,
            "source_smask_xref": smask_xref,
            "source_base_stream_sha256": hashlib.sha256(base_stream).hexdigest(),
            "source_mask_stream_sha256": hashlib.sha256(mask_stream).hexdigest() if mask_stream else None,
            "extraction_method": extraction_method,
            "modification_status": modification_status,
            "rights_status": (
                "Author-created artwork and locally generated examples, as declared by L. Kulyk; "
                "published under CC BY 4.0 in Visnyk of Vinnytsia Politechnical Institute, "
                "where authors retain copyright; underlying methods remain attributed in the caption"
            ),
            "permission_status": "cleared: author-owned artwork reused under the article's CC BY 4.0 licence",
            "role": role,
            "caption_attribution": (
                f"Chapter 1 Figure {figure_id}; caption cites Kulyk and Mokin [133], original {original_figure}, "
                "and identifies the underlying ControlNet/StyleAligned source."
            ),
            "data_or_concept_source": "official published illustration reused only as literature-review context",
            "licence": "CC BY 4.0",
            "official_article_url": "https://visnyk.vntu.edu.ua/index.php/visnyk/article/view/3070",
            "doi": "10.31649/1997-9266-2024-175-4-75-85",
            "author_declaration": "Artwork manually created or assembled by L. Kulyk; depicted generated samples were produced locally with open models.",
        }
        record.update(dimensions(target))
        records.append(record)
    document.close()
    return records


def hash_native_outputs():
    records = []
    for stem in NATIVE_METADATA:
        for suffix in (".png", ".pdf"):
            path = OUT / f"{stem}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(native_record(path, stem))
    return records


def main():
    fig1_1_system_boundary()
    fig1_2_hidden_viewport()
    fig1_3_teacher_self_fed()
    fig1_4_resource_locations()
    fig1_6b_video_prediction_example()
    fig1_6_consistency_taxonomy()
    fig1_7_static_to_video()
    fig1_8_failure_montage()
    fig1_8_trajectory_consistency()
    fig1_9_cross_level_consistency()
    fig1_11_control_interfaces()
    fig1_12_support_identity_bridge()
    fig1_15_activation_tape()
    records = extract_published_2024() + hash_native_outputs()
    (OUT / "figure_provenance.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
