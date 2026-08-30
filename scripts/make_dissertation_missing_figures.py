"""Generate the remaining core Chapter 3/4 figures from code and measured logs.

Conceptual panels are deterministic vector drawings. Result panels parse only the logged probe
rows named in the dissertation; they do not recompute or alter any experimental estimand.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
LOGS = ROOT / "logs_rhcp"

INK = "#20252b"
MUTED = "#65717d"
GRID = "#d8dde3"
COARSE = "#6f86b6"
FINE = "#d9965b"
TARGET = "#70a97f"
DIFF = "#2563a6"
REG = "#d97706"
EBT = "#8a5aa5"
BAD = "#b6463d"
GOOD = "#3a9d78"
PDF_TIMESTAMP = datetime(2026, 8, 3, tzinfo=timezone.utc)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
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
    fig.savefig(OUT / f"{stem}.png", facecolor="white", transparent=False)
    fig.savefig(
        OUT / f"{stem}.pdf",
        facecolor="white",
        transparent=False,
        metadata={
            "Creator": "hierarchical-video-prediction",
            "CreationDate": PDF_TIMESTAMP,
            "ModDate": PDF_TIMESTAMP,
        },
    )
    plt.close(fig)


def box(ax, xy, width, height, text, color, *, fontsize=9, linewidth=1.0) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=color,
        edgecolor=INK,
        linewidth=linewidth,
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


def arrow(ax, start, end, *, color=INK, style="-|>", linewidth=1.2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            color=color,
            linewidth=linewidth,
        )
    )


def make_token_layout() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.1), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "One prediction step: 1,152 context tokens and 320 target tokens",
        loc="left",
        fontweight="bold",
    )
    x0, width, y, height = 0.05, 0.90, 0.64, 0.16
    segments = [
        (768, COARSE, "48 coarse frames\n48 × 16 = 768"),
        (384, FINE, "6 recent fine frames\n6 × 64 = 384"),
        (64, TARGET, "target 1\n16+64"),
        (64, TARGET, "target 2\n16+64"),
        (64, TARGET, "target 3\n16+64"),
        (64, TARGET, "target 4\n16+64"),
    ]
    cursor = x0
    for tokens, color, label in segments:
        segment_width = width * tokens / 1472
        ax.add_patch(
            Rectangle(
                (cursor, y),
                segment_width,
                height,
                facecolor=color,
                edgecolor="white",
                linewidth=1.2,
            )
        )
        ax.text(
            cursor + segment_width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=8 if tokens == 64 else 9,
        )
        cursor += segment_width
    context_end = x0 + width * 1152 / 1472
    ax.plot([x0, context_end], [y - 0.055, y - 0.055], color=INK, linewidth=1.2)
    ax.plot([x0, x0], [y - 0.07, y - 0.04], color=INK, linewidth=1.2)
    ax.plot(
        [context_end, context_end],
        [y - 0.07, y - 0.04],
        color=INK,
        linewidth=1.2,
    )
    ax.text(
        (x0 + context_end) / 2,
        y - 0.105,
        "equal 1,152-token context budget of the contiguous-fine control",
        ha="center",
        va="center",
        fontweight="bold",
    )
    for row_y, label, color in [
        (0.40, "token content: standardized coarse/fine latent", "#eef2f6"),
        (0.29, "embedding sum: spatial + level + role + absolute frame", "#e5ebf2"),
        (0.18, "conditioning: diffusion noise level + per-target-frame action", "#f2eadf"),
    ]:
        box(ax, (0.08, row_y), 0.84, 0.07, label, color)
    arrow(ax, (0.50, 0.47), (0.50, 0.61))
    ax.text(0.05, 0.84, "past frames 0…47", fontsize=9, color=MUTED)
    ax.text(0.75, 0.84, "future frames 48…51", fontsize=9, color=MUTED)
    ax.text(
        0.5,
        0.06,
        "The target interface is held fixed for the central 896/1,152/1,792/2,432/3,072 comparisons.",
        ha="center",
        color=MUTED,
    )
    save(fig, "fig22_token_sequence_layout")


def make_rollout_buffer() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.4), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Оновлення двох буферів після прогнозу чотирьох кадрів", loc="left", fontweight="bold")

    def lane(y, label, old_count, color):
        ax.text(0.025, y + 0.035, label, ha="left", va="center", fontsize=9.3, fontweight="bold", color=color)
        x0, width, height = 0.18, 0.72, 0.09
        slot = width / 12
        for i in range(12):
            face = "#e7ecef" if i < 2 else color if i >= 10 else "white"
            ax.add_patch(Rectangle((x0 + i * slot, y), slot, height, facecolor=face, edgecolor=INK, linewidth=0.8))
        ax.text(x0 + slot, y + height / 2, "−4", ha="center", va="center", fontsize=8.5, color=MUTED, fontweight="bold")
        ax.text(x0 + 6 * slot, y + height / 2, f"збережено {old_count - 4}", ha="center", va="center", fontsize=8.5)
        ax.text(x0 + 11 * slot, y + height / 2, "+4", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.text(0.92, y + height / 2, f"разом {old_count}", ha="left", va="center", fontsize=8.7)

    ax.text(0.18, 0.81, "стан до оновлення", fontsize=8.7, color=MUTED)
    ax.text(0.78, 0.81, "нові прогнози", fontsize=8.7, color=MUTED)
    lane(0.64, "Грубий рівень", 48, COARSE)
    lane(0.43, "Детальний рівень", 6, FINE)
    ax.text(0.50, 0.27, "Оператор tail відкидає чотири найстаріші позиції та додає чотири нові; ширина стану не змінюється.", ha="center", color=MUTED, fontsize=9.2)
    ax.add_patch(Rectangle((0.08, 0.08), 0.24, 0.07, facecolor="#dbe7f3", edgecolor=INK))
    ax.add_patch(Rectangle((0.32, 0.08), 0.60, 0.07, facecolor="#f2f2f2", edgecolor=INK))
    ax.text(0.20, 0.115, "48 кадрів розігріву", ha="center", va="center", fontsize=8.5)
    ax.text(0.62, 0.115, "152 кадри самоподачі, блоками по 4", ha="center", va="center", fontsize=8.5)
    ax.text(0.50, 0.025, "Схема показує структуру буфера, а не збереження якості його вмісту на довгому горизонті.", ha="center", color=BAD, fontsize=8.5)
    save(fig, "fig23_rollout_ring_buffer")


def make_generation_objectives() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), constrained_layout=True)
    panels = [
        (
            REG,
            "A  Direct regression",
            ["context", "predict target latent", "SmoothL1"],
            "one deterministic estimate\ncan average uncertain futures",
        ),
        (
            DIFF,
            "B  Denoising diffusion",
            ["context + noise level", "predict velocity", "DDIM sample"],
            "commits to one sample\nfrom a learned conditional field",
        ),
        (
            EBT,
            "C  Energy-based descent",
            ["context + candidate", "scalar energy", "gradient descent"],
            "quality requires both correct\nordering and reachable minima",
        ),
    ]
    for ax, (color, title, stages, note) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, loc="left", fontweight="bold", color=color)
        ys = [0.75, 0.49, 0.23]
        for stage, y in zip(stages, ys):
            box(ax, (0.16, y), 0.68, 0.12, stage, color if y == 0.49 else "#eef1f4")
        arrow(ax, (0.50, 0.75), (0.50, 0.63), color=color)
        arrow(ax, (0.50, 0.49), (0.50, 0.37), color=color)
        ax.text(0.50, 0.06, note, ha="center", va="center", color=MUTED)
    fig.suptitle(
        "The three heads differ in objective and inference, not in the shared context question",
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "fig24_generation_objectives")


def make_ebt_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Energy-based head as tested: ordering and reachability are separate capabilities",
        loc="left",
        fontweight="bold",
    )
    box(ax, (0.04, 0.58), 0.18, 0.16, "context tokens\n+ actions", "#e6edf4")
    box(ax, (0.30, 0.58), 0.18, 0.16, "candidate future\n1,280-D/frame", "#f3e7dc")
    box(ax, (0.56, 0.58), 0.18, 0.16, "reference\ntransformer", "#e8e3ef")
    box(ax, (0.82, 0.58), 0.13, 0.16, "scalar\nenergy", EBT)
    arrow(ax, (0.22, 0.66), (0.30, 0.66))
    arrow(ax, (0.48, 0.66), (0.56, 0.66))
    arrow(ax, (0.74, 0.66), (0.82, 0.66))
    box(ax, (0.35, 0.24), 0.30, 0.15, "candidate ← candidate − α∇candidate E\n(+ registered Langevin noise)", "#f1f1f1")
    arrow(ax, (0.88, 0.58), (0.64, 0.39), color=EBT)
    arrow(ax, (0.35, 0.31), (0.16, 0.55), color=EBT)
    ax.text(0.50, 0.14, "repeat K steps; best-of-4 paper sampler also tested", ha="center")
    ax.text(
        0.50,
        0.06,
        "A correctly ordered landscape can still fail if descent from the registered initialization cannot reach target structure.",
        ha="center",
        color=MUTED,
    )
    save(fig, "fig25_ebt_generation_loop")


PROBE = re.compile(
    r"\[scaled_s(?P<recipe>[12])_step(?P<step>\d+)\] landscape E\(last frame\): "
    r"tgt (?P<tgt>-?\d+\.\d+) \| mean (?P<mean>-?\d+\.\d+) \| "
    r"randn (?P<randn>-?\d+\.\d+).*?d_tgt (?P<d_tgt>\d+\.\d+).*?"
    r"fg-sharp (?P<sharp>\d+\.\d+)"
)


def parse_probe(path: Path) -> list[dict]:
    rows = []
    for match in PROBE.finditer(path.read_text(encoding="utf-8")):
        row = {key: float(value) for key, value in match.groupdict().items()}
        row["recipe"] = int(row["recipe"])
        row["step"] = int(row["step"])
        rows.append(row)
    if not rows:
        raise ValueError(f"no EBT probe rows parsed from {path}")
    return rows


def make_ebt_results() -> None:
    recipes = {
        "S1 learnable step size": parse_probe(LOGS / "ebt_scaled_s1.log"),
        "S2 fixed step size": parse_probe(LOGS / "ebt_scaled_s2.log"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharex=True, constrained_layout=True)
    for ax, (name, rows) in zip(axes, recipes.items()):
        step = np.asarray([row["step"] for row in rows])
        for key, label, color in [
            ("tgt", "target", GOOD),
            ("mean", "batch mean", REG),
            ("randn", "random", MUTED),
        ]:
            ax.plot(step, [row[key] for row in rows], label=label, color=color, linewidth=1.7)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xlabel("training update")
        ax.set_ylabel("energy (lower is preferred)")
        ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "Landscape ordering is recipe-dependent: S1 learns target < mean; S2 does not",
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "fig26_ebt_landscape_ordering")

    fig, ax = plt.subplots(figsize=(8.8, 4.4), constrained_layout=True)
    for name, rows in recipes.items():
        color = GOOD if name.startswith("S1") else BAD
        marker = "o" if name.startswith("S1") else "s"
        ax.plot(
            [row["step"] for row in rows],
            [row["sharp"] for row in rows],
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.4,
            label=name,
        )
    ax.axhline(0.08, color=INK, linestyle="--", linewidth=1.1, label="structure threshold")
    ax.axhline(0.24, color=MUTED, linestyle=":", linewidth=1.1, label="decoder reference ≈0.24")
    ax.set_xlabel("training update")
    ax.set_ylabel("cold-descent foreground sharpness")
    ax.set_ylim(0, 0.26)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_title(
        "Correct ordering is insufficient: cold descent remains in the collapse regime",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "fig27_ebt_descent_sharpness")


HARD_ROW = re.compile(
    r"HARD re-entry \(>0\.1\)\s+\d+\s+\d+\.\d+\s+\d+\.\d+\s+"
    r"(?P<mean>[+-]\d+\.\d+)\s+\[(?P<lo>[+-]\d+\.\d+),(?P<hi>[+-]\d+\.\d+)\]"
)


def parse_memory_gate(path: Path) -> dict:
    matches = list(HARD_ROW.finditer(path.read_text(encoding="utf-8")))
    if len(matches) != 1:
        raise ValueError(f"expected one HARD row in {path}, got {len(matches)}")
    return {key: float(value) for key, value in matches[0].groupdict().items()}


def make_memory_gate() -> None:
    cells = [
        ("seed 0", "6k", parse_memory_gate(LOGS / "mem_gate_full_s0.log")),
        ("seed 1", "6k", parse_memory_gate(LOGS / "mem_gate_full_s1.log")),
        ("seed 0", "20k", parse_memory_gate(LOGS / "mem_gate_full_s0_t20k.log")),
        ("seed 1", "20k", parse_memory_gate(LOGS / "mem_gate_full_s1_t20k.log")),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    y = np.arange(len(cells))
    means = np.asarray([row["mean"] for _, _, row in cells])
    lo = np.asarray([row["lo"] for _, _, row in cells])
    hi = np.asarray([row["hi"] for _, _, row in cells])
    colors = [MUTED, MUTED, GOOD, GOOD]
    for i in range(len(cells)):
        ax.errorbar(
            means[i],
            y[i],
            xerr=[[means[i] - lo[i]], [hi[i] - means[i]]],
            color=colors[i],
            marker="o",
            capsize=3,
            linewidth=1.4,
        )
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{seed}, {budget}" for seed, budget, _ in cells])
    ax.invert_yaxis()
    ax.set_xlabel("HARD-bin effect: short error − long-reach error")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_title(
        "Single-step retrieval emerges only after the larger optimization budget",
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0.99,
        0.05,
        "6k: both intervals include zero\n20k: both intervals exclude zero",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "fig28_memory_gate_optimization")


def make_mechanism_samples() -> None:
    regression_path = LOGS / "step1_mech_clean_reg.png"
    diffusion_path = LOGS / "step1_mech_clean_diff.png"
    if not regression_path.is_file() or not diffusion_path.is_file():
        frozen_montage = OUT / "fig29_mechanism_single_step_samples.png"
        if not frozen_montage.is_file():
            raise FileNotFoundError(
                "The source panels and the frozen full montage are both unavailable."
            )
        from make_dissertation_compact_galleries import mechanism_gallery

        mechanism_gallery()
        print("preserved frozen fig29 montage; rebuilt its fixed compact view")
        return
    regression = plt.imread(regression_path)
    diffusion = plt.imread(diffusion_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 7.8), constrained_layout=True)
    for ax, image, title, color in [
        (axes[0], regression, "А  Пряма регресія", REG),
        (axes[1], diffusion, "Б  Дифузійне знешумлення", DIFF),
    ]:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, loc="left", fontweight="bold", color=color)
        ax.axis("off")
    fig.suptitle(
        "Однакові однокрокові входи: останній контекст | прогноз | еталон",
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "fig29_mechanism_single_step_samples")

    # The main text uses a fixed compact crop; the complete montage remains an appendix asset.
    rows = 3
    compact_regression = regression[: regression.shape[0] * rows // 8]
    compact_diffusion = diffusion[: diffusion.shape[0] * rows // 8]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.6), constrained_layout=True)
    for ax, image, heading, color in [
        (axes[0], compact_regression, "А  Пряма регресія", REG),
        (axes[1], compact_diffusion, "Б  Дифузійне знешумлення", DIFF),
    ]:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(heading, loc="left", fontweight="bold", color=color)
        ax.axis("off")
    fig.suptitle("Три перші зафіксовані записи: контекст | прогноз | еталон", fontsize=11.5, fontweight="bold")
    save(fig, "fig29_mechanism_single_step_samples_compact")


def main() -> None:
    style()
    make_token_layout()
    make_rollout_buffer()
    make_generation_objectives()
    make_ebt_loop()
    ebt_logs = [LOGS / "ebt_scaled_s1.log", LOGS / "ebt_scaled_s2.log"]
    if all(path.is_file() for path in ebt_logs):
        make_ebt_results()
    else:
        print("preserved frozen EBT result figures; legacy probe logs are unavailable")
    memory_logs = [
        LOGS / "mem_gate_full_s0.log",
        LOGS / "mem_gate_full_s1.log",
        LOGS / "mem_gate_full_s0_t20k.log",
        LOGS / "mem_gate_full_s1_t20k.log",
    ]
    if all(path.is_file() for path in memory_logs):
        make_memory_gate()
    else:
        print("preserved frozen memory-gate figure; legacy training logs are unavailable")
    make_mechanism_samples()
    print("DISSERTATION_MISSING_FIGURES_DONE")


if __name__ == "__main__":
    main()
