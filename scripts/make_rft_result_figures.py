"""Generate manuscript result figures from immutable JSON artifacts.

The script never recomputes an estimand.  It renders only the estimates and confidence
intervals stored by the frozen analyses and labels clip-conditional and training-seed
uncertainty separately.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

INK = "#20252b"
MUTED = "#65717d"
GRID = "#d8dde3"
DIFF = "#2563a6"
REG = "#d97706"
INTERACTION = "#8a5aa5"
EARLY = "#3a9d78"
LATE = "#b6463d"
ORIGINAL = "#4f6f8f"
ALTERNATE = "#b56a3a"
RAW = "#2f7d68"
RAW_LIGHT = "#78a99a"
HIER = "#2563a6"
FLAT = "#b6463d"
UNRESOLVED = "#65717d"
PDF_TIMESTAMP = datetime(2026, 7, 17, tzinfo=timezone.utc)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(
        OUT / f"{stem}.pdf",
        metadata={
            "Creator": "hierarchical-video-prediction",
            "CreationDate": PDF_TIMESTAMP,
            "ModDate": PDF_TIMESTAMP,
        },
    )
    plt.close(fig)


def _errorbar(
    ax: plt.Axes,
    y: np.ndarray,
    rows: list[dict],
    *,
    color: str,
    label: str | None = None,
    marker: str = "o",
) -> None:
    mean = np.asarray([row["mean"] for row in rows], dtype=float)
    lo = np.asarray([row["lo"] for row in rows], dtype=float)
    hi = np.asarray([row["hi"] for row in rows], dtype=float)
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - lo, hi - mean]),
        color=color,
        marker=marker,
        markersize=5.5,
        capsize=2.5,
        linewidth=1.35,
        linestyle="none",
        label=label,
    )


def make_confirmed_frontier() -> None:
    """Render the untouched, family-wise frontier and its exact token arithmetic."""
    result = _read("artifacts/confirmation_e2/results.json")
    rows = result["frontier"]["contrasts"]
    plotted = [
        (896, "s0", rows["b896-s0"]),
        (896, "s1", rows["b896-s1"]),
        (1152, "s0", rows["b1152-s0"]),
        (1152, "s1", rows["b1152-s1"]),
        (1152, "s2", rows["b1152-s2"]),
        (1792, "s0", rows["b1792-s0"]),
        (2432, "30k", rows["b2432-30k-s0"]),
        (2432, "60k", rows["b2432-60k-s0"]),
        (3072, "s0", rows["b3072-s0"]),
        (3072, "s1", rows["b3072-s1"]),
    ]
    markers = {"s0": "o", "s1": "s", "s2": "^", "30k": "D", "60k": "P"}
    offsets = {"s0": 0.975, "s1": 1.025, "s2": 1.0, "30k": 0.975, "60k": 1.025}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.2),
        gridspec_kw={"width_ratios": [1.55, 1.15]},
        constrained_layout=True,
    )
    ax = axes[0]
    seen_labels: set[str] = set()
    for budget, label, row in plotted:
        color = HIER if row["lo"] > 0 else FLAT if row["hi"] < 0 else UNRESOLVED
        legend_label = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.errorbar(
            budget * offsets[label],
            row["mean"],
            yerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            color=color,
            marker=markers[label],
            markersize=5.5,
            capsize=2.5,
            linewidth=1.25,
            linestyle="none",
            label=legend_label,
        )
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([896, 1152, 1792, 2432, 3072])
    ax.set_xticklabels(["896", "1 152", "1 792", "2 432", "3 072"])
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_xlabel("Бюджет токенів контексту")
    ax.set_ylabel("Похибка плоского − похибка ієрархічного")
    ax.set_title("А  Одночасна межа на відкладених даних", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.97,
        "синій: підтверджено ієрархію\n"
        "сірий: невирішено\n"
        "червоний: підтверджено плоский пакет",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    ax.legend(frameon=False, ncol=5, loc="lower left", fontsize=8, title="навчання / режим")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    packages = [
        ("48 детальних", 3072, FLAT),
        ("48 грубих + 6 детальних", 1152, HIER),
        ("48 грубих + 2 детальні", 896, HIER),
    ]
    y = np.arange(len(packages))
    values = [value for _, value, _ in packages]
    bars = ax.barh(
        y,
        values,
        color=[color for _, _, color in packages],
        alpha=0.88,
        height=0.58,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in packages])
    ax.invert_yaxis()
    ax.set_xlim(0, 3400)
    ax.set_xlabel("Токени контексту за номінальної досяжності 48 кадрів")
    ax.set_title("Б  Точна арифметика контексту", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    bar_notes = ["еталон", "у 2,67 раза\nменше", "у 3,43 раза\nменше"]
    for bar, value, note in zip(bars, values, bar_notes):
        yc = bar.get_y() + bar.get_height() / 2
        ax.text(value + 45, yc, f"{value:,}".replace(",", " "), va="center")
        # Put the efficiency statement inside its own bar instead of over the
        # neighbouring dense-context bar.  White text remains readable in
        # greyscale and the exact value stays outside the bar.
        ax.text(
            max(value * 0.53, 250),
            yc,
            note,
            ha="center",
            va="center",
            color="white",
            fontsize=7.7,
            fontweight="bold",
        )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        "Ієрархія збільшує досяжність, а плоский контекст зберігає точність відтворення",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Інтервали на панелі А утворюють одну студентовану max-статистичну родину з кластеризацією за кліпами; "
        "інтерполяцію та єдину точку переходу не виводили.",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, "article_confirmed_frontier")


def make_head_package() -> None:
    result = _read("artifacts/e4b_head_package/results.json")
    seeds = result["seeds"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 5.0), sharey=True, constrained_layout=True)
    panels = [
        ("diffusion", DIFF, "A  Diffusion", "temporal package advantage"),
        ("regression", REG, "B  Regression", "temporal package advantage"),
        ("interaction", INTERACTION, "C  Head × package", "regression − diffusion"),
    ]
    y = np.arange(len(seeds))
    for ax, (key, color, title, xlabel) in zip(axes, panels):
        rows = [row[key] for row in seeds]
        _errorbar(ax, y, rows, color=color)
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(f"Effect ({xlabel})")
        ax.set_yticks(y)
        ax.set_yticklabels([f"seed {row['seed']}" for row in seeds])
        ax.invert_yaxis()
        summary = result["training_seed_summaries"][key]
        ax.axvspan(summary["lo"], summary["hi"], color=color, alpha=0.10)
        ax.axvline(summary["mean"], color=color, linewidth=1.3, linestyle="--")
        ax.text(
            0.50,
            -0.18,
            f"across-seed mean {summary['mean']:+.4f}\n"
            f"t95 [{summary['lo']:+.4f}, {summary['hi']:+.4f}]",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            clip_on=False,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "Clip intervals are precise, but training-seed heterogeneity changes the cross-head conclusion",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, "fig17_head_package_seed_heterogeneity")


def make_trust_replication() -> None:
    result = _read("artifacts/e5a_trust_replication/results.json")
    seeds = result["seeds"]
    fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    y = np.arange(len(seeds), dtype=float)
    early = [row["primary"]["early_benefit"] for row in seeds]
    late = [row["primary"]["late_inversion"] for row in seeds]
    _errorbar(ax, y - 0.12, early, color=EARLY, label="early h1–24", marker="o")
    _errorbar(ax, y + 0.12, late, color=LATE, label="late h61–256", marker="s")
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"training seed {row['seed']}" for row in seeds])
    ax.invert_yaxis()
    ax.set_xlabel("Comparator error − hierarchy error (positive favors hierarchy)")
    ax.set_title(
        "Self-fed reliability boundary: the early benefit replicates; the late inversion does not",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.text(
        1.0,
        1.08,
        "registered verdict: NOT_REPLICATED",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=LATE,
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _save(fig, "fig18_trust_negative_replication")


def make_sampler_robustness() -> None:
    result = _read("artifacts/e7c_sampler_nfe/results.json")
    contrasts = [
        ("b896-reach", "896 reach"),
        ("b1152-interior", "1,152 interior"),
        ("b1152-package-shape", "1,088 vs 1,152\npackage shape"),
        ("b2432-inversion", "2,432 inversion"),
    ]
    variants = [
        ("nfe8-ss0", "NFE 8, s0"),
        ("nfe16-ss0", "NFE 16, s0"),
        ("nfe32-ss0", "NFE 32, s0"),
        ("nfe16-ss1", "NFE 16, s1"),
        ("nfe16-ss2", "NFE 16, s2"),
    ]
    lookup = {(row["contrast"], row["variant"]): row for row in result["primary"]}
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.7), sharey=True, constrained_layout=True)
    y = np.arange(len(variants))
    for index, (ax, (contrast, label)) in enumerate(zip(axes, contrasts)):
        rows = [lookup[(contrast, variant)] for variant, _ in variants]
        color = DIFF if contrast != "b2432-inversion" else REG
        _errorbar(ax, y, rows, color=color)
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_title(f"{chr(65 + index)}  {label}", loc="left", fontweight="bold")
        ax.set_xlabel("Registered effect")
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label in variants])
        ax.invert_yaxis()
        for row, yi in zip(rows, y):
            if row["lo"] <= 0:
                ax.scatter([row["mean"]], [yi], s=90, facecolors="none", edgecolors=LATE, linewidths=1.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "All directions survive inference variation; two upper-inversion family intervals include zero",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, "fig19_sampler_nfe_robustness")


def make_codec_robustness() -> None:
    path = ROOT / "artifacts/e7a_codec_robustness/results.json"
    if not path.is_file():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    contrasts = [
        ("b896-reach", "896 reach"),
        ("b1152-interior", "1,152 interior"),
        ("b1152-package-shape", "1,088 vs 1,152\npackage shape"),
        ("b2432-inversion", "2,432 inversion"),
    ]
    lookup = {
        (row["codec"], row["contrast"]): row
        for row in result["primary"]
    }
    fig, axes = plt.subplots(
        1, 4, figsize=(12.4, 4.0), sharey=True, constrained_layout=True
    )
    y = np.asarray([0.0, 1.0])
    for index, (ax, (contrast, label)) in enumerate(zip(axes, contrasts)):
        rows = [lookup[("original", contrast)], lookup[("alternate", contrast)]]
        _errorbar(ax, y[:1], [rows[0]], color=ORIGINAL, label="original codec")
        _errorbar(
            ax,
            y[1:],
            [rows[1]],
            color=ALTERNATE,
            label="alternate codec seed",
            marker="s",
        )
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_title(f"{chr(65 + index)}  {label}", loc="left", fontweight="bold")
        ax.set_xlabel("Registered effect")
        ax.set_yticks(y)
        ax.set_yticklabels(["original", "alternate"])
        ax.invert_yaxis()
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "Central directions under an independently trained latent codec",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, "fig20_alternate_codec_robustness")


def make_pose_mosaic_reference() -> None:
    path = ROOT / "artifacts/e8a_pose_mosaic/results.json"
    if not path.is_file():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    storage = sorted(result["mosaic_storage"], key=lambda row: row["span"])
    effects = result["primary"]
    effect_order = [
        ("raw-reach-14-to-48", "raw K14 → K48"),
        ("raw-reach-38-to-48", "raw K38 → K48"),
        ("hier-min-to-raw48", "hier-min → raw K48"),
        ("hier-long-to-raw48", "hier-long → raw K48"),
        ("strided-flat-to-raw48", "strided-flat → raw K48"),
        ("flat-c2-to-raw48", "flat-c2 → raw K48"),
    ]
    lookup = {row["effect"]: row for row in effects}

    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [0.92, 1.35]},
        constrained_layout=True,
    )
    ax = axes[0]
    spans = np.asarray([row["span"] for row in storage], dtype=float)
    columns = np.asarray(
        [row["median_stored_world_columns"] for row in storage], dtype=float
    )
    crop_coverage = np.asarray(
        [row["mean_current_crop_observed_fraction"] for row in storage], dtype=float
    )
    ax.plot(spans, columns, color=RAW, marker="o", linewidth=1.7, label="stored world columns")
    ax.set_xlabel("Stored warm-up frames K")
    ax.set_ylabel("Median stored world columns", color=RAW)
    ax.tick_params(axis="y", labelcolor=RAW)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_title("A  Explicit coverage", loc="left", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(
        spans, crop_coverage, color=RAW_LIGHT, marker="s", linestyle="--",
        linewidth=1.5, label="future crop observed",
    )
    ax2.set_ylabel("Mean future-crop observed fraction", color=RAW_LIGHT)
    ax2.tick_params(axis="y", labelcolor=RAW_LIGHT)
    ax2.set_ylim(0, 1.02)
    ax.set_xticks(spans)
    for side in ("top",):
        ax.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)

    ax = axes[1]
    ordered = [lookup[effect] for effect, _ in effect_order]
    y = np.arange(len(ordered))
    raw_rows = ordered[:2]
    neural_rows = ordered[2:]
    _errorbar(ax, y[:2], raw_rows, color=RAW_LIGHT, label="raw span diagnostics", marker="s")
    _errorbar(ax, y[2:], neural_rows, color=RAW, label="neural → raw K48", marker="o")
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in effect_order])
    ax.invert_yaxis()
    ax.set_xlabel("Baseline error − raw-K48 error")
    ax.set_title("B  Registered simultaneous family", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    verdict = result["decision"]["verdict"].replace("_", " ")
    saturation = result["decision"].get("coverage_saturated_by_k38", False)
    ax.text(
        0.99,
        0.96,
        f"{verdict}\nK38 saturation: {'yes' if saturation else 'no'}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=RAW,
        fontsize=8.5,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        "Privileged pose-indexed raw memory: explicit coverage and neural-reference effects",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Raw pixels and exact poses are supplied; this reference is not token matched.",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, "fig21_pose_indexed_raw_memory")


def make_phasec_v2_boundary() -> None:
    result = _read("artifacts/phasec_v2/results.json")
    lookup = {
        (row["endpoint"], int(row["seed"])): row
        for row in result["primary"]
    }
    unitless = [
        ("foreground_mse", "Foreground MSE"),
        ("union_mse", "Union MSE"),
        ("identity_accuracy", "Identity accuracy"),
        ("object_coverage", "Object coverage"),
    ]
    seeds = [0, 1, 2]
    colors = [HIER, "#5485ba", "#86a9cf"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.4),
        gridspec_kw={"width_ratios": [1.55, 0.8]},
        constrained_layout=True,
    )

    ax = axes[0]
    base = np.arange(len(unitless), dtype=float)
    offsets = np.asarray([-0.20, 0.0, 0.20])
    for seed, offset, color in zip(seeds, offsets, colors):
        rows = [lookup[(endpoint, seed)] for endpoint, _ in unitless]
        _errorbar(
            ax,
            base + offset,
            rows,
            color=color,
            label=f"seed {seed}",
            marker="o",
        )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_yticks(base)
    ax.set_yticklabels([label for _, label in unitless])
    ax.invert_yaxis()
    ax.set_xlabel("Registered effect (positive favors hierarchy)")
    ax.set_title("A  Support and identity effects", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    trajectory = [lookup[("trajectory_error", seed)] for seed in seeds]
    y = np.arange(len(seeds))
    for index, (row, color) in enumerate(zip(trajectory, colors)):
        _errorbar(ax, np.asarray([index]), [row], color=color, marker="o")
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"seed {seed}" for seed in seeds])
    ax.invert_yaxis()
    ax.set_xlabel("Flat trajectory error − hierarchy error (px)")
    ax.set_title("B  Persistent-object trajectory", loc="left", fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        "Long Moving-MNIST: hierarchy preserves support, flat context preserves identity",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Positive favors hierarchy; intervals belong to one simultaneous 15-effect family. "
        "Registered verdict: MIXED_BOUNDARY.",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, "fig22_phasec_v2_boundary")


def main() -> None:
    _style()
    make_confirmed_frontier()
    make_head_package()
    make_trust_replication()
    make_sampler_robustness()
    make_codec_robustness()
    make_pose_mosaic_reference()
    make_phasec_v2_boundary()
    print("RFT_RESULT_FIGURES_DONE")


if __name__ == "__main__":
    main()
