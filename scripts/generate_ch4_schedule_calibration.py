"""Generate the dissertation A4-COR schedule-calibration figure.

The inputs are sealed copies of the two 30k seed-0 validation logs and their
machine-readable paired analysis.  The figure deliberately reports schedule
selection only; it is not evidence that either context allocation is better.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "dissertation" / "a4_cor"
OUTPUT = ROOT / "figures" / "restructured" / "ch4_fig4_3b_schedule_calibration.svg"
METRIC = "horizon4_foreground_mse"
EXPECTED_STEPS = list(range(2_000, 30_001, 2_000))


def load_series(path: Path) -> tuple[list[int], list[float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    steps = [int(row["step"]) for row in rows if row.get("event") == "validation"]
    values = [float(row[METRIC]) for row in rows if row.get("event") == "validation"]
    if steps != EXPECTED_STEPS:
        raise ValueError(f"Unexpected validation schedule in {path}: {steps}")
    return steps, values


def comma(value: float, _position: int) -> str:
    return f"{value:.3f}".replace(".", ",")


def main() -> None:
    flat_steps, flat = load_series(DATA / "flat_recent_seed0_validation_metrics.jsonl")
    hier_steps, hierarchy = load_series(DATA / "balanced_recent_seed0_validation_metrics.jsonl")
    if flat_steps != hier_steps:
        raise ValueError("Calibration runs use different validation milestones")

    analysis = json.loads(
        (DATA / "article4_schedule_calibration_2026-09-01.json").read_text(encoding="utf-8")
    )
    if analysis["recommended_updates"] != 30_000 or analysis["primary_metric"] != METRIC:
        raise ValueError("The sealed analysis does not authorize the 30k primary-metric decision")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "svg.hashsalt": "hvp-dissertation-a4-cor-2026-09-04",
        }
    )
    colors = {"flat": "#0072B2", "hier": "#D55E00"}
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.1), constrained_layout=True)

    for axis in axes:
        axis.plot(
            [step / 1_000 for step in flat_steps],
            flat,
            color=colors["flat"],
            marker="o",
            markersize=4.4,
            linewidth=1.8,
            label="Плоский недавній контекст",
        )
        axis.plot(
            [step / 1_000 for step in hier_steps],
            hierarchy,
            color=colors["hier"],
            marker="s",
            markersize=4.2,
            linewidth=1.8,
            label="Збалансована ієрархія",
        )
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        axis.set_xlabel("Кількість кроків навчання, тис.")
        axis.yaxis.set_major_formatter(FuncFormatter(comma))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_title("а) Повна траєкторія валідації")
    axes[0].set_ylabel("MSE переднього плану на горизонті 4 ↓")
    axes[0].set_xlim(1.2, 30.8)
    axes[0].set_xticks([2, 6, 10, 14, 18, 22, 26, 30])
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.95)

    late_indices = [index for index, step in enumerate(flat_steps) if step >= 12_000]
    late_values = [flat[index] for index in late_indices] + [hierarchy[index] for index in late_indices]
    padding = (max(late_values) - min(late_values)) * 0.18
    axes[1].set_title("б) Ділянка прийняття рішення")
    axes[1].set_xlim(11.4, 30.6)
    axes[1].set_ylim(min(late_values) - padding, max(late_values) + padding)
    axes[1].set_xticks([12, 16, 20, 24, 28, 30])
    axes[1].axvline(16, color="#6B7280", linestyle="--", linewidth=1.0)
    axes[1].axvline(30, color="#009E73", linestyle="--", linewidth=1.4)
    axes[1].annotate(
        "зміна порядку\nміж 12 і 16 тис.",
        xy=(16, max(flat[7], hierarchy[7])),
        xytext=(16.5, max(late_values) - 0.001),
        color="#374151",
        fontsize=9,
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 0.9},
    )
    axes[1].annotate(
        "30 тис.: обидві криві\nу межах 2 % і порядок сталий",
        xy=(30, max(flat[-1], hierarchy[-1])),
        xytext=(29.2, max(late_values) + 0.0005),
        color="#00664B",
        fontsize=9,
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#009E73", "lw": 1.0},
    )
    axes[1].annotate(
        "+6,80 % до пізнішого мінімуму",
        xy=(24, hierarchy[11]),
        xytext=(18.0, 0.0305),
        color=colors["hier"],
        fontsize=8.5,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": colors["hier"], "lw": 0.8},
    )

    figure.suptitle(
        "Калібрування тривалості навчання A4-COR (ініціалізація 0, бюджет 1 792)",
        fontsize=12,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="svg", metadata={"Date": None})
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
