"""Build compact main-text galleries from frozen full-resolution dissertation figures.

The crops are fixed by source-image geometry, not selected by model performance.  Full
source montages remain unchanged and are cited in Appendix E.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"


def mechanism_gallery() -> None:
    source = Image.open(FIGURES / "fig29_mechanism_single_step_samples.png").convert("RGB")
    width, height = source.size
    y0, y1 = round(0.067 * height), round(0.412 * height)
    panels = [
        source.crop((round(0.010 * width), y0, round(0.430 * width), y1)),
        source.crop((round(0.568 * width), y0, round(0.990 * width), y1)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), constrained_layout=True)
    for ax, panel, heading in zip(axes, panels, ["А  Пряма регресія", "Б  Дифузійне знешумлення"]):
        ax.imshow(panel)
        ax.set_title(heading, loc="left", fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Три перші зафіксовані записи: контекст | прогноз | еталон", fontsize=11.5, fontweight="bold")
    fig.savefig(FIGURES / "fig29_mechanism_single_step_samples_compact.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rollout_gallery() -> None:
    source = Image.open(FIGURES / "fig7_samples_claim1.png").convert("RGB")
    # Exact grid coordinates of the first predeclared clip (clip 0).
    xs = [(274, 441), (456, 623), (638, 805), (820, 987), (1002, 1169)]
    ys = [(115, 282), (301, 468), (487, 654)]
    horizons = [1, 24, 60, 100, 148]
    row_labels = ["Еталон", "Ієрархічний пакет", "Плоский пакет"]
    fig, axes = plt.subplots(3, 5, figsize=(8.8, 5.25), constrained_layout=True)
    for row, (y0, y1) in enumerate(ys):
        for column, (x0, x1) in enumerate(xs):
            ax = axes[row, column]
            ax.imshow(source.crop((x0, y0, x1, y1)))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"горизонт {horizons[column]}", fontsize=9)
            if column == 0:
                ax.set_ylabel(row_labels[row], fontsize=8.5)
    fig.suptitle("Фіксований кліп 0, ініціалізація навчання 1", fontsize=11.5, fontweight="bold")
    fig.savefig(FIGURES / "fig7_samples_claim1_compact.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    mechanism_gallery()
    rollout_gallery()
    print("DISSERTATION_COMPACT_GALLERIES_DONE")


if __name__ == "__main__":
    main()
