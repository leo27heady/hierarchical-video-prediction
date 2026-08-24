"""Build the missing Chapter 2-4 visuals for the restructured dissertation.

The script has two deliberately separate paths:

* published artwork is copied from the author's source tree or extracted from a
  supplied PDF image XObject without changing the encoded bytes;
* dissertation-native schematics and evidence-derived charts are generated on
  a white background as vector PDF and high-resolution PNG pairs.

Every output is recorded in ``ch2_ch4_figure_provenance.json`` with a SHA-256
digest, source, figure/page identity, and modification status.  The script does
not overwrite any asset outside ``docs/figures/restructured``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "restructured"
OUT.mkdir(parents=True, exist_ok=True)

TEMPVERSE_PDF = Path(r"C:/Users/leoni/Downloads/Temporal_Modeling_with_Reversible_Transformers__NCAv2_.pdf")
VBLT_PDF = Path(r"C:/Users/leoni/Downloads/Article_LKulyk_OMokin_124_2025_mt.pdf")
TEMPVERSE_SOURCE = Path(r"C:/Users/leoni/Documents/projects/TempVerseFormer/resources/figures")
E2_JSON = ROOT / "artifacts" / "confirmation_e2" / "results.json"
LOGS = ROOT / "logs_rhcp"

BLUE = "#1F4E79"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
PURPLE = "#6F52A2"
GOLD = "#D9A441"
GREEN = "#3D8B67"
RED = "#B54747"
INK = "#1F2937"
MUTED = "#5F6B7A"
GRID = "#D8DEE6"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#E8F5F2"
PALE_ORANGE = "#FCEFEA"
PALE_PURPLE = "#F1ECF8"
PALE_GREEN = "#EAF5EE"
PALE_GOLD = "#FFF7DF"
PALE_GRAY = "#F4F6F8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

PROVENANCE: list[dict] = []
SCHEMA_VERSION = "1.0"
GENERATOR = "scripts/make_restructured_ch2_ch4_figures.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(
    path: Path,
    *,
    figure_id: str,
    source: str,
    original_figure: str,
    source_page: int | str | None,
    modification_status: str,
    rights_status: str,
    data_source: str | None = None,
) -> None:
    published = "byte-identical" in modification_status or "embedded PDF image-object" in modification_status
    if published:
        asset_class = "published-reused-unchanged"
        permission_status = (
            "OPEN: confirm the final publisher/journal reuse licence before institutional deposit"
            if "confirm" in rights_status.lower()
            else "cleared under stated source terms"
        )
        generator = None
        generator_hash = None
        if "10.1007/s00521-025-11707-3" in source:
            source_paper = (
                "Kulyk, Temporal Modeling with Reversible Transformers, Neural Computing and "
                "Applications 38, article 30, 2026, DOI 10.1007/s00521-025-11707-3"
            )
        elif "10.31649/1997-9266-2025-182-5-129-139" in source:
            source_paper = (
                "Kulyk and Mokin, Scaling Video Prediction with Spatio-Temporal Patches, "
                "Visnyk of Vinnytsia Politechnical Institute 182(5), 2025, "
                "DOI 10.31649/1997-9266-2025-182-5-129-139"
            )
        else:
            source_paper = source
        extraction_method = modification_status
    else:
        source_lower = (data_source or "").lower()
        if any(token in source_lower for token in ("artifacts/", "logs_", "published table", "exact token", "frozen")):
            asset_class = "author-generated-artifact-derived"
        elif any(token in source_lower for token in ("implemented", "implementation", "codec", "windowdit", "evaluator")):
            asset_class = "author-generated-implementation-informed"
        else:
            asset_class = "author-generated-conceptual"
        permission_status = "no external artwork permission required"
        generator = GENERATOR
        generator_hash = sha256(ROOT / GENERATOR)
        source_paper = None
        extraction_method = None

    match = re.fullmatch(r"(\d+\.\d+)([a-z])?", figure_id)
    caption_id = match.group(1) if match else figure_id
    panel = f", panel {match.group(2)}" if match and match.group(2) else ""
    entry = {
        "schema_version": SCHEMA_VERSION,
        "figure_id": figure_id,
        "file": path.relative_to(ROOT).as_posix(),
        "asset_class": asset_class,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "source": source,
        "source_paper": source_paper,
        "original_figure": original_figure,
        "source_page": source_page,
        "extraction_method": extraction_method,
        "modification_status": modification_status,
        "rights_status": rights_status,
        "permission_status": permission_status,
        "generator": generator,
        "generator_sha256": generator_hash,
        "data_source": data_source,
        "data_or_concept_source": data_source,
        "role": data_source or source,
        "caption_attribution": (
            f"Chapter {caption_id.split('.', 1)[0]} Figure {caption_id}{panel}; "
            + ("caption identifies the source paper and unchanged reuse status."
               if published else "caption identifies the figure as author-generated, derived, or conceptual.")
        ),
    }
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        with Image.open(path) as image:
            entry["pixel_dimensions"] = [int(image.width), int(image.height)]
    elif suffix == ".pdf":
        document = fitz.open(path)
        entry["page_dimensions_points"] = [
            round(float(document[0].rect.width), 3),
            round(float(document[0].rect.height), 3),
        ]
        document.close()
    else:
        raise ValueError(f"unsupported provenance media type: {path}")
    PROVENANCE.append(entry)


def canvas(width: float = 12.0, height: float = 6.4):
    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.text(0.5, 0.975, text, ha="center", va="top", fontsize=15, weight="bold")
    if subtitle:
        ax.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=9.5, color=MUTED)


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = PALE_BLUE,
    ec: str = BLUE,
    lw: float = 1.7,
    fs: float = 9.5,
    weight: str = "normal",
    dashed: bool = False,
    rounding: float = 0.015,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color: str = INK,
    lw: float = 1.5,
    dashed: bool = False,
    curve: float = 0.0,
    style: str = "-|>",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        color=color,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)
    return patch


def save_native(fig, stem: str, figure_id: str, data_source: str | None = None) -> None:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for path in (png, pdf):
        record(
            path,
            figure_id=figure_id,
            source="dissertation-native deterministic generator",
            original_figure=figure_id,
            source_page=None,
            modification_status="new dissertation-native figure",
            rights_status="author-generated",
            data_source=data_source,
        )


def copy_published(
    source: Path,
    stem: str,
    *,
    figure_id: str,
    original_figure: str,
    source_page: int | str | None,
    source_label: str,
    rights_status: str,
) -> Path:
    target = OUT / f"{stem}{source.suffix.lower()}"
    shutil.copyfile(source, target)
    if source.read_bytes() != target.read_bytes():
        raise RuntimeError(f"published copy changed bytes: {source}")
    record(
        target,
        figure_id=figure_id,
        source=source_label,
        original_figure=original_figure,
        source_page=source_page,
        modification_status="byte-identical copy; no crop, redraw, recolour, or relabelling",
        rights_status=rights_status,
        data_source=str(source),
    )
    return target


def extract_pdf_image(
    pdf_path: Path,
    page_index: int,
    xref: int,
    stem: str,
    *,
    figure_id: str,
    original_figure: str,
    source_page: int,
    source_label: str,
    rights_status: str,
) -> Path:
    document = fitz.open(pdf_path)
    page = document[page_index]
    if xref not in {image[0] for image in page.get_images(full=True)}:
        raise RuntimeError(f"xref {xref} is not present on page {page_index + 1} of {pdf_path}")
    payload = document.extract_image(xref)
    target = OUT / f"{stem}.{payload['ext']}"
    target.write_bytes(payload["image"])
    document.close()
    record(
        target,
        figure_id=figure_id,
        source=source_label,
        original_figure=original_figure,
        source_page=source_page,
        modification_status="exact embedded PDF image-object extraction; encoded bytes unchanged",
        rights_status=rights_status,
        data_source=f"{pdf_path}#page={source_page};xref={xref}",
    )
    return target


def write_published_assets() -> None:
    tv_rights = (
        "author-created source artwork; author reuse declaration recorded; the Research Square "
        "version (DOI 10.21203/rs.3.rs-6293520/v1) is CC BY 4.0"
    )
    vb_rights = (
        "author-created source artwork reused from the Visnyk article under CC BY 4.0; "
        "authors retain copyright"
    )
    # Original author source files for vector-like architecture artwork.
    originals = [
        ("rev-block-final.png", "ch2_fig2_5_tempverse_reversible_block_published", "2.5", "Fig. 2", 9),
        ("tempverse-arch-final.png", "ch2_fig2_6_tempverse_architecture_published", "2.6", "Fig. 3", 12),
        ("chain-example-final.png", "ch2_fig2_7_tempverse_chaining_published", "2.7", "Fig. 4", 17),
    ]
    for filename, stem, fid, original, page in originals:
        copy_published(
            TEMPVERSE_SOURCE / filename,
            stem,
            figure_id=fid,
            original_figure=original,
            source_page=page,
            source_label="TempVerseFormer author source artwork; DOI 10.1007/s00521-025-11707-3",
            rights_status=tv_rights,
        )

    tv_images = [
        (19, 58, "ch2_fig2_8_tempverse_time_agnostic_published", "2.8", "Fig. 5", 20),
        (23, 72, "ch2_fig2_9_tempverse_constant_five_step_published", "2.9", "Fig. 6", 24),
        (24, 75, "ch2_fig2_10_tempverse_constant_fid_published", "2.10", "Fig. 7", 25),
        (24, 76, "ch2_fig2_11_tempverse_eleven_step_published", "2.11", "Fig. 8", 25),
        (26, 82, "ch2_fig2_12_tempverse_accelerated_published", "2.12", "Fig. 9", 27),
        (27, 85, "ch2_fig2_13_tempverse_decelerated_published", "2.13", "Fig. 10", 28),
        (28, 88, "ch2_fig2_14_tempverse_oscillating_published", "2.14", "Fig. 11", 29),
        (29, 91, "ch2_fig2_15_tempverse_interrupted_published", "2.15", "Fig. 12", 30),
        (30, 94, "ch2_fig2_16_tempverse_combined_fid_published", "2.16", "Fig. 13", 31),
        (31, 98, "ch2_fig2_18_tempverse_memory_published", "2.18", "Fig. 14", 32),
        (32, 102, "ch2_fig2_19_tempverse_time_published", "2.19", "Fig. 15", 33),
        (33, 106, "ch2_fig2_17a_tempverse_mmnist_published", "2.17a", "Fig. 16", 34),
        (34, 110, "ch2_fig2_17b_tempverse_mmnist_cifar_published", "2.17b", "Fig. 17", 35),
        (35, 113, "ch2_fig2_17c_tempverse_kth_published", "2.17c", "Fig. 18", 36),
        (36, 116, "ch2_fig2_17d_tempverse_taxibj_published", "2.17d", "Fig. 19", 37),
    ]
    for page_index, xref, stem, fid, original, page in tv_images:
        extract_pdf_image(
            TEMPVERSE_PDF,
            page_index,
            xref,
            stem,
            figure_id=fid,
            original_figure=original,
            source_page=page,
            source_label="Temporal Modeling with Reversible Transformers; DOI 10.1007/s00521-025-11707-3",
            rights_status=tv_rights,
        )

    vb_images = [
        (3, 48, "ch2_fig2_21_vblt_architecture_published", "2.21", "Fig. 3", 4),
        (5, 113, "ch2_fig2_22_vblt_causal_mask_published", "2.22", "Fig. 4", 6),
        (6, 118, "ch2_fig2_1_vblt_2d_sequence_published", "2.1", "Fig. 5", 7),
        (7, 127, "ch2_fig2_2_vblt_3d_sequence_published", "2.2", "Fig. 6", 8),
        (8, 130, "ch2_fig2_25a_vblt_2d_entropy_published", "2.25a", "Fig. 7", 9),
        (9, 133, "ch2_fig2_25b_vblt_3d_entropy_published", "2.25b", "Fig. 8", 10),
    ]
    for page_index, xref, stem, fid, original, page in vb_images:
        extract_pdf_image(
            VBLT_PDF,
            page_index,
            xref,
            stem,
            figure_id=fid,
            original_figure=original,
            source_page=page,
            source_label="Scaling Video Prediction with Spatio-Temporal Patches; DOI 10.31649/1997-9266-2025-182-5-129-139",
            rights_status=vb_rights,
        )


# ==============================================================================================
# Chapter 2: training-memory and spatial-representation visuals
# ==============================================================================================


def fig2_3_dataset_ladder() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.72, 3.0)
    ax.set_ylim(-0.16, 2.45)
    ax.axis("off")
    fig.suptitle("Набори даних виконують різні доказові ролі", x=0.08, y=0.985, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.08,
        0.935,
        "Розміщення є категоріальним: воно не задає числового рейтингу складності наборів.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.89)

    column_titles = [
        "Перевірка окремого\nмеханізму",
        "Стандартна прогностична\nкомпетентність",
        "Ідентифікація\nдовготривалої пам’яті",
    ]
    row_titles = [
        "Контрольовані\nпроцедурні сцени",
        "Стандартні, складені\nабо реальні сцени",
    ]
    role_colors = [BLUE, TEAL, ORANGE]

    for col in range(3):
        ax.add_patch(
            Rectangle(
                (col, 0), 1, 2,
                facecolor="#FAFBFD" if col != 2 else "#FFF9F1",
                edgecolor=GRID,
                linewidth=1.0,
                zorder=0,
            )
        )
        ax.text(col + 0.5, 2.17, column_titles[col], ha="center", va="center", fontsize=10.2, weight="bold", color=role_colors[col])
    ax.plot([0, 3], [1, 1], color=GRID, lw=1.0)
    for row, title in enumerate(row_titles):
        ax.text(-0.10, 1.5 - row, title, ha="right", va="center", fontsize=9.5, weight="bold", color="#263238")

    cards = [
        (0.5, 1.62, "2-D обертання", "TempVerseFormer; V-BLT", BLUE),
        (0.5, 1.28, "3-D полікуби", "V-BLT", BLUE),
        (1.5, 1.50, "Moving MNIST", "стандартна синтетична задача", TEAL),
        (2.5, 1.62, "LoopMom", "точно спостережуваний повторний вхід", ORANGE),
        (2.5, 1.26, "Довготривалий Moving-MNIST", "ідентичність, покриття, траєкторія", ORANGE),
        (1.5, 0.70, "Moving MNIST-CIFAR", "складена текстурована сцена", TEAL),
        (1.5, 0.42, "KTH Action", "природний рух людини", TEAL),
        (1.5, 0.14, "TaxiBJ", "сіткові міські потоки", TEAL),
    ]
    for x, y, title, subtitle, color in cards:
        width = 0.84
        height = 0.24
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2), width, height,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(x, y + 0.025, title, ha="center", va="center", fontsize=8.7, weight="bold", zorder=3)
        ax.text(x, y - 0.075, subtitle, ha="center", va="center", fontsize=6.9, color=MUTED, zorder=3)

    ax.text(
        0.03,
        -0.10,
        "Порожня клітинка означає відсутність окремого набору з такою роллю в цій дисертації, а не неможливість відповідного експерименту.",
        ha="left",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )
    save_native(fig, "ch2_fig2_3_dataset_ladder", "2.3", "Chapter 2 Table 2.1 and published experiment protocols")


def _tape_panel(ax, title_text: str, mode: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title_text, loc="left", fontsize=11, pad=8)
    xs = np.linspace(0.12, 0.88, 7)
    for i, x in enumerate(xs):
        retained = mode == "bptt" or (mode == "checkpoint" and i in {0, 3, 6}) or (mode == "reversible" and i == 6)
        fc = PALE_ORANGE if retained else "white"
        ec = ORANGE if retained else GRID
        ax.add_patch(Circle((x, 0.62), 0.045, facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(x, 0.62, f"h{i}", ha="center", va="center", fontsize=8)
        if i < 6:
            arrow(ax, (x + 0.05, 0.62), (xs[i + 1] - 0.05, 0.62), color=INK, lw=1.0)
    if mode == "bptt":
        ax.text(0.5, 0.40, "all intermediate activations retained", ha="center", color=ORANGE, fontsize=9)
        ax.plot([0.1, 0.9], [0.34, 0.34], color=ORANGE, lw=5, solid_capstyle="round")
    elif mode == "checkpoint":
        for a, b in [(0.12, 0.50), (0.50, 0.88)]:
            arrow(ax, (b, 0.42), (a, 0.42), color=TEAL, dashed=True, lw=1.3)
        ax.text(0.5, 0.30, "segment replay from stored checkpoints", ha="center", color=TEAL, fontsize=9)
    else:
        for i in range(6, 0, -1):
            arrow(ax, (xs[i] - 0.01, 0.42), (xs[i - 1] + 0.01, 0.42), color=PURPLE, dashed=True, lw=1.2)
        ax.text(0.5, 0.28, "reconstruct one preceding state, rebuild local graph", ha="center", color=PURPLE, fontsize=8.8)
    box(ax, 0.12, 0.06, 0.32, 0.11, "parameters + optimizer", fc=PALE_BLUE, ec=BLUE, fs=8.5)
    box(ax, 0.56, 0.05, 0.32, 0.13, "recurrent state /\nmetadata", fc=PALE_GRAY, ec=MUTED, fs=8.1)


def fig2_4_storage_schedules() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), facecolor="white", constrained_layout=True)
    fig.suptitle("Differentiation schedules retain different temporal information", fontsize=15, weight="bold")
    _tape_panel(axes[0], "(a) standard BPTT", "bptt")
    _tape_panel(axes[1], "(b) checkpointed segments", "checkpoint")
    _tape_panel(axes[2], "(c) reversible local replay", "reversible")
    fig.text(
        0.5,
        0.005,
        "Activation tape, recurrent state and replay metadata are distinct; parameters and optimizer state are common to all three.",
        ha="center",
        fontsize=9.2,
        color=MUTED,
    )
    save_native(fig, "ch2_fig2_4_storage_schedules", "2.4", "Conceptual comparison grounded in Chapter 2 Section 2.2.1")


def fig2_20_fixed_patch_flow() -> None:
    fig, ax = canvas(13, 7)
    title(ax, "Measured V-BLT path uses a fixed 4 x 4 x 4 support", "The dashed branch is a proposal, not part of the reported experiment")
    # Tensor block.
    x0, y0 = 0.045, 0.41
    for k in range(4):
        dx, dy = 0.018 * k, 0.018 * k
        ax.add_patch(Rectangle((x0 + dx, y0 + dy), 0.15, 0.22, facecolor="white", edgecolor=BLUE, linewidth=1.2))
        for q in range(1, 4):
            ax.plot([x0 + dx + 0.15 * q / 4] * 2, [y0 + dy, y0 + dy + 0.22], color=GRID, lw=0.6)
            ax.plot([x0 + dx, x0 + dx + 0.15], [y0 + dy + 0.22 * q / 4] * 2, color=GRID, lw=0.6)
    ax.text(0.13, 0.33, "T x H x W support\n4 x 4 x 4; channel bytes inside sites", ha="center", fontsize=9)
    stages = [
        (0.27, "local\nencoder", PALE_BLUE, BLUE),
        (0.45, "fixed latent\npatch sequence", PALE_TEAL, TEAL),
        (0.64, "global causal\ntransformer", PALE_PURPLE, PURPLE),
        (0.83, "local\ndecoder", PALE_ORANGE, ORANGE),
    ]
    for x, label, fc, ec in stages:
        box(ax, x, 0.48, 0.12, 0.15, label, fc=fc, ec=ec, weight="bold")
    arrow(ax, (0.21, 0.555), (0.27, 0.555))
    for a, b in zip(stages[:-1], stages[1:]):
        arrow(ax, (a[0] + 0.12, 0.555), (b[0], 0.555))
    box(
        ax,
        0.39,
        0.13,
        0.34,
        0.13,
        "proposed entropy-driven variable support\nnot implemented or evaluated",
        fc="white",
        ec=ORANGE,
        dashed=True,
        fs=9.2,
    )
    arrow(ax, (0.51, 0.46), (0.51, 0.27), color=ORANGE, dashed=True)
    ax.text(0.91, 0.36, "categorical\nbyte logits", ha="center", fontsize=8.8, color=ORANGE)
    arrow(ax, (0.89, 0.48), (0.91, 0.40), color=ORANGE)
    ax.text(0.5, 0.055, "Local-global-local processing is hierarchical; the measured patch geometry remains static.", ha="center", fontsize=9.5)
    save_native(fig, "ch2_fig2_20_fixed_patch_flow", "2.20", "Published V-BLT fixed 4 x 4 x 4 specification and audited implementation boundary")


def fig2_24_entropy_pipeline() -> None:
    fig, ax = canvas(13.2, 6.5)
    title(ax, "Categorical byte predictions expose a spatial dispersion diagnostic")
    stages = [
        (0.04, 0.52, 0.14, "256 logits\nper byte", PALE_BLUE, BLUE),
        (0.23, 0.52, 0.13, "softmax\np(b)", PALE_TEAL, TEAL),
        (0.41, 0.52, 0.16, "entropy\n- sum p log p", PALE_GOLD, GOLD),
        (0.62, 0.52, 0.15, "reshape\nB,T,C,H,W", PALE_PURPLE, PURPLE),
        (0.82, 0.52, 0.14, "display\nchannel plane", PALE_ORANGE, ORANGE),
    ]
    for x, y, w, label, fc, ec in stages:
        box(ax, x, y, w, 0.15, label, fc=fc, ec=ec, weight="bold")
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(ax, (left[0] + left[2], 0.595), (right[0], 0.595))
    box(ax, 0.39, 0.23, 0.20, 0.12, "optional declared\nchannel aggregation", fc="white", ec=MUTED, dashed=True, fs=8.8)
    arrow(ax, (0.695, 0.51), (0.59, 0.35), color=MUTED, dashed=True)
    arrow(ax, (0.59, 0.29), (0.85, 0.51), color=MUTED, dashed=True)
    ax.text(0.13, 0.39, "categorical raster\npixel-channel byte", ha="center", fontsize=8.8, color=MUTED)
    ax.text(0.49, 0.405, "nats in audited code", ha="center", fontsize=8.8, color=GOLD)
    ax.text(0.89, 0.38, "clamp: 6 nats\ninverted grayscale", ha="center", fontsize=8.8, color=ORANGE)
    box(
        ax,
        0.20,
        0.055,
        0.60,
        0.09,
        "Not calibration, epistemic uncertainty, or a validated dynamic patch controller",
        fc=PALE_GRAY,
        ec=MUTED,
        fs=9.2,
    )
    save_native(fig, "ch2_fig2_24_entropy_pipeline", "2.24", "Audited V-BLT categorical-output and entropy-visualization path")


def fig2_26_vblt_point_estimates() -> None:
    models = ["V-BLT", "ViViT", "UNet-3D"]
    colors = [BLUE, TEAL, ORANGE]
    sizes = np.array([6.5, 6.5, 24.0])
    gflops = np.array([51.2, 59.3, 70.4])
    mse_2d, mse_3d = np.array([0.039, 0.181, 0.125]), np.array([0.11, 0.19, 0.20])
    ssim_2d, ssim_3d = np.array([0.98, 0.87, 0.88]), np.array([0.91, 0.83, 0.75])
    psnr_2d, psnr_3d = np.array([32.3, 20.7, 21.8]), np.array([23.4, 24.8, 20.3])
    fig = plt.figure(figsize=(14.5, 7.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.88, 1.55], left=0.055, right=0.985, bottom=0.15, top=0.87, wspace=0.25)
    left = gs[0].subgridspec(2, 1, hspace=0.30)
    right = gs[1].subgridspec(1, 3, wspace=0.28)
    ax_size = fig.add_subplot(left[0])
    ax_flop = fig.add_subplot(left[1])
    for ax, values, label in [(ax_size, sizes, "model size (MB)"), (ax_flop, gflops, "GFLOPs")]:
        ax.bar(models, values, color=colors, width=0.62)
        ax.set_ylabel(label)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            ax.text(i, value + max(values) * 0.035, f"{value:g}", ha="center", fontsize=9)
    x = np.arange(3)
    width = 0.34
    metric_axes = []
    for slot, a, b, ylabel, ylim in [
        (right[0], mse_2d, mse_3d, "MSE (lower)", (0, 0.22)),
        (right[1], ssim_2d, ssim_3d, "SSIM (higher)", (0.68, 1.02)),
        (right[2], psnr_2d, psnr_3d, "PSNR dB (higher)", (18, 35)),
    ]:
        ax = fig.add_subplot(slot)
        metric_axes.append(ax)
        ax.bar(x - width / 2, a, width, color=BLUE, label="2-D shapes")
        ax.bar(x + width / 2, b, width, color=ORANGE, label="3-D polycubes")
        ax.set_xticks(x, models, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    metric_axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    metric_axes[2].annotate(
        "ViViT has the highest\nreported 3-D PSNR",
        xy=(1 + width / 2, 24.8),
        xytext=(0.25, 31.5),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.2},
        color=ORANGE,
        fontsize=8.5,
    )
    fig.suptitle("Published V-BLT cost and quality point estimates", fontsize=15, weight="bold")
    fig.text(
        0.5,
        0.025,
        "Published point estimates; uncertainty not reported. Metrics retain separate axes and are not combined into one rank.",
        ha="center",
        fontsize=9.4,
        color=MUTED,
    )
    save_native(fig, "ch2_fig2_26_vblt_point_estimates", "2.26", "V-BLT published Tables 1-2; Chapter 2 Table 2.9")


def fig2_27_measured_vs_proposed() -> None:
    fig, ax = canvas(13.2, 7.0)
    title(ax, "Measured representation and proposed allocation are different claims")
    box(ax, 0.05, 0.18, 0.39, 0.62, "", fc=PALE_GREEN, ec=GREEN, lw=2.2)
    box(ax, 0.56, 0.18, 0.39, 0.62, "", fc="white", ec=ORANGE, lw=2.2, dashed=True)
    ax.text(0.245, 0.74, "IMPLEMENTED / EVALUATED", ha="center", color=GREEN, weight="bold", fontsize=11)
    ax.text(0.755, 0.74, "PROPOSED FUTURE EXTENSION", ha="center", color=ORANGE, weight="bold", fontsize=11)
    left = ["fixed 4 x 4 x 4 support", "local / global / local model", "RoPE-ND coordinates", "256-way byte objective", "predictive entropy maps"]
    right = ["calibrated allocation signal", "variable-region controller", "budget-constrained routing", "allocation-overhead accounting", "end-to-end compute-quality test"]
    for i, text in enumerate(left):
        y = 0.64 - i * 0.095
        ax.text(0.10, y, "●", color=GREEN, fontsize=10, va="center")
        ax.text(0.13, y, text, fontsize=9.5, va="center")
    for i, text in enumerate(right):
        y = 0.64 - i * 0.095
        ax.text(0.61, y, "○", color=ORANGE, fontsize=11, va="center")
        ax.text(0.64, y, text, fontsize=9.5, va="center")
    arrow(ax, (0.445, 0.49), (0.555, 0.49), color=ORANGE, dashed=True, lw=2.0)
    ax.text(0.50, 0.54, "motivates", ha="center", color=ORANGE, fontsize=9)
    ax.text(
        0.5,
        0.075,
        "Entropy structure justifies a testable policy hypothesis; it is not evidence that dynamic allocation already improves efficiency.",
        ha="center",
        fontsize=9.5,
    )
    save_native(fig, "ch2_fig2_27_measured_vs_proposed", "2.27", "Published V-BLT evidence and implementation audit")


# ==============================================================================================
# Chapter 3: hierarchical context, rollout, and evaluation visuals
# ==============================================================================================


def _timeline_cells(ax, y: float, xs: Iterable[float], color: str, label: str, size: float = 0.022) -> None:
    for x in xs:
        ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, facecolor=color, edgecolor="white", linewidth=0.4))
    ax.text(0.025, y, label, ha="left", va="center", fontsize=8.5, color=color, weight="bold")


def fig3_1_reentry_memory() -> None:
    fig, ax = canvas(13, 7.2)
    title(ax, "Re-entry can depend on an appearance outside a short dense window")
    rows = [(0.72, "world and viewport"), (0.46, "last detailed observation"), (0.20, "prediction boundary")]
    for y, label in rows:
        ax.text(0.025, y + 0.07, label, fontsize=9.5, weight="bold", color=MUTED)
        ax.add_patch(Rectangle((0.15, y - 0.04), 0.78, 0.14, facecolor=PALE_GRAY, edgecolor=INK, linewidth=1.2))
        for k in range(1, 3):
            ax.plot([0.15 + 0.26 * k] * 2, [y - 0.04, y + 0.10], color=GRID, lw=1)
        viewport_x = {0.72: 0.15, 0.46: 0.41, 0.20: 0.67}[y]
        ax.add_patch(Rectangle((viewport_x, y - 0.04), 0.26, 0.14, fill=False, edgecolor=BLUE, linewidth=2.2))
        digit_x = {0.72: 0.31, 0.46: 0.62, 0.20: 0.75}[y]
        ax.text(digit_x, y + 0.027, "5", ha="center", va="center", fontsize=29, color=ORANGE, weight="bold")
    ax.text(0.30, 0.365, "older observation: retained only at coarse level", ha="center", fontsize=9, color=BLUE)
    arrow(ax, (0.48, 0.43), (0.35, 0.28), color=BLUE, curve=0.12)
    ax.add_patch(Rectangle((0.16, 0.105), 0.45, 0.055, facecolor="#E5E7EB", edgecolor="none"))
    ax.add_patch(Rectangle((0.61, 0.105), 0.15, 0.055, facecolor=ORANGE, edgecolor="none"))
    ax.add_patch(Rectangle((0.16, 0.055), 0.60, 0.032, facecolor=BLUE, edgecolor="none"))
    ax.text(0.385, 0.132, "outside short fine window", ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.text(0.685, 0.132, "recent fine", ha="center", va="center", fontsize=8.5, color="white", weight="bold")
    ax.text(0.46, 0.071, "long coarse history remains addressable", ha="center", va="center", fontsize=8.5, color="white", weight="bold")
    ax.text(0.88, 0.13, "target re-entry", ha="center", va="center", fontsize=9.2, color=ORANGE, weight="bold")
    arrow(ax, (0.77, 0.13), (0.84, 0.13), color=ORANGE)
    save_native(fig, "ch3_fig3_1_reentry_memory", "3.1", "LoopMom geometry and Chapter 3 conceptual re-entry definition")


def fig3_3_reach_fidelity_axes() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("addressable historical reach", fontsize=11)
    ax.set_ylabel("retained task-relevant fidelity", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Reach and fidelity are separate package properties", loc="left", pad=14)
    threshold = 0.56
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.5)
    ax.text(threshold + 0.018, 0.94, "task event: relevant past becomes included", rotation=90, va="top", fontsize=8.8, color=MUTED)
    points = [
        (0.30, 0.86, "dense short\nflat", ORANGE, "full detail; event absent"),
        (0.79, 0.50, "long coarse\n+ recent fine", BLUE, "event present; old detail compressed"),
        (0.78, 0.78, "sparse fine anchors\n+ recent tail", TEAL, "fine anchors; gaps between them"),
    ]
    for x, y, label, color, note in points:
        ax.scatter([x], [y], s=240, facecolor="white", edgecolor=color, linewidth=2.3, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, y - 0.11, note, ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.text(0.5, 0.035, "Conceptual positions only: no fitted curve, result frontier, or interpolated crossing.", ha="center", fontsize=9.2, color=MUTED)
    save_native(fig, "ch3_fig3_3_reach_fidelity_axes", "3.3", "Conceptual package hypothesis; no empirical values")


def fig3_4_codec_dimensions() -> None:
    fig, ax = canvas(14, 7.2)
    title(ax, "Frozen two-level codec exposes exact coarse and fine token rates")
    box(ax, 0.035, 0.50, 0.12, 0.15, "input\n1 x 64 x 64", fc=PALE_GRAY, ec=INK, weight="bold")
    box(ax, 0.21, 0.50, 0.15, 0.15, "shared conv trunk\n128 x 8 x 8", fc=PALE_BLUE, ec=BLUE, weight="bold")
    arrow(ax, (0.155, 0.575), (0.21, 0.575))
    box(ax, 0.43, 0.67, 0.15, 0.13, "1 x 1 + tanh\n16 x 8 x 8", fc=PALE_ORANGE, ec=ORANGE, weight="bold")
    box(ax, 0.43, 0.35, 0.15, 0.15, "stride 2 -> 128 x 4 x 4\n1 x 1 + tanh\n16 x 4 x 4", fc=PALE_TEAL, ec=TEAL, weight="bold")
    arrow(ax, (0.36, 0.59), (0.43, 0.735), color=ORANGE)
    arrow(ax, (0.36, 0.555), (0.43, 0.425), color=TEAL)
    box(ax, 0.65, 0.67, 0.14, 0.13, "fine decoder\n1 x 64 x 64", fc=PALE_ORANGE, ec=ORANGE)
    box(ax, 0.65, 0.35, 0.14, 0.15, "coarse decoder\n1 x 64 x 64", fc=PALE_TEAL, ec=TEAL)
    arrow(ax, (0.58, 0.735), (0.65, 0.735), color=ORANGE)
    arrow(ax, (0.58, 0.425), (0.65, 0.425), color=TEAL)
    box(ax, 0.84, 0.67, 0.12, 0.13, "64 tokens\nper frame", fc="white", ec=ORANGE, weight="bold")
    box(ax, 0.84, 0.35, 0.12, 0.15, "16 tokens\nper frame", fc="white", ec=TEAL, weight="bold")
    arrow(ax, (0.79, 0.735), (0.84, 0.735), color=ORANGE)
    arrow(ax, (0.79, 0.425), (0.84, 0.425), color=TEAL)
    ax.plot([0.18, 0.98], [0.26, 0.26], color=MUTED, linestyle="--", linewidth=1.4)
    ax.text(0.58, 0.225, "frozen after codec training - predictor gradients stop at standardized latent values", ha="center", color=MUTED, fontsize=9.3)
    box(ax, 0.30, 0.07, 0.56, 0.095, "per-level, per-channel mean and standard deviation computed on training clips only", fc=PALE_PURPLE, ec=PURPLE, fs=9)
    save_native(fig, "ch3_fig3_4_codec_dimensions", "3.4", "Implemented HierConvAE tensor shapes and frozen latent interface")


def fig3_6_joint_attention() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.84, bottom=0.18, wspace=0.24)
    fig.suptitle("Joint attention is full inside a block; causality is structural across rollout blocks", fontsize=14.5, weight="bold")
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(a) one WindowDiT prediction block", loc="left", fontsize=11)
    groups = [("coarse\ncontext", BLUE), ("fine\ncontext", ORANGE), ("noisy coarse\ntarget", TEAL), ("noisy fine\ntarget", PURPLE)]
    for i, (label, color) in enumerate(groups):
        y = 0.78 - i * 0.18
        box(ax, 0.05, y, 0.25, 0.105, label, fc="white", ec=color, fs=9)
        arrow(ax, (0.30, y + 0.052), (0.43, 0.52), color=color)
    box(ax, 0.43, 0.37, 0.38, 0.30, "single all-to-all\nattention matrix\n\nfull attention within block\n(no triangular intra-block mask)", fc=PALE_BLUE, ec=BLUE, weight="bold")
    arrow(ax, (0.81, 0.52), (0.94, 0.52), color=INK)
    ax.text(0.94, 0.45, "coarse + fine\noutputs", ha="center", fontsize=9)
    ax = axes[1]
    ax.set_title("(b) rollout-block availability", loc="left", fontsize=11)
    matrix = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            matrix[i, j] = 1 if j <= i else 0
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(["#F4D6D6", PALE_TEAL]), vmin=0, vmax=1)
    ax.set_xticks(range(5), [f"source {i}" for i in range(5)], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(5), [f"current {i}" for i in range(5)], fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "available" if j <= i else "future", ha="center", va="center", fontsize=7.5, color=GREEN if j <= i else RED)
    ax.set_xlabel("completed blocks may supply context; future blocks do not exist yet", fontsize=9)
    ax.set_ylabel("prediction block")
    fig.text(0.5, 0.025, "Panel (b) describes rollout construction, not a layer-level attention mask.", ha="center", fontsize=9.3, color=MUTED)
    save_native(fig, "ch3_fig3_6_joint_attention", "3.6", "Implemented WindowDiT forward path and rollout recurrence")


def fig3_7_gradient_paths() -> None:
    fig, ax = canvas(14, 7)
    title(ax, "Training updates the predictor but not the codec or previous rollout windows")
    box(ax, 0.035, 0.58, 0.13, 0.14, "frozen\nencoder", fc=PALE_GRAY, ec=MUTED, weight="bold")
    box(ax, 0.21, 0.58, 0.16, 0.14, "detached standardized\ncontext tokens", fc=PALE_BLUE, ec=BLUE, weight="bold")
    box(ax, 0.43, 0.58, 0.13, 0.14, "trainable token\nprojection", fc=PALE_PURPLE, ec=PURPLE)
    box(ax, 0.62, 0.58, 0.15, 0.14, "8 trainable\ntransformer blocks", fc=PALE_PURPLE, ec=PURPLE, weight="bold")
    box(ax, 0.83, 0.58, 0.12, 0.14, "trainable\noutput heads", fc=PALE_PURPLE, ec=PURPLE)
    for a, b, color in [(0.165, 0.21, MUTED), (0.37, 0.43, BLUE), (0.56, 0.62, PURPLE), (0.77, 0.83, PURPLE)]:
        arrow(ax, (a, 0.65), (b, 0.65), color=color)
    box(ax, 0.80, 0.28, 0.16, 0.13, "coarse/fine\ntarget loss", fc=PALE_ORANGE, ec=ORANGE, weight="bold")
    arrow(ax, (0.89, 0.58), (0.88, 0.41), color=ORANGE)
    arrow(ax, (0.80, 0.345), (0.50, 0.50), color=PURPLE, curve=-0.12)
    box(ax, 0.07, 0.25, 0.22, 0.13, "previous-window output\nin rollout buffer", fc="white", ec=ORANGE, dashed=True)
    arrow(ax, (0.29, 0.315), (0.44, 0.57), color=ORANGE, dashed=True)
    ax.text(0.34, 0.44, "stop gradient", color=RED, fontsize=8.8, weight="bold")
    ax.text(0.18, 0.52, "STOP", color=RED, fontsize=10, weight="bold")
    ax.text(0.83, 0.15, "frozen fine decoder\nused for evaluation", ha="center", fontsize=9, color=MUTED)
    box(ax, 0.07, 0.055, 0.18, 0.08, "trainable", fc=PALE_PURPLE, ec=PURPLE, fs=8.5)
    box(ax, 0.29, 0.055, 0.18, 0.08, "frozen", fc=PALE_GRAY, ec=MUTED, fs=8.5)
    box(ax, 0.51, 0.055, 0.18, 0.08, "detached value", fc=PALE_BLUE, ec=BLUE, fs=8.5)
    save_native(fig, "ch3_fig3_7_gradient_paths", "3.7", "Implemented optimization graph and frozen-codec boundary")


def fig3_8_objective_interfaces() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.7), facecolor="white", constrained_layout=True)
    fig.suptitle("Three prediction objectives expose different training and inference interfaces", fontsize=14.5, weight="bold")
    specs = [
        ("(a) diffusion - principal head", BLUE, ["clean 4-frame\ncoarse + fine targets", "shared time; independent\nnoise by level", "noisy targets + clean\ncontext + actions", "one WindowDiT", "coarse/fine v targets\nsummed MSE"]),
        ("(b) regression - capacity control", TEAL, ["zero target\nplaceholders", "clean context\n+ actions", "one WindowDiT\nforward pass", "coarse/fine x0", "smooth-L1"]),
        ("(c) energy - separate capability", PURPLE, ["candidate future\nlatents", "context + candidate\n+ step embedding", "scalar energy", "gradient update of\ncandidate state", "not a matched\nWindowDiT result cell"]),
    ]
    for ax, (heading, color, nodes) in zip(axes, specs):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(heading, loc="left", color=color, fontsize=11)
        ys = np.linspace(0.78, 0.18, len(nodes))
        for i, (node, y) in enumerate(zip(nodes, ys)):
            fc = "white" if i not in {2, 3} else PALE_BLUE if color == BLUE else PALE_TEAL if color == TEAL else PALE_PURPLE
            box(ax, 0.14, y - 0.052, 0.72, 0.104, node, fc=fc, ec=color, fs=8.8, weight="bold" if i in {2, 3} else "normal")
            if i < len(nodes) - 1:
                arrow(ax, (0.50, y - 0.055), (0.50, ys[i + 1] + 0.055), color=color)
    fig.text(0.5, 0.006, "The figure defines interfaces only; outcome comparisons appear in Chapter 4.", ha="center", fontsize=9.3, color=MUTED)
    save_native(fig, "ch3_fig3_8_objective_interfaces", "3.8", "Implemented diffusion/regression interfaces and separately scaled EBT capability study")


def fig3_10_ddim_timeline() -> None:
    fig, ax = canvas(14, 6.4)
    title(ax, "Deterministic DDIM reuses clean conditioning at every denoiser call", "eta = 0; sampler variation comes from the initial noise")
    xs = np.linspace(0.12, 0.88, 16)
    colors = [plt.cm.Blues(0.25 + 0.65 * i / 15) for i in range(16)]
    for i, (x, color) in enumerate(zip(xs, colors)):
        ax.add_patch(Circle((x, 0.58), 0.026, facecolor=color, edgecolor=BLUE, linewidth=0.7))
        if i < 15:
            arrow(ax, (x + 0.029, 0.58), (xs[i + 1] - 0.029, 0.58), color=BLUE, lw=0.8)
    ax.text(xs[0], 0.67, "Gaussian coarse/fine\n4-frame target", ha="center", fontsize=8.5)
    ax.text(xs[-1], 0.67, "clean latent\nestimate", ha="center", fontsize=8.5)
    box(ax, 0.38, 0.31, 0.24, 0.12, "shared WindowDiT denoiser", fc=PALE_BLUE, ec=BLUE, weight="bold")
    for x in xs[::3]:
        arrow(ax, (0.50, 0.43), (x, 0.545), color=MUTED, lw=0.8)
    box(ax, 0.08, 0.15, 0.24, 0.10, "clean packaged context", fc=PALE_TEAL, ec=TEAL)
    box(ax, 0.38, 0.15, 0.24, 0.10, "target-frame actions", fc=PALE_ORANGE, ec=ORANGE)
    box(ax, 0.68, 0.15, 0.24, 0.10, "diffusion-time embedding", fc=PALE_PURPLE, ec=PURPLE)
    for x in [0.20, 0.50, 0.80]:
        arrow(ax, (x, 0.25), (0.50, 0.31), color=TEAL if x == 0.20 else ORANGE if x == 0.50 else PURPLE)
    ax.text(0.5, 0.055, "16 NFE per four-frame block = 4 NFE/frame; 38 rollout blocks = 608 NFE.", ha="center", fontsize=10, weight="bold")
    save_native(fig, "ch3_fig3_10_ddim_timeline", "3.10", "Implemented DDIM-16 rollout accounting")


def fig3_12_anchor_geometry() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.2), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-50, 5)
    ax.set_ylim(-0.7, 5.8)
    ax.axvline(0, color=INK, linewidth=1.8)
    ax.text(0.3, 5.45, "first target t", fontsize=9.5, weight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([5, 4, 3, 2, 1, 0], ["coarse dense", "recent fine", "flat-18", "sparse old fine", "sparse recent fine", "targets / actions"])
    ax.set_xlabel("frame offset relative to first target t")
    ax.set_title("Exact anchors, not a nominal reach label, define package semantics", loc="left", pad=14)
    ax.scatter(np.arange(-48, 0), np.full(48, 5), s=24, color=BLUE, label="coarse token frame")
    ax.scatter(np.arange(-6, 0), np.full(6, 4), s=45, marker="s", color=ORANGE, label="fine token frame")
    ax.scatter(np.arange(-18, 0), np.full(18, 3), s=42, marker="s", color=ORANGE)
    sparse_old = np.array([-47, -43, -39, -35, -31, -27, -23, -19, -15, -11, -7])
    ax.scatter(sparse_old, np.full(len(sparse_old), 2), s=48, marker="D", color=TEAL)
    ax.scatter(np.arange(-6, 0), np.full(6, 1), s=45, marker="s", color=ORANGE)
    targets = np.arange(0, 4)
    ax.scatter(targets, np.full(4, 0), s=70, marker="s", facecolor=PALE_PURPLE, edgecolor=PURPLE, linewidth=1.5)
    for j in targets:
        ax.text(j, 0.22, "t" if j == 0 else f"t+{j}", ha="center", fontsize=7.2, color=PURPLE)
    ax.text(1.5, -0.34, "actions u_t ... u_(t+3)", ha="center", fontsize=8.2, color=PURPLE)
    ax.text(-24, 5.30, "t-48 ... t-1", ha="center", color=BLUE, fontsize=8.5)
    ax.text(-3.5, 4.30, "t-6 ... t-1", ha="center", color=ORANGE, fontsize=8.5)
    ax.text(-9.5, 3.30, "t-18 ... t-1", ha="center", color=ORANGE, fontsize=8.5)
    ax.text(-26.5, 2.30, "t-47, t-43, ..., t-7", ha="center", color=TEAL, fontsize=8.5)
    ax.text(-3.5, 1.30, "t-6 ... t-1", ha="center", color=ORANGE, fontsize=8.5)
    ax.text(-49.5, -0.52, "Registered nominal reach never substitutes for this anchor list.", fontsize=9, color=MUTED)
    save_native(fig, "ch3_fig3_12_anchor_geometry", "3.12", "Implemented WindowDiT._ctx_tokens anchor formula; strided-flat anchors verified as lags 47,43,...,7 plus 6,...,1")


def fig3_13_token_costs() -> None:
    packages = ["48 fine", "48 coarse\n+ 6 fine", "48 coarse\n+ 2 fine"]
    context = np.array([3072, 1152, 896])
    payload = np.array([192, 72, 56])
    total = np.array([3392, 1472, 1216])
    colors = [ORANGE, BLUE, TEAL]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.20, wspace=0.30)
    fig.suptitle("Equal nominal reach has different context, payload and total-token costs", fontsize=14.5, weight="bold")
    for ax, values, ylabel, suffix in [
        (axes[0], context, "context tokens", ""),
        (axes[1], payload, "fp32 latent payload (KiB)", " KiB"),
        (axes[2], total, "total tokens incl. common 320 target", ""),
    ]:
        bars = ax.bar(packages, values, color=colors, width=0.64)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.035, f"{value:,}{suffix}", ha="center", fontsize=9)
    fig.text(0.5, 0.03, "Exact tensor arithmetic; bars are not measured VRAM, runtime, or attention-cost ratios.", ha="center", fontsize=9.2, color=MUTED)
    save_native(fig, "ch3_fig3_13_token_costs", "3.13", "Frozen codec/package registry and Chapter 3 Table 3.3")


def fig3_15_reentry_storyboard() -> None:
    fig, axes = plt.subplots(1, 6, figsize=(15.5, 4.8), facecolor="white", constrained_layout=True)
    fig.suptitle("A LoopMom re-entry event separates recent continuity from retained history", fontsize=14.5, weight="bold")
    labels = [
        "visible and\nremembered",
        "leaving",
        "fully absent",
        "short window has\nno digit evidence",
        "old coarse view\nstill supplied",
        "re-entry target",
    ]
    positions = [0.45, 0.90, 1.35, 1.70, 2.05, 2.42]
    times = ["t-48", "t-42", "t-30", "t-15", "t-7", "t"]
    digit_positions = [0.52, 0.88, None, None, None, 0.18]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0.07, 0.31), 0.86, 0.48, facecolor=PALE_GRAY, edgecolor=INK, linewidth=1.2))
        for x in np.linspace(0.2, 0.8, 4):
            ax.plot([x, x], [0.31, 0.79], color=GRID, lw=0.5)
        if digit_positions[i] is not None:
            ax.text(digit_positions[i], 0.54, "5", fontsize=34, color=ORANGE, weight="bold", ha="center", va="center", clip_on=True)
        if i == 4:
            ax.add_patch(Rectangle((0.23, 0.40), 0.52, 0.28, facecolor=PALE_BLUE, edgecolor=BLUE, linestyle="--", linewidth=1.8))
            ax.text(0.49, 0.54, "coarse old\nobservation", ha="center", va="center", fontsize=8, color=BLUE)
        if i == 5:
            ax.add_patch(Rectangle((0.07, 0.31), 0.86, 0.48, fill=False, edgecolor=RED, linewidth=1.5, linestyle=":"))
            ax.text(0.5, 0.84, "ground-truth evaluation geometry", ha="center", fontsize=7.5, color=RED)
        ax.text(0.5, 0.21, labels[i], ha="center", va="top", fontsize=8.6, weight="bold")
        ax.text(0.5, 0.06, f"viewport x={positions[i]:.2f}W; {times[i]}", ha="center", fontsize=7.5, color=MUTED)
    fig.text(0.5, 0.005, "Illustrative generator frames only; no model prediction or performance claim is shown.", ha="center", fontsize=9, color=MUTED)
    save_native(fig, "ch3_fig3_15_reentry_storyboard", "3.15", "LoopMom generator geometry; illustrative synthetic storyboard")


def fig3_16_reach_population() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.4), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time since last relevant visibility, Delta t (frames)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Pair-specific reach populations follow from package geometry", loc="left", pad=14)
    ax.axvspan(0, 14, color=PALE_TEAL)
    ax.axvspan(14, 48, color=PALE_BLUE)
    ax.axvspan(48, 65, color=PALE_GRAY)
    ax.text(7, 0.61, "inside both reaches", ha="center", fontsize=10, weight="bold", color=GREEN)
    ax.text(31, 0.61, "inside only the longer reach", ha="center", fontsize=10, weight="bold", color=BLUE)
    ax.text(56.5, 0.61, "beyond both", ha="center", fontsize=10, weight="bold", color=MUTED)
    ax.axvline(14, color=INK, linewidth=1.6)
    ax.axvline(48, color=INK, linewidth=1.6)
    ax.text(14, 0.80, "short reach r_s", ha="center", fontsize=9)
    ax.text(48, 0.80, "long reach r_l", ha="center", fontsize=9)
    ax.scatter([31], [0.36], s=150, facecolor="white", edgecolor=BLUE, linewidth=2)
    ax.annotate("one target record", xy=(31, 0.36), xytext=(37, 0.18), arrowprops={"arrowstyle": "-|>", "color": BLUE}, fontsize=9, color=BLUE)
    box_ax = ax.inset_axes([0.18, 0.02, 0.64, 0.18])
    box_ax.axis("off")
    box_ax.text(0.5, 0.70, "mechanism population:  r_s < Delta t <= r_l", ha="center", weight="bold", fontsize=10)
    box_ax.text(0.5, 0.25, "gates: non-empty target foreground  AND  rho >= 0.5", ha="center", fontsize=9, color=MUTED)
    ax.text(31, 0.92, "registered example: 14 < Delta t <= 48", ha="center", color=BLUE, fontsize=10, weight="bold")
    save_native(fig, "ch3_fig3_16_reach_population", "3.16", "Registered pair-specific E2 geometry and Chapter 3 Eq. 3.37")


def fig3_17_persistent_assignment() -> None:
    fig, ax = canvas(15, 7.8)
    title(ax, "Persistent-object scoring uses one-to-one assignment and fixed ground-truth denominators")
    # Three schematic frames.
    for i, x in enumerate([0.04, 0.25, 0.46]):
        ax.add_patch(Rectangle((x, 0.53), 0.17, 0.25, facecolor=PALE_GRAY, edgecolor=INK, linewidth=1.2))
        ax.text(x + 0.045 + 0.04 * i, 0.65, "3", fontsize=27, color=BLUE, weight="bold", ha="center")
        ax.text(x + 0.125 - 0.04 * i, 0.65, "8", fontsize=27, color=ORANGE, weight="bold", ha="center")
        ax.add_patch(Rectangle((x + 0.018 + 0.04 * i, 0.585), 0.055, 0.10, fill=False, edgecolor=BLUE, linewidth=1.3))
        ax.add_patch(Rectangle((x + 0.098 - 0.04 * i, 0.585), 0.055, 0.10, fill=False, edgecolor=ORANGE, linewidth=1.3))
        ax.text(x + 0.085, 0.50, ["before crossing", "crossing", "after crossing"][i], ha="center", fontsize=8.3)
    ax.text(0.035, 0.82, "generator IDs and boxes", fontsize=9.5, weight="bold", color=BLUE)
    # Cost matrix.
    mx, my = 0.69, 0.55
    values = [[0.14, 0.79], [0.74, 0.20]]
    ax.text(mx + 0.11, 0.82, "illustrative Hungarian cost", ha="center", fontsize=9.5, weight="bold")
    for r in range(2):
        for c in range(2):
            fc = PALE_GREEN if r == c else "white"
            ax.add_patch(Rectangle((mx + 0.11 * c, my + 0.11 * (1 - r)), 0.11, 0.11, facecolor=fc, edgecolor=INK, linewidth=1))
            ax.text(mx + 0.055 + 0.11 * c, my + 0.055 + 0.11 * (1 - r), f"{values[r][c]:.2f}", ha="center", va="center", fontsize=9)
    ax.text(mx - 0.015, my + 0.165, "GT 3", ha="right", va="center", fontsize=8)
    ax.text(mx - 0.015, my + 0.055, "GT 8", ha="right", va="center", fontsize=8)
    ax.text(mx + 0.055, my + 0.235, "pred A", ha="center", fontsize=8)
    ax.text(mx + 0.165, my + 0.235, "pred B", ha="center", fontsize=8)
    arrow(ax, (0.63, 0.655), (0.68, 0.655), color=PURPLE)
    box(ax, 0.05, 0.18, 0.24, 0.18, "accepted links\nGT 3 -> pred A\nGT 8 -> pred B", fc=PALE_GREEN, ec=GREEN, weight="bold")
    box(ax, 0.37, 0.18, 0.20, 0.18, "miss mini-case\none visible GT\nhas no accepted link", fc=PALE_ORANGE, ec=ORANGE)
    box(ax, 0.65, 0.18, 0.20, 0.18, "extra mini-case\none predicted component\nhas no GT link", fc=PALE_PURPLE, ec=PURPLE)
    box(ax, 0.88, 0.18, 0.09, 0.18, "coverage\nM / n_GT\n\nidentity\ncorrect /\nn_GT,labelled", fc=PALE_BLUE, ec=BLUE, fs=8.2)
    ax.text(0.5, 0.065, "Misses stay in trajectory and identity denominators; horizontal order is never used as object identity.", ha="center", fontsize=9.5)
    save_native(fig, "ch3_fig3_17_persistent_assignment", "3.17", "Frozen E6 evaluator contract; schematic values are illustrative, not experiment outcomes")


# ==============================================================================================
# Chapter 4: corrected negative evidence, untouched effects, arithmetic, and decision rule
# ==============================================================================================


EBT_PROBE = re.compile(
    r"\[scaled_s(?P<recipe>[12])_step(?P<step>\d+)\] landscape E\(last frame\): "
    r"tgt (?P<tgt>-?\d+\.\d+) \| mean (?P<mean>-?\d+\.\d+) \| "
    r"randn (?P<randn>-?\d+\.\d+).*?d_tgt (?P<d_tgt>\d+\.\d+).*?"
    r"fg-sharp (?P<sharp>\d+\.\d+)"
)


def parse_ebt(path: Path) -> list[dict]:
    rows: list[dict] = []
    for match in EBT_PROBE.finditer(path.read_text(encoding="utf-8")):
        row = {key: float(value) for key, value in match.groupdict().items()}
        row["recipe"] = int(row["recipe"])
        row["step"] = int(row["step"])
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no EBT probe rows parsed from {path}")
    return rows


def fig4_4_ebt_composite() -> None:
    s1 = parse_ebt(LOGS / "ebt_scaled_s1.log")
    s2 = parse_ebt(LOGS / "ebt_scaled_s2.log")
    fig = plt.figure(figsize=(16, 6.5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 1.15], left=0.025, right=0.99, bottom=0.14, top=0.84, wspace=0.16)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(a) candidate-state recursion", loc="left", fontsize=11)
    box(ax, 0.05, 0.68, 0.30, 0.14, "fixed context\n$c$", fc=PALE_BLUE, ec=BLUE, weight="bold", fs=9.0)
    box(ax, 0.05, 0.40, 0.30, 0.14, "candidate future\n$z^{(k)}$", fc=PALE_ORANGE, ec=ORANGE, weight="bold", fs=9.0)
    box(ax, 0.51, 0.55, 0.28, 0.17, "energy model\nE(c, z(k))", fc=PALE_PURPLE, ec=PURPLE, weight="bold")
    arrow(ax, (0.35, 0.75), (0.51, 0.66), color=BLUE)
    arrow(ax, (0.35, 0.47), (0.51, 0.60), color=ORANGE)
    box(ax, 0.45, 0.18, 0.44, 0.17,
        "candidate update\n$z^{(k+1)}=z^{(k)}-\\alpha\\nabla_zE+\\eta$",
        fc="white", ec=PURPLE, fs=7.9)
    arrow(ax, (0.66, 0.55), (0.66, 0.34), color=PURPLE)
    arrow(ax, (0.45, 0.265), (0.35, 0.43), color=PURPLE, curve=-0.15, lw=2.1)
    ax.text(0.27, 0.23, "update returns to\ncandidate future", ha="center", va="center", fontsize=8.0, color=PURPLE, weight="bold")
    ax.text(0.5, 0.06, "Context is fixed throughout descent.", ha="center", fontsize=9, color=MUTED)

    ax = fig.add_subplot(gs[1])
    ax.set_title("(b) learned energy ordering", loc="left", fontsize=11)
    series_colors = {"tgt": GREEN, "mean": BLUE, "randn": MUTED}
    labels = {"tgt": "target", "mean": "batch mean", "randn": "random"}
    for rows, recipe, ls in [(s1, "S1", "-"), (s2, "S2", "--")]:
        steps = [row["step"] for row in rows]
        for key in ["tgt", "mean", "randn"]:
            ax.plot(steps, [row[key] for row in rows], color=series_colors[key], linestyle=ls, linewidth=1.5, label=f"{recipe} {labels[key]}")
    ax.set_xlabel("training update")
    ax.set_ylabel("energy (lower preferred)")
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="best")
    ax.text(0.02, 0.02, "solid: S1; dashed: S2", transform=ax.transAxes, fontsize=8.3, color=MUTED)

    ax = fig.add_subplot(gs[2])
    ax.set_title("(c) cold-descent structure", loc="left", fontsize=11)
    for rows, label, color, marker in [(s1, "S1 learnable step", GREEN, "o"), (s2, "S2 fixed step", RED, "s")]:
        ax.plot([r["step"] for r in rows], [r["sharp"] for r in rows], color=color, marker=marker, markersize=3.5, linewidth=1.4, label=label)
    ax.axhline(0.08, color=INK, linestyle="--", linewidth=1.2, label="structure threshold")
    ax.axhline(0.24, color=MUTED, linestyle=":", linewidth=1.2, label="decoder reference ~0.24")
    ax.set_ylim(0, 0.26)
    ax.set_xlabel("training update")
    ax.set_ylabel("foreground sharpness")
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    fig.suptitle("Energy ordering can improve while candidate descent remains unreachable", fontsize=14.5, weight="bold")
    fig.text(0.5, 0.025, "Exact logged probes; no smoothing, interpolation, or uncertainty added.", ha="center", fontsize=9.2, color=MUTED)
    save_native(fig, "ch4_fig4_4_ebt_composite", "4.4", "logs_rhcp/ebt_scaled_s1.log and ebt_scaled_s2.log; hash-listed legacy receipts")


def fig4_6_reach_effects() -> None:
    data = json.loads(E2_JSON.read_text(encoding="utf-8"))
    rows = [r for r in data["primary"] if r["pair"] == "reach-48-vs-14" and r["metric"] in {"legacy_fg", "fg"}]
    lookup = {(r["seed"], r["metric"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(10.8, 6.2), facecolor="white")
    ax.set_facecolor("white")
    ybase = {0: 1.0, 1: 0.0}
    for seed in [0, 1]:
        whole = lookup[(seed, "legacy_fg")]
        band = lookup[(seed, "fg")]
        for r, offset, color, marker, label in [
            (band, 0.13, BLUE, "o", "registered 14 < Delta t <= 48 band" if seed == 0 else None),
            (whole, -0.13, MUTED, "s", "whole population (context)" if seed == 0 else None),
        ]:
            ax.errorbar(
                r["mean"],
                ybase[seed] + offset,
                xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
                markersize=7,
                markerfacecolor="white" if marker == "s" else color,
                label=label,
            )
            ax.text(r["hi"] + 0.003, ybase[seed] + offset, f"{r['mean']:+.4f} [{r['lo']:.4f}, {r['hi']:.4f}]", va="center", fontsize=8.5, color=color)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_yticks([0, 1], ["training seed 1", "training seed 0"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(-0.01, 0.145)
    ax.set_xlabel("paired foreground-error effect; positive favours hierarchy")
    ax.set_title("Untouched reach effects concentrate in the pair-derived population", loc="left", pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.8)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.20)
    fig.text(0.5, 0.035, "Clip-cluster intervals shown here are estimand-specific; they are not the simultaneous frontier family.", ha="center", fontsize=8.8, color=MUTED)
    save_native(fig, "ch4_fig4_6_reach_effects", "4.6", "artifacts/confirmation_e2/results.json and docs/confirmation/E2_RESULTS.md")


def fig4_8_equal_reach() -> None:
    fig = plt.figure(figsize=(14, 7.4), facecolor="white", constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1.0])
    packages = [
        ("48 dense fine", ORANGE, 3072, 96, 0),
        ("48 coarse + 6 recent fine", BLUE, 1152, 36, 6),
        ("48 coarse + 2 recent fine", TEAL, 896, 28, 2),
    ]
    max_tokens = 3072
    for row, (name, color, tokens, kib, fine_tail) in enumerate(packages):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_xlim(0, 48); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0, 0.90, name, fontsize=10, weight="bold", color=color)
        if row == 0:
            for i in range(48):
                ax.add_patch(Rectangle((i + 0.05, 0.33), 0.90, 0.28, facecolor=ORANGE, edgecolor="white", linewidth=0.25))
        else:
            for i in range(48):
                ax.add_patch(Rectangle((i + 0.05, 0.33), 0.90, 0.18, facecolor=color, edgecolor="white", linewidth=0.25))
            for i in range(48 - fine_tail, 48):
                ax.add_patch(Rectangle((i + 0.05, 0.54), 0.90, 0.16, facecolor=ORANGE, edgecolor="white", linewidth=0.25))
        ax.text(0, 0.12, "oldest", fontsize=7.8, color=MUTED)
        ax.text(48, 0.12, "recent", ha="right", fontsize=7.8, color=MUTED)
        ax = fig.add_subplot(gs[row, 1])
        ax.set_xlim(0, max_tokens * 1.18); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0, 0.35), tokens, 0.30, facecolor=color, edgecolor="none"))
        ax.text(tokens + 45, 0.50, f"{tokens:,} tokens\n{kib} KiB fp16", va="center", fontsize=9.5, weight="bold", color=color)
    fig.suptitle("Equal nominal 48-frame reach does not require equal context-state cost", fontsize=14.5, weight="bold")
    fig.text(0.5, 0.005, "Orange tail cells are recent fine maps; blue/teal base cells are coarse maps. Ratios exclude targets, activations, runtime and model parameters.", ha="center", fontsize=9.2, color=MUTED)
    save_native(fig, "ch4_fig4_8_equal_reach", "4.8", "Chapter 4 Table 4.5 exact token and fp16 payload arithmetic")


def fig1_18_systems_map() -> None:
    """Map one long-horizon problem, three constraints, and bounded integration evidence."""

    fig, ax = canvas(15.2, 8.4)
    title(
        ax,
        "Консистентність за обмежених ресурсів відеомоделі",
        "Одна довгогоризонтна проблема, центральне ієрархічне рішення та два засоби усунення обмежень масштабування",
    )
    box(
        ax,
        0.22,
        0.78,
        0.56,
        0.075,
        "Збереження релевантного стану й траєкторії на довгому горизонті",
        fc=PALE_BLUE,
        ec=BLUE,
        weight="bold",
        fs=10.5,
    )

    lanes = [
        {
            "x": 0.035,
            "w": 0.265,
            "color": PURPLE,
            "pale": PALE_PURPLE,
            "bottleneck": "Стрічка активацій зростає\nз горизонтом навчання",
            "method": "TempVerseFormer:\nреверсивне часове відтворення",
            "evidence": "Опублікована межа\nпам'ять–час виконання",
        },
        {
            "x": 0.34,
            "w": 0.32,
            "color": BLUE,
            "pale": PALE_BLUE,
            "bottleneck": "Досяжність збереженого контексту\nобмежена токенним бюджетом",
            "method": "Ієрархічний контекстний\nвіконний предиктор",
            "evidence": "Підтверджена межа\nдосяжності й точності відтворення",
        },
        {
            "x": 0.70,
            "w": 0.265,
            "color": GOLD,
            "pale": PALE_GOLD,
            "bottleneck": "Рівномірний просторовий розподіл\nне враховує локальну складність",
            "method": "V-BLT + RoPE-ND:\nфіксовані блоки й ентропія",
            "evidence": "Опубліковані оцінки якості й вартості;\nвиміряні карти ентропії",
        },
    ]

    for lane in lanes:
        x = lane["x"]
        width = lane["w"]
        center = x + width / 2
        arrow(ax, (0.50, 0.78), (center, 0.735), color=MUTED, lw=1.4)
        box(ax, x, 0.635, width, 0.10, lane["bottleneck"], fc="white", ec=lane["color"], weight="bold", fs=8.9)
        arrow(ax, (center, 0.635), (center, 0.555), color=lane["color"], lw=1.8)
        box(ax, x, 0.455, width, 0.10, lane["method"], fc=lane["pale"], ec=lane["color"], weight="bold", fs=8.9)
        arrow(ax, (center, 0.455), (center, 0.345), color=lane["color"], lw=1.8)
        box(ax, x, 0.235, width, 0.105, lane["evidence"], fc="white", ec=lane["color"], fs=8.4)

    # A bounded dissertation-only composition was executed after the causal studies.
    for center in (0.1675, 0.50, 0.8325):
        arrow(ax, (center, 0.235), (0.50 if center == 0.50 else 0.27 if center < 0.5 else 0.73, 0.175), color=TEAL, lw=1.5)
    box(
        ax,
        0.20,
        0.075,
        0.60,
        0.10,
        "Обмежена інтеграційна перевірка технічної здійсненності\nсталий грубо-детальний стан + регресійний перехід + реверсивне розгортання",
        fc=PALE_TEAL,
        ec=TEAL,
        fs=8.8,
        weight="bold",
    )

    ax.text(
        0.5,
        0.025,
        "Причинні твердження трьох гілок оцінено окремо; нижній прототип доводить сумісність інтерфейсів, але не спільний причинний ефект.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )
    save_native(
        fig,
        "ch1_fig1_18_systems_map",
        "1.25",
        "Dissertation scope ledger: published component studies, frozen hierarchical-context evidence, and bounded integration feasibility",
    )


def fig4_16_two_resource_axes() -> None:
    """Place context-state and activation-memory evidence side by side only."""

    data = json.loads(E2_JSON.read_text(encoding="utf-8"))
    contrasts = data["frontier"]["contrasts"]
    points = [
        (896, "b896-s0", "o", -13),
        (896, "b896-s1", "s", 13),
        (1152, "b1152-s0", "o", -22),
        (1152, "b1152-s1", "s", 0),
        (1152, "b1152-s2", "^", 22),
        (1792, "b1792-s0", "o", 0),
        (2432, "b2432-30k-s0", "D", -16),
        (2432, "b2432-60k-s0", "P", 16),
        (3072, "b3072-s0", "o", -13),
        (3072, "b3072-s1", "s", 13),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.80, wspace=0.25)
    fig.suptitle("Two measured resource axes", fontsize=15, weight="bold", y=0.955)
    fig.text(
        0.5,
        0.895,
        "Context-state packaging and activation replay were evaluated in separate studies and are not combined into a multiplicative saving claim",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )

    for budget, key, marker, offset in points:
        row = contrasts[key]
        mean, lo, hi = row["mean"], row["lo"], row["hi"]
        color = BLUE if lo > 0 else ORANGE if hi < 0 else MUTED
        ax1.errorbar(
            budget + offset,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt=marker,
            markersize=6.7,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
    ax1.axhline(0, color=INK, linewidth=1.0)
    ax1.axvline(48 * 8 * 8, color=GRID, linestyle="--", linewidth=1.4)
    ax1.text(48 * 8 * 8 - 22, 0.052, "dense 48-frame\nfine history", ha="right", va="top", fontsize=8, color=MUTED)
    ax1.set_xticks([896, 1152, 1792, 2432, 3072])
    ax1.set_xlabel("context tokens")
    ax1.set_ylabel("flat error - hierarchy error")
    ax1.set_title("a  Context/state axis: simultaneous frontier", loc="left", fontsize=11.5)
    ax1.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(900, 0.058, "hierarchy lower error", color=BLUE, fontsize=8.2, weight="bold")
    ax1.text(2300, -0.022, "flat lower error", color=ORANGE, fontsize=8.2, weight="bold")
    marker_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=6, label="seed 0"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=6, label="seed 1"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=6, label="seed 2"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=6, label="30k"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=7, label="60k"),
    ]
    ax1.legend(handles=marker_legend, loc="lower left", ncol=3, frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=0.8)

    horizons = np.array([1, 4, 16])
    positions = np.arange(3)
    memory = {
        "TempVerseFormer": (np.array([0.89, 0.89, 0.89]), PURPLE, "o"),
        "TempFormer": (np.array([0.85, 0.85, 2.61]), BLUE, "s"),
        "Standard Transformer": (np.array([0.85, 0.86, 2.60]), ORANGE, "^"),
        "LSTM": (np.array([1.27, 1.28, 1.28]), GREEN, "D"),
    }
    for label, (values, color, marker) in memory.items():
        ax2.plot(positions, values, marker=marker, markersize=6.5, linewidth=2.0, color=color, label=label)
    ax2.set_xticks(positions, horizons)
    ax2.set_xlabel("prediction horizon (frames)")
    ax2.set_ylabel("temporal-model memory (GB)")
    ax2.set_ylim(0.65, 2.85)
    ax2.set_title("b  Activation axis: published temporal-core memory", loc="left", fontsize=11.5)
    ax2.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.2)
    ax2.annotate(
        "0.89 GB at horizon 16",
        xy=(2, 0.89),
        xytext=(1.15, 0.70),
        arrowprops={"arrowstyle": "->", "color": PURPLE, "linewidth": 1.2},
        fontsize=8.2,
        color=PURPLE,
    )
    ax2.text(0.02, 0.02, "Batch size 32; exact published table values", transform=ax2.transAxes, fontsize=8, color=MUTED)

    fig.text(
        0.5,
        0.055,
        "Left: frozen clip-cluster max-statistic intervals. Right: published temporal-core measurements. No joint factorial experiment was run.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    save_native(
        fig,
        "ch4_fig4_16_two_resource_axes",
        "4.16",
        "artifacts/confirmation_e2/results.json and published TempVerseFormer table values at batch size 32",
    )


def fig4_17_decision_flow() -> None:
    fig, ax = canvas(15.5, 9.2)
    title(ax, "Choose an intervention by diagnosing the binding constraint", "Outputs occupy different interfaces; they do not form one tested architecture")
    box(ax, 0.34, 0.79, 0.32, 0.09, "Which consistency criterion matters?", fc=PALE_BLUE, ec=BLUE, weight="bold", fs=10.5)
    arrow(ax, (0.50, 0.79), (0.50, 0.75), color=INK)
    box(ax, 0.34, 0.66, 0.32, 0.09, "Is relevant history outside\nthe affordable flat window?", fc="white", ec=INK, weight="bold", fs=9.1)
    # Context branch: yes -> choose history form; no -> fidelity question.
    arrow(ax, (0.34, 0.705), (0.22, 0.705), color=GREEN)
    ax.text(0.28, 0.725, "yes", fontsize=8.5, color=GREEN)
    box(ax, 0.035, 0.65, 0.18, 0.13, "Does hidden state change\nbetween sparse anchors?", fc=PALE_TEAL, ec=TEAL, fs=8.6)
    arrow(ax, (0.125, 0.65), (0.125, 0.58), color=TEAL)
    box(ax, 0.015, 0.48, 0.15, 0.10, "spatial hierarchy\n(dense coarse history)", fc=PALE_BLUE, ec=BLUE, fs=8.4)
    box(ax, 0.18, 0.48, 0.15, 0.10, "temporal anchors\n+ recent fine tail", fc=PALE_TEAL, ec=TEAL, fs=8.4)
    arrow(ax, (0.10, 0.58), (0.09, 0.58), color=TEAL)
    arrow(ax, (0.15, 0.58), (0.255, 0.58), color=TEAL)
    ax.text(0.06, 0.595, "yes", fontsize=7.8, color=TEAL)
    ax.text(0.25, 0.595, "no / slow", fontsize=7.8, color=TEAL)

    arrow(ax, (0.66, 0.705), (0.75, 0.705), color=ORANGE)
    ax.text(0.69, 0.725, "no", fontsize=8.5, color=ORANGE)
    box(ax, 0.75, 0.67, 0.22, 0.09, "Is fine identity or detail limiting?", fc="white", ec=ORANGE, weight="bold", fs=9)
    arrow(ax, (0.86, 0.67), (0.86, 0.58), color=ORANGE)
    box(ax, 0.76, 0.48, 0.20, 0.10, "dense or richer\nflat fine context", fc=PALE_ORANGE, ec=ORANGE, weight="bold", fs=8.8)
    ax.text(0.87, 0.60, "yes", fontsize=8, color=ORANGE)
    arrow(ax, (0.78, 0.67), (0.70, 0.615), color=MUTED)
    ax.text(0.745, 0.65, "no", fontsize=8, color=MUTED)
    box(ax, 0.56, 0.54, 0.17, 0.07, "retain least-cost context\nmeeting the criterion", fc=PALE_GRAY, ec=MUTED, fs=7.7)

    # Independent activation-memory lane.
    ax.plot([0.04, 0.96], [0.425, 0.425], color=GRID, linewidth=1.2)
    ax.text(0.04, 0.445, "independent training-memory check", fontsize=9, weight="bold", color=PURPLE)
    box(ax, 0.34, 0.32, 0.32, 0.085, "Does activation storage block\nthe required training horizon?", fc="white", ec=PURPLE, weight="bold", fs=8.7)
    arrow(ax, (0.34, 0.362), (0.25, 0.362), color=PURPLE)
    box(ax, 0.05, 0.28, 0.19, 0.085, "checkpointing", fc=PALE_PURPLE, ec=PURPLE, weight="bold")
    arrow(ax, (0.66, 0.362), (0.75, 0.362), color=PURPLE)
    box(ax, 0.76, 0.28, 0.19, 0.085, "reversible replay\nwhen state boundary permits", fc=PALE_PURPLE, ec=PURPLE, weight="bold", fs=8.7)
    ax.text(0.285, 0.38, "broad blocks", fontsize=7.8, color=PURPLE)
    ax.text(0.69, 0.38, "invertible path", fontsize=7.8, color=PURPLE)

    # Prospective spatial lane.
    ax.plot([0.04, 0.96], [0.235, 0.235], color=GRID, linewidth=1.2)
    ax.text(0.04, 0.252, "spatial-allocation check", fontsize=9, weight="bold", color=GOLD)
    box(ax, 0.34, 0.14, 0.32, 0.08, "Is spatial information demand\nstrongly non-uniform?", fc="white", ec=GOLD, weight="bold", fs=8.7)
    arrow(ax, (0.34, 0.18), (0.25, 0.18), color=GREEN)
    box(ax, 0.05, 0.10, 0.19, 0.08, "fixed spatial allocation", fc=PALE_GREEN, ec=GREEN, fs=8.8)
    arrow(ax, (0.66, 0.18), (0.75, 0.18), color=ORANGE, dashed=True, lw=2)
    box(
        ax,
        0.75,
        0.075,
        0.22,
        0.13,
        "candidate entropy-informed\nadaptive policy\nend-to-end dynamic benefit\nnot yet measured",
        fc="white",
        ec=ORANGE,
        dashed=True,
        fs=8.1,
    )
    ax.text(0.285, 0.195, "no", fontsize=7.8, color=GREEN)
    ax.text(0.69, 0.195, "yes / proposed", fontsize=7.8, color=ORANGE)
    ax.text(0.5, 0.025, "Solid arrows: executed evidence or established comparator. Dashed orange arrow: proposed policy requiring a new end-to-end test.", ha="center", fontsize=9, color=MUTED)
    save_native(fig, "ch4_fig4_17_decision_flow", "4.17", "Chapter 4 operating-region evidence and Chapter 2 scope boundaries")


NATIVE_STEMS = [
    "ch1_fig1_18_systems_map",
    "ch2_fig2_3_dataset_ladder",
    "ch2_fig2_4_storage_schedules",
    "ch2_fig2_20_fixed_patch_flow",
    "ch2_fig2_24_entropy_pipeline",
    "ch2_fig2_26_vblt_point_estimates",
    "ch2_fig2_27_measured_vs_proposed",
    "ch3_fig3_1_reentry_memory",
    "ch3_fig3_3_reach_fidelity_axes",
    "ch3_fig3_4_codec_dimensions",
    "ch3_fig3_6_joint_attention",
    "ch3_fig3_7_gradient_paths",
    "ch3_fig3_8_objective_interfaces",
    "ch3_fig3_10_ddim_timeline",
    "ch3_fig3_12_anchor_geometry",
    "ch3_fig3_13_token_costs",
    "ch3_fig3_15_reentry_storyboard",
    "ch3_fig3_16_reach_population",
    "ch3_fig3_17_persistent_assignment",
    "ch4_fig4_4_ebt_composite",
    "ch4_fig4_6_reach_effects",
    "ch4_fig4_8_equal_reach",
    "ch4_fig4_16_two_resource_axes",
    "ch4_fig4_17_decision_flow",
]


def validate_native_pairs() -> None:
    problems: list[str] = []
    for stem in NATIVE_STEMS:
        png = OUT / f"{stem}.png"
        pdf = OUT / f"{stem}.pdf"
        if not png.exists() or not pdf.exists():
            problems.append(f"missing PNG/PDF pair for {stem}")
            continue
        image = Image.open(png).convert("RGB")
        if image.width < 1200 or image.height < 650:
            problems.append(f"low raster dimensions for {stem}: {image.size}")
        corners = [image.getpixel((0, 0)), image.getpixel((image.width - 1, 0)), image.getpixel((0, image.height - 1)), image.getpixel((image.width - 1, image.height - 1))]
        if any(min(pixel) < 245 for pixel in corners):
            problems.append(f"non-white raster corner for {stem}: {corners}")
        document = fitz.open(pdf)
        if len(document) != 1 or document[0].rect.width <= 0 or document[0].rect.height <= 0:
            problems.append(f"invalid one-page vector PDF for {stem}")
        document.close()
    if problems:
        raise RuntimeError("native figure validation failed:\n" + "\n".join(problems))


def main() -> None:
    PROVENANCE.clear()
    write_published_assets()
    fig1_18_systems_map()
    fig2_3_dataset_ladder()
    fig2_4_storage_schedules()
    fig2_20_fixed_patch_flow()
    fig2_24_entropy_pipeline()
    fig2_26_vblt_point_estimates()
    fig2_27_measured_vs_proposed()
    fig3_1_reentry_memory()
    fig3_3_reach_fidelity_axes()
    fig3_4_codec_dimensions()
    fig3_6_joint_attention()
    fig3_7_gradient_paths()
    fig3_8_objective_interfaces()
    fig3_10_ddim_timeline()
    fig3_12_anchor_geometry()
    fig3_13_token_costs()
    fig3_15_reentry_storyboard()
    fig3_16_reach_population()
    fig3_17_persistent_assignment()
    fig4_4_ebt_composite()
    fig4_6_reach_effects()
    fig4_8_equal_reach()
    fig4_16_two_resource_axes()
    fig4_17_decision_flow()
    validate_native_pairs()
    ledger = OUT / "ch2_ch4_figure_provenance.json"
    ledger.write_text(json.dumps(sorted(PROVENANCE, key=lambda row: (row["figure_id"], row["file"])), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(PROVENANCE)} provenance records and {len(NATIVE_STEMS)} native PNG/PDF pairs")


if __name__ == "__main__":
    main()
