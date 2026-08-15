"""Distill the retained dissertation DOCX into a reproducible template contract.

The script is read-only with respect to the reference document.  It records the
page system, styles used by semantic paragraph roles, document-part inventory,
and front-matter flow needed to build the condensed English working copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emu_inches(value) -> float | None:
    return None if value is None else round(value.inches, 3)


def points(value) -> float | None:
    return None if value is None else round(value.pt, 2)


def color(run) -> str | None:
    value = run.font.color.rgb
    return None if value is None else str(value)


def paragraph_signature(paragraph) -> dict:
    fmt = paragraph.paragraph_format
    run = next((r for r in paragraph.runs if r.text.strip()), None)
    style_font = paragraph.style.font if paragraph.style is not None else None
    font_name = None
    font_size = None
    bold = None
    italic = None
    font_color = None
    if run is not None:
        font_name = run.font.name
        if run._element.rPr is not None and run._element.rPr.rFonts is not None:
            font_name = font_name or run._element.rPr.rFonts.get(qn("w:ascii"))
        font_size = points(run.font.size)
        bold = run.bold
        italic = run.italic
        font_color = color(run)
    if style_font is not None:
        font_name = font_name or style_font.name
        font_size = font_size or points(style_font.size)
        bold = style_font.bold if bold is None else bold
        italic = style_font.italic if italic is None else italic
    return {
        "style": paragraph.style.name if paragraph.style else None,
        "font": font_name,
        "font_size_pt": font_size,
        "bold": bold,
        "italic": italic,
        "color": font_color,
        "alignment": str(paragraph.alignment or fmt.alignment),
        "first_line_indent_in": emu_inches(fmt.first_line_indent),
        "left_indent_in": emu_inches(fmt.left_indent),
        "right_indent_in": emu_inches(fmt.right_indent),
        "space_before_pt": points(fmt.space_before),
        "space_after_pt": points(fmt.space_after),
        "line_spacing": fmt.line_spacing,
        "keep_with_next": fmt.keep_with_next,
        "keep_together": fmt.keep_together,
        "page_break_before": fmt.page_break_before,
    }


def classify(text: str, style_name: str) -> str:
    stripped = text.strip()
    if style_name.startswith("Heading"):
        return style_name.lower().replace(" ", "_")
    if re.fullmatch(r"(?:РОЗДІЛ|CHAPTER)\s+\d+", stripped, re.I):
        return "chapter_label"
    if stripped in {"АНОТАЦІЯ", "АННОТАЦІЯ", "ABSTRACT", "ВСТУП", "ЗМІСТ"}:
        return "front_heading"
    if stripped.startswith(("Рисунок ", "Figure ", "Таблиця ", "Table ")):
        return "caption"
    if len(stripped) > 30 and stripped == stripped.upper():
        return "uppercase_heading"
    return "body"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    reference = args.reference.resolve()
    document = Document(reference)
    role_samples: dict[str, list[dict]] = defaultdict(list)
    style_counts = Counter()
    nonempty = 0
    for index, paragraph in enumerate(document.paragraphs):
        if not paragraph.text.strip():
            continue
        nonempty += 1
        style_name = paragraph.style.name if paragraph.style else ""
        style_counts[style_name] += 1
        role = classify(paragraph.text, style_name)
        if len(role_samples[role]) < 8:
            role_samples[role].append(
                {
                    "paragraph_index": index,
                    "text": paragraph.text.strip()[:240],
                    **paragraph_signature(paragraph),
                }
            )

    sections = []
    for section in document.sections:
        sections.append(
            {
                "page_width_in": emu_inches(section.page_width),
                "page_height_in": emu_inches(section.page_height),
                "orientation": str(section.orientation),
                "left_margin_in": emu_inches(section.left_margin),
                "right_margin_in": emu_inches(section.right_margin),
                "top_margin_in": emu_inches(section.top_margin),
                "bottom_margin_in": emu_inches(section.bottom_margin),
                "header_distance_in": emu_inches(section.header_distance),
                "footer_distance_in": emu_inches(section.footer_distance),
                "different_first_page": section.different_first_page_header_footer,
                "header_linked": section.header.is_linked_to_previous,
                "footer_linked": section.footer.is_linked_to_previous,
            }
        )

    with zipfile.ZipFile(reference) as archive:
        parts = [
            {
                "part": info.filename,
                "bytes": info.file_size,
                "sha256": hashlib.sha256(archive.read(info.filename)).hexdigest(),
            }
            for info in archive.infolist()
        ]

    payload = {
        "reference": str(reference),
        "reference_sha256": sha256(reference),
        "paragraphs": len(document.paragraphs),
        "nonempty_paragraphs": nonempty,
        "tables": len(document.tables),
        "inline_shapes": len(document.inline_shapes),
        "sections": sections,
        "style_counts": style_counts,
        "role_samples": role_samples,
        "package_parts": parts,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("reference", "reference_sha256", "paragraphs", "tables", "inline_shapes", "sections")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
