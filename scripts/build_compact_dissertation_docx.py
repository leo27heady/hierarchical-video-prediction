#!/usr/bin/env python3
"""Build the compact English dissertation in the author's institutional Word template."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

import build_final_documents as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "dissertation" / "condensed_english"
REFERENCE = ROOT / "tmp" / "dissertation_compaction" / "LKulyk_dissertation_demo_reference.docx"
OUTPUT = ROOT / "docs" / "dissertation" / "final" / "LKulyk_dissertation_condensed_english.docx"
FILES = [SOURCE / f"0{i}_{name}.md" for i, name in enumerate((
    "FRONT_MATTER", "INTRODUCTION", "CHAPTER_1", "CHAPTER_2", "CHAPTER_3",
    "CHAPTER_4", "CONCLUSIONS", "REFERENCES", "APPENDICES"
))]


def set_font(style, name: str, size: float, *, bold: bool | None = None) -> None:
    style.font.name = name
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold


def clear_body(doc: Document) -> None:
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def configure_template(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.268)
    section.page_height = Inches(11.693)
    section.left_margin = Inches(0.985)
    section.right_margin = Inches(0.492)
    section.top_margin = Inches(0.787)
    section.bottom_margin = Inches(0.787)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    set_font(normal, "Times New Roman", 14)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Cm(1.25)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.widow_control = True

    h1 = doc.styles["Heading 1"]
    set_font(h1, "Times New Roman", 14, bold=True)
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.first_line_indent = Cm(0)
    h1.paragraph_format.space_before = Pt(0)
    h1.paragraph_format.space_after = Pt(0)
    h1.paragraph_format.line_spacing = 1.5
    h1.paragraph_format.page_break_before = True
    h1.paragraph_format.keep_with_next = True

    for name in ("Heading 2", "Heading 3"):
        style = doc.styles[name]
        set_font(style, "Times New Roman", 14, bold=True)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style.paragraph_format.first_line_indent = Cm(1.25 if name == "Heading 3" else 0)
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        try:
            style = doc.styles[name]
        except KeyError:
            style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style.base_style = normal
        set_font(style, "Times New Roman", 14)
        style.paragraph_format.left_indent = Cm(1.25)
        style.paragraph_format.first_line_indent = Cm(-0.63)
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.line_spacing = 1.5

    # Preserve the template's top-right page numbering but remove historical running text.
    header = section.header
    for paragraph in header.paragraphs:
        paragraph.clear()
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    base._add_page_number(hp)
    for run in hp.runs:
        base._font(run, name="Times New Roman", size=14)
    for paragraph in section.footer.paragraphs:
        paragraph.clear()


def add_plain(doc: Document, text: str = "", *, align=None, bold=False, indent=True, spacing=1.5):
    p = doc.add_paragraph()
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(1.25 if indent else 0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = spacing
    if text:
        base._add_inline(p, text, size=14, bold=bold)
    return p


def add_title_page(doc: Document) -> None:
    centered = lambda text="", bold=False: add_plain(
        doc, text, align=WD_ALIGN_PARAGRAPH.CENTER, bold=bold, indent=False
    )
    centered("Vinnytsia National Technical University")
    centered("Ministry of Education and Science of Ukraine")
    for _ in range(2):
        centered()
    p = add_plain(doc, "Qualifying scientific work submitted as a manuscript", align=WD_ALIGN_PARAGRAPH.LEFT, indent=False)
    p.paragraph_format.left_indent = Cm(11.4)
    centered()
    centered("LEONID RUSLANOVYCH KULYK", bold=True)
    p = add_plain(doc, "UDC 004.054:[004.032.26+004.85]", indent=False)
    p.paragraph_format.left_indent = Cm(10.0)
    centered()
    centered("DISSERTATION", bold=True)
    centered("SYSTEMS ANALYSIS AND INTELLIGENT TECHNOLOGIES FOR CONSISTENT MODELLING OF HIERARCHICAL VIDEO-DATA GENERATION PROCESSES", bold=True)
    centered()
    centered("Specialty 124 — System Analysis")
    centered("Field of knowledge 12 — Information Technologies")
    centered()
    centered("Submitted for the degree of Doctor of Philosophy")
    centered()
    p = add_plain(doc, "Academic supervisor: Oleksandr Borysovych Mokin, Doctor of Technical Sciences, Professor", align=WD_ALIGN_PARAGRAPH.LEFT, indent=False)
    p.paragraph_format.left_indent = Cm(10.0)
    for _ in range(2):
        centered()
    centered("Vinnytsia — 2026")
    doc.add_page_break()


def add_front_heading(doc: Document, text: str, *, page_break: bool = False) -> None:
    if page_break:
        doc.add_page_break()
    p = add_plain(doc, text.upper(), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    p.paragraph_format.keep_with_next = True


def add_toc(doc: Document) -> None:
    add_front_heading(doc, "CONTENTS", page_break=True)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Update field in Microsoft Word to generate the table of contents."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, placeholder, end])


def split_front(text: str) -> dict[str, str]:
    parts = re.split(r"(?m)^##\s+", text)
    result: dict[str, str] = {}
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        result[title.strip()] = body.strip()
    return result


def markdown_lines(doc: Document, text: str, source_path: Path, *, skip_h1: bool = False) -> None:
    lines = text.splitlines()
    index = 0
    ordered_num_id = None
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped == "---":
            index += 1
            continue
        if stripped.startswith("<!--"):
            while "-->" not in stripped and index + 1 < len(lines):
                index += 1
                stripped = lines[index].strip()
            index += 1
            continue
        if stripped.startswith("$$"):
            equation = stripped[2:]
            while not equation.rstrip().endswith("$$") and index + 1 < len(lines):
                index += 1
                equation += " " + lines[index].strip()
            equation = equation.removesuffix("$$").strip()
            try:
                base._add_display_equation(doc, equation)
            except Exception as exc:
                raise RuntimeError(f"Equation conversion failed in {source_path.name}: {equation}") from exc
            p = doc.paragraphs[-1]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            for omath in p._p.xpath(".//m:oMath | .//m:oMathPara"):
                pass
            index += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = base._parse_table(table_lines)
            base._add_table(doc, rows)
            table = doc.tables[-1]
            for row in table.rows:
                for cell in row.cells:
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    for p in cell.paragraphs:
                        p.paragraph_format.line_spacing = 1.0
                        p.paragraph_format.space_after = Pt(0)
                        for run in p.runs:
                            base._font(run, name="Times New Roman", size=10.5)
            continue
        image = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image:
            target = (source_path.parent / image.group(2)).resolve()
            base._add_image(doc, target, image.group(1))
            doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            if skip_h1 and level == 1:
                index += 1
                continue
            title = heading.group(2)
            if level == 1:
                title = title.replace(" — ", "\n").upper()
            p = doc.add_paragraph(style=f"Heading {level}")
            base._add_inline(p, title, size=14, bold=(level <= 2))
            index += 1
            continue
        if stripped.startswith("**Figure "):
            p = add_plain(doc, stripped, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, spacing=1.0)
            p.paragraph_format.keep_together = True
            index += 1
            continue
        if stripped.startswith("**Table "):
            p = add_plain(doc, stripped, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, spacing=1.0)
            p.paragraph_format.keep_with_next = True
            index += 1
            continue
        item = re.match(r"^(\s*)([-+*]|\d+\.)\s+(.+)$", line)
        if item:
            ordered = item.group(2)[0].isdigit()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1.25)
            p.paragraph_format.first_line_indent = Cm(-0.63)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.5
            parts = [item.group(3)]
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate or re.match(r"^(\s*)([-+*]|\d+\.)\s+", lines[index]) or candidate.startswith(("#", "|", "![", "$$")):
                    break
                parts.append(candidate)
                index += 1
            prefix = item.group(2) if ordered else "•"
            base._add_inline(p, prefix + " " + " ".join(parts), size=14)
            continue

        parts = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (not candidate or candidate == "---" or candidate.startswith(("#", "|", "![", "$$", "<!--"))
                    or re.match(r"^(\s*)([-+*]|\d+\.)\s+", lines[index])):
                break
            parts.append(candidate)
            index += 1
        p = add_plain(doc, " ".join(parts))
        ordered_num_id = None


def build() -> None:
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)
    doc = Document(REFERENCE)
    clear_body(doc)
    configure_template(doc)
    add_title_page(doc)

    front = split_front(FILES[0].read_text(encoding="utf-8"))
    add_front_heading(doc, "АНОТАЦІЯ")
    markdown_lines(doc, front["Ukrainian annotation"], FILES[0], skip_h1=True)
    add_front_heading(doc, "ABSTRACT", page_break=True)
    markdown_lines(doc, front["English abstract"], FILES[0], skip_h1=True)
    add_front_heading(doc, "LIST OF APPLICANT PUBLICATIONS ON THE DISSERTATION TOPIC", page_break=True)
    markdown_lines(doc, front["Applicant publications"], FILES[0], skip_h1=True)
    add_toc(doc)
    add_front_heading(doc, "LIST OF SYMBOLS AND ABBREVIATIONS", page_break=True)
    markdown_lines(doc, front["Symbols and abbreviations"], FILES[0], skip_h1=True)

    for path in FILES[1:]:
        markdown_lines(doc, path.read_text(encoding="utf-8"), path)

    settings = doc.settings.element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")
    doc.core_properties.author = "Leonid Ruslanovych Kulyk"
    doc.core_properties.title = "Systems Analysis and Intelligent Technologies for Consistent Modelling of Hierarchical Video-Data Generation Processes"
    doc.core_properties.subject = "Compact English dissertation working edition"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
