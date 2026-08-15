"""Build polished DOCX review editions from the assembled Markdown manuscripts.

The design resolves the ``narrative_proposal`` preset with an ``editorial_cover`` opening.
LaTeX expressions are converted to native Word equations (OMML), so both the editable DOCX
and the Word-exported PDF contain typeset mathematics rather than literal source commands.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_MD = ROOT / "docs" / "article" / "final" / "ARTICLE_REACH_FIDELITY_TRUST.md"
DISSERTATION_MD = ROOT / "docs" / "dissertation" / "final" / "DISSERTATION_MANUSCRIPT.md"

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
MUTED = "687386"
LIGHT = "F4F6F9"
TABLE_HEADER = "F4F6F9"
CAUTION = "FFF3CD"
PAGE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MATHML_TO_OMML: etree.XSLT | None = None


def _font(run, name: str = "Calibri", size: float | None = None, **kwargs) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    for key, value in kwargs.items():
        if key == "color":
            run.font.color.rgb = value
        elif key in {"bold", "italic", "underline"}:
            setattr(run, key, value)
        else:
            setattr(run.font, key, value)


def _set_cell_margins(table) -> None:
    tbl_pr = table._tbl.tblPr
    margins = tbl_pr.find(qn("w:tblCellMar"))
    if margins is None:
        margins = OxmlElement("w:tblCellMar")
        tbl_pr.append(margins)
    for edge, value in (("top", 80), ("bottom", 80), ("start", 120), ("end", 120)):
        node = margins.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _set_table_geometry(table, widths: list[int]) -> None:
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    for tag, value in (("tblW", sum(widths)), ("tblInd", TABLE_INDENT_DXA)):
        node = tbl_pr.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tbl_pr.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
    _set_cell_margins(table)


def _shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def _new_numbering_id(doc: Document, start: int = 1) -> int:
    """Create a real Word numbering instance that restarts at ``start``."""
    numbering = doc.part.numbering_part.element
    style = doc.styles["List Number"]
    base_num_id = int(style.element.pPr.numPr.numId.val)
    base_num = next(
        node for node in numbering.findall(qn("w:num"))
        if int(node.get(qn("w:numId"))) == base_num_id
    )
    abstract_num_id = base_num.find(qn("w:abstractNumId")).get(qn("w:val"))
    new_id = max(int(node.get(qn("w:numId"))) for node in numbering.findall(qn("w:num"))) + 1
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(new_id))
    abstract = OxmlElement("w:abstractNumId")
    abstract.set(qn("w:val"), abstract_num_id)
    num.append(abstract)
    override = OxmlElement("w:lvlOverride")
    override.set(qn("w:ilvl"), "0")
    start_override = OxmlElement("w:startOverride")
    start_override.set(qn("w:val"), str(start))
    override.append(start_override)
    num.append(override)
    numbering.append(num)
    return new_id


def _apply_numbering(paragraph, num_id: int, level: int = 0) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        p_pr.append(num_pr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), str(level))
    number = OxmlElement("w:numId")
    number.set(qn("w:val"), str(num_id))
    num_pr.extend([ilvl, number])


def _column_widths(rows: list[list[str]]) -> list[int]:
    columns = max(len(row) for row in rows)
    weights = []
    for index in range(columns):
        values = [row[index] if index < len(row) else "" for row in rows]
        longest = max((min(len(value), 80) for value in values), default=1)
        weights.append(max(8, longest))
    total = sum(weights)
    widths = [max(800, round(PAGE_WIDTH_DXA * weight / total)) for weight in weights]
    difference = PAGE_WIDTH_DXA - sum(widths)
    widths[-1] += difference
    if widths[-1] < 800:
        deficit = 800 - widths[-1]
        widest = max(range(len(widths) - 1), key=widths.__getitem__)
        widths[widest] -= deficit
        widths[-1] = 800
    return widths


def _add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    _font(run, size=9)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def _configure_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.333
    normal.paragraph_format.widow_control = True

    tokens = {
        "Heading 1": (16, BLUE, 18, 10),
        "Heading 2": (13, BLUE, 12, 6),
        "Heading 3": (12, DARK_BLUE, 8, 4),
    }
    for name, (size, color, before, after) in tokens.items():
        style = doc.styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.widow_control = True

    for name in ("List Bullet", "List Number"):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(11)
        style.paragraph_format.left_indent = Inches(0.375)
        style.paragraph_format.first_line_indent = Inches(-0.194)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.208

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run("HIERARCHICAL VIDEO PREDICTION · REVIEW EDITION")
    _font(run, size=8.5, color=RGBColor.from_string(MUTED))
    footer = section.footer.paragraphs[0]
    _add_page_number(footer)


INLINE = re.compile(
    r"(\\\(.*?\\\)|(?<!\$)\$(?!\$)[^$]+(?<!\$)\$(?!\$)|"
    r"\*\*.*?\*\*|`.*?`|\*[^*].*?\*|\[[^\]]+\]\([^)]+\))"
)


def _mml2omml_stylesheet() -> Path:
    override = os.environ.get("MML2OMML_XSL")
    if override:
        path = Path(override)
        if path.is_file():
            return path
        raise FileNotFoundError(f"MML2OMML_XSL does not point to a file: {path}")

    candidates: list[Path] = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if not root:
            continue
        office = Path(root) / "Microsoft Office"
        candidates.extend(
            office / "root" / f"Office{version}" / "MML2OMML.XSL"
            for version in (16, 15, 14)
        )
        candidates.extend(
            office / f"Office{version}" / "MML2OMML.XSL"
            for version in (16, 15, 14)
        )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Microsoft Office's MML2OMML.XSL was not found. "
        "Set MML2OMML_XSL to its absolute path."
    )


def _mathml_to_omml_transform() -> etree.XSLT:
    global _MATHML_TO_OMML
    if _MATHML_TO_OMML is None:
        _MATHML_TO_OMML = etree.XSLT(etree.parse(str(_mml2omml_stylesheet())))
    return _MATHML_TO_OMML


def _normalize_latex(source: str) -> str:
    """Normalize a few TeX spellings that MathML/OMML otherwise leaves literal."""
    normalized = source.strip()
    replacements = (
        (r"\bigl\lVert", r"\left\Vert "),
        (r"\bigr\rVert", r"\right\Vert"),
        (r"\bigl\|", r"\left\Vert "),
        (r"\bigr\|", r"\right\Vert"),
        (r"\left\lVert", r"\left\Vert "),
        (r"\right\rVert", r"\right\Vert"),
        (r"\left\|", r"\left\Vert "),
        (r"\right\|", r"\right\Vert"),
        (r"\lVert", r"\Vert "),
        (r"\rVert", r"\Vert"),
        (r"\ ", r"\,"),
    )
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    normalized = re.sub(
        r"\\arg\s*\\min",
        r"\\operatorname{arg\\,min}",
        normalized,
    )
    return normalized


def _split_equation_tag(source: str) -> tuple[str, str | None]:
    match = re.search(r"\\tag\{([^{}]+)\}", source)
    if match is None:
        return source, None
    equation = (source[: match.start()] + source[match.end() :]).strip()
    return equation, match.group(1).strip()


def _omml_element(
    source: str,
    *,
    size: float | None = None,
    bold: bool = False,
) -> etree._Element:
    mathml = latex_to_mathml(_normalize_latex(source))
    # Some aligned LaTeX environments are emitted with a literal alignment ampersand by
    # latex2mathml. Escape only ampersands that are not already XML entities before parsing.
    mathml = re.sub(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z]+;)", "&amp;", mathml)
    mathml_root = etree.fromstring(mathml.encode("utf-8"))
    omml = _mathml_to_omml_transform()(mathml_root).getroot()
    for math_run in omml.iter(f"{{{MATH_NS}}}r"):
        run_properties = math_run.find(f"{{{WORD_NS}}}rPr")
        if run_properties is None:
            run_properties = etree.Element(f"{{{WORD_NS}}}rPr")
            math_run.insert(0, run_properties)
        fonts = etree.SubElement(run_properties, f"{{{WORD_NS}}}rFonts")
        fonts.set(f"{{{WORD_NS}}}ascii", "Cambria Math")
        fonts.set(f"{{{WORD_NS}}}hAnsi", "Cambria Math")
        if size is not None:
            half_points = str(round(size * 2))
            etree.SubElement(
                run_properties,
                f"{{{WORD_NS}}}sz",
            ).set(f"{{{WORD_NS}}}val", half_points)
            etree.SubElement(
                run_properties,
                f"{{{WORD_NS}}}szCs",
            ).set(f"{{{WORD_NS}}}val", half_points)
        if bold:
            etree.SubElement(run_properties, f"{{{WORD_NS}}}b")
    return omml


def _append_math(
    paragraph,
    source: str,
    *,
    size: float | None = None,
    bold: bool = False,
) -> None:
    paragraph._p.append(_omml_element(source, size=size, bold=bold))


def _add_display_equation(doc: Document, source: str) -> None:
    equation, tag = _split_equation_tag(source)
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.keep_together = True
    if tag is None:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _append_math(paragraph, equation, size=10.5)
        return

    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.tab_stops.add_tab_stop(
        Inches(3.25),
        WD_TAB_ALIGNMENT.CENTER,
    )
    paragraph.paragraph_format.tab_stops.add_tab_stop(
        Inches(6.5),
        WD_TAB_ALIGNMENT.RIGHT,
    )
    paragraph.add_run("\t")
    _append_math(paragraph, equation, size=10.5)
    paragraph.add_run("\t")
    _font(paragraph.add_run(f"({tag})"), name="Cambria Math", size=10.5)


def _add_inline(
    paragraph,
    text: str,
    *,
    size: float | None = None,
    bold: bool = False,
    italic: bool = False,
) -> None:
    text = html.unescape(text)
    cursor = 0
    for match in INLINE.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor : match.start()])
            _font(run, size=size, bold=bold, italic=italic)
        token = match.group(0)
        if token.startswith(r"\("):
            _append_math(paragraph, token[2:-2], size=size, bold=bold)
        elif token.startswith("$"):
            _append_math(paragraph, token[1:-1], size=size, bold=bold)
        elif token.startswith("**"):
            _add_inline(
                paragraph,
                token[2:-2],
                size=size,
                bold=True,
                italic=italic,
            )
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            _font(
                run,
                name="Consolas",
                size=(size or 10),
                bold=bold,
                italic=italic,
            )
        elif token.startswith("*"):
            _add_inline(
                paragraph,
                token[1:-1],
                size=size,
                bold=bold,
                italic=True,
            )
        else:
            link = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token)
            run = paragraph.add_run(link.group(1) if link else token)
            _font(
                run,
                size=size,
                bold=bold,
                italic=italic,
                color=RGBColor.from_string(BLUE),
            )
            run.underline = True
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        _font(run, size=size, bold=bold, italic=italic)
    if "[AUTHOR:" in text:
        p_pr = paragraph._p.get_or_add_pPr()
        shd = p_pr.find(qn("w:shd"))
        if shd is None:
            shd = OxmlElement("w:shd")
            p_pr.append(shd)
        shd.set(qn("w:fill"), CAUTION)


def _parse_table(lines: list[str]) -> list[list[str]]:
    result = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        result.append(cells)
    if len(result) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in result[1]):
        del result[1]
    return result


def _add_table(doc: Document, rows: list[list[str]]) -> None:
    columns = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=columns)
    table.style = "Table Grid"
    _set_table_geometry(table, _column_widths(rows))
    for ri, values in enumerate(rows):
        for ci in range(columns):
            cell = table.cell(ri, ci)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.08
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER
                if ci and all(len(row[ci]) < 24 for row in rows if ci < len(row))
                else WD_ALIGN_PARAGRAPH.LEFT
            )
            _add_inline(paragraph, values[ci] if ci < len(values) else "", size=9)
            if ri == 0:
                for run in paragraph.runs:
                    run.bold = True
                _shade_cell(cell, TABLE_HEADER)
    _repeat_header(table.rows[0])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _add_image(doc: Document, source: Path, alt: str) -> None:
    from PIL import Image

    with Image.open(source) as image:
        width, height = image.size
    max_width, max_height = 6.25, 7.2
    display_width = min(max_width, max_height * width / height)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    run.add_picture(str(source), width=Inches(display_width))
    doc_pr = run._r.xpath(".//wp:docPr")
    if doc_pr:
        doc_pr[0].set("descr", alt)
    paragraph.paragraph_format.space_after = Pt(4)


def _add_cover(doc: Document, *, kicker: str, title: str, subtitle: str, author: str) -> None:
    for _ in range(4):
        doc.add_paragraph()
    kicker_p = doc.add_paragraph()
    kicker_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker_p.paragraph_format.space_after = Pt(18)
    _font(kicker_p.add_run(kicker.upper()), size=10, bold=True,
          color=RGBColor.from_string(BLUE))
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(10)
    _font(title_p.add_run(title), size=27, bold=True, color=RGBColor.from_string(DARK_BLUE))
    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_p.paragraph_format.space_after = Pt(56)
    _font(subtitle_p.add_run(subtitle), size=14, color=RGBColor.from_string(MUTED))
    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(author_p.add_run(author), size=11, bold=True)
    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(date_p.add_run("English review edition · 3 August 2026"), size=9.5,
          color=RGBColor.from_string(MUTED))
    doc.add_page_break()


def _markdown_to_docx(markdown_path: Path, output: Path, *, cover: dict[str, str]) -> None:
    doc = Document()
    _configure_styles(doc)
    _add_cover(doc, **cover)
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []
    ordered_num_id: int | None = None
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            ordered_num_id = None
            if in_code:
                paragraph = doc.add_paragraph()
                paragraph.paragraph_format.left_indent = Inches(0.25)
                paragraph.paragraph_format.space_after = Pt(8)
                p_pr = paragraph._p.get_or_add_pPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), LIGHT)
                p_pr.append(shd)
                _font(paragraph.add_run("\n".join(code_lines)), name="Consolas", size=8.5)
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if stripped.startswith("$$"):
            ordered_num_id = None
            equation = stripped[2:]
            while not equation.rstrip().endswith("$$") and index + 1 < len(lines):
                index += 1
                equation += " " + lines[index].strip()
            equation = equation.removesuffix("$$").strip()
            _add_display_equation(doc, equation)
            index += 1
            continue

        if re.search(r"page-break-after", stripped):
            ordered_num_id = None
            doc.add_page_break()
            index += 1
            continue
        if stripped.startswith("<!--"):
            ordered_num_id = None
            while "-->" not in stripped and index + 1 < len(lines):
                index += 1
                stripped = lines[index].strip()
            index += 1
            continue
        if not stripped or stripped == "---":
            index += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            ordered_num_id = None
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            _add_table(doc, _parse_table(table_lines))
            continue

        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            ordered_num_id = None
            target = (markdown_path.parent / image.group(2)).resolve()
            _add_image(doc, target, image.group(1))
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            ordered_num_id = None
            level = len(heading.group(1))
            paragraph = doc.add_paragraph(style=f"Heading {level}")
            _add_inline(paragraph, heading.group(2))
            index += 1
            continue

        item = re.match(r"^(\s*)([-+*]|\d+\.)\s+(.+)$", line)
        if item:
            ordered = item.group(2)[0].isdigit()
            style = "List Number" if ordered else "List Bullet"
            paragraph = doc.add_paragraph(style=style)
            depth = min(len(item.group(1)) // 2, 2)
            paragraph.paragraph_format.left_indent = Inches(0.375 + 0.25 * depth)
            paragraph.paragraph_format.first_line_indent = Inches(-0.194)
            if ordered:
                if ordered_num_id is None:
                    ordered_num_id = _new_numbering_id(doc, int(item.group(2).rstrip(".")))
                _apply_numbering(paragraph, ordered_num_id, depth)
            else:
                ordered_num_id = None
            index += 1
            parts = [item.group(3)]
            base_indent = len(item.group(1))
            while index < len(lines):
                continuation = lines[index]
                continuation_stripped = continuation.strip()
                continuation_indent = len(continuation) - len(continuation.lstrip())
                if (
                    not continuation_stripped
                    or continuation_indent <= base_indent
                    or continuation_stripped.startswith(("#", "|", "```", "$$", ">", "![", "<div"))
                    or re.match(r"^(\s*)([-+*]|\d+\.)\s+", continuation)
                ):
                    break
                parts.append(continuation_stripped)
                index += 1
            _add_inline(paragraph, " ".join(parts))
            continue

        if stripped.startswith(">"):
            ordered_num_id = None
            quote = stripped.lstrip("> ")
            while quote.count("$") % 2 and index + 1 < len(lines):
                candidate = lines[index + 1].strip()
                if not candidate.startswith(">"):
                    break
                index += 1
                quote += " " + candidate.lstrip("> ")
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.3)
            paragraph.paragraph_format.right_indent = Inches(0.2)
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(8)
            p_pr = paragraph._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:fill"), LIGHT)
            p_pr.append(shd)
            _add_inline(paragraph, quote)
            index += 1
            continue

        # Fold wrapped Markdown lines into one paragraph until the next structural boundary.
        ordered_num_id = None
        parts = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate == "---"
                or candidate.startswith(("#", "|", "```", "$$", ">", "![", "<div"))
                or re.match(r"^(\s*)([-+*]|\d+\.)\s+", lines[index])
            ):
                break
            parts.append(candidate)
            index += 1
        paragraph = doc.add_paragraph()
        _add_inline(paragraph, " ".join(parts))

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.author = "Leonid Kulyk"
    doc.core_properties.title = cover["title"]
    doc.core_properties.subject = cover["subtitle"]
    doc.core_properties.keywords = "hierarchical video prediction, context packaging, reach, fidelity"
    doc.save(output)
    print(f"WROTE {output}")


def main() -> None:
    _markdown_to_docx(
        ARTICLE_MD,
        ARTICLE_MD.with_suffix(".docx"),
        cover={
            "kicker": "Research Article",
            "title": "When Does Hierarchical Video Memory Pay?",
            "subtitle": "Reach–Fidelity Trade-offs under Fixed Context Budgets",
            "author": "Leonid Kulyk",
        },
    )
    _markdown_to_docx(
        DISSERTATION_MD,
        DISSERTATION_MD.with_name("DISSERTATION_REVIEW_MANUSCRIPT.docx"),
        cover={
            "kicker": "Dissertation Manuscript",
            "title": "Systems Analysis and Intelligent Technologies for Consistent Modeling of "
                     "Hierarchical Video Data Generation Processes",
            "subtitle": "Reach-efficient context memory and activation-efficient temporal optimization",
            "author": "Leonid Ruslanovych Kulyk · VNTU · 2026",
        },
    )


if __name__ == "__main__":
    main()
