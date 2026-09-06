"""Build current Ukrainian sources with the repository's institutional geometry.

Intermediate DOCX is exported by Word to PDF. No English assembled manuscript
is substituted; source hashes and actual source paths are recorded.
"""
from pathlib import Path
import hashlib
import json
import re
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
import cairosvg
import build_compact_dissertation_docx as compact
import build_final_documents as base

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/dissertation/condensed_translated_humanized_ukrainian'
OUTPUT=ROOT/'output/pdf/DISSERTATION_UKRAINIAN_HIERARCHY_2026-09-06.docx'
CACHE=ROOT/'tmp/ukrainian_pdf'


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    original_image=base._add_image
    def image(doc,path,alt):
        if path.suffix.lower()=='.svg':
            target=CACHE/(hashlib.sha256(path.read_bytes()).hexdigest()+'.png')
            if not target.exists(): cairosvg.svg2png(url=str(path),write_to=str(target),output_width=1800)
            path=target
        original_image(doc,path,alt)
    base._add_image=image
    doc=Document()
    compact.configure_template(doc)
    for level in (1,2,3):
        doc.styles[f'Heading {level}'].font.color.rgb=RGBColor(0,0,0)
    # Bare page number, with no inherited English "Page" prefix.
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    hp=doc.sections[0].header.paragraphs[0]
    hp.clear()
    field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); hp._p.append(field)
    # A readable review export: institutional page geometry and typeface;
    # no claim that unfilled front matter is ready for institutional submission.
    doc.styles['Normal'].paragraph_format.line_spacing=1.25
    files=sorted(SOURCE.glob('0*.md'))
    for path in files:
        text=path.read_text(encoding='utf-8')
        # Evidence anchors are machine-readable source metadata, never prose.
        text=re.sub(r'<!--.*?-->\s*','',text,flags=re.S)
        if path.name == '00_FRONT_MATTER.md':
            # The generated contents below replaces the source's contents note.
            text=re.sub(r'(?ms)^## Зміст\s*.*?(?=^## |\Z)','',text)
        # Caption recognition is language-local; preserve the actual Ukrainian text.
        before=len(doc.paragraphs)
        compact.markdown_lines(doc,text,path)
        for paragraph in doc.paragraphs[before:]:
            for run in paragraph.runs:
                run.font.name='Times New Roman'
                run._element.get_or_add_rPr().rFonts.set(qn('w:ascii'),'Times New Roman')
                run._element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'),'Times New Roman')
            if paragraph.text.startswith('Рисунок '):
                paragraph.paragraph_format.keep_together=True
                paragraph.paragraph_format.first_line_indent=Pt(0)
                paragraph.paragraph_format.line_spacing=1.0
                paragraph.paragraph_format.space_after=Pt(6)
                for run in paragraph.runs: run.font.size=Pt(11)
            elif paragraph.text.startswith('Таблиця '):
                paragraph.paragraph_format.keep_with_next=True
                paragraph.paragraph_format.first_line_indent=Pt(0)
                paragraph.paragraph_format.line_spacing=1.0
                for run in paragraph.runs: run.font.size=Pt(11)
        if path.name == '00_FRONT_MATTER.md':
            doc.add_page_break()
            heading=doc.add_paragraph('ЗМІСТ')
            heading.alignment=WD_ALIGN_PARAGRAPH.CENTER
            heading.paragraph_format.first_line_indent=Pt(0)
            heading.runs[0].bold=True
            toc=doc.add_paragraph()
            toc.paragraph_format.first_line_indent=Pt(0)
            field=OxmlElement('w:fldSimple')
            field.set(qn('w:instr'),r'TOC \o "1-3" \h \z \u')
            toc._p.append(field)
    # Keep rows intact; repeat headers is supplied by the base table builder.
    from docx.oxml import OxmlElement
    for table in doc.tables:
        for row in table.rows:
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.first_line_indent=Pt(0)
                    paragraph.paragraph_format.left_indent=Pt(0)
                    paragraph.paragraph_format.right_indent=Pt(0)
                    paragraph.alignment=WD_ALIGN_PARAGRAPH.LEFT
        for cell in table.rows[0].cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next=True
        header=' '.join(c.text for c in table.rows[0].cells)
        if 'MSE переднього плану h=4' in header:
            base._set_table_geometry(table,[1200,3500,2330,2330])
        elif 'Статус доказу' in header:
            base._set_table_geometry(table,[2600,1600,1600,3560])
        elif 'Повний пік виділеної' in header:
            base._set_table_geometry(table,[2600,3000,1880,1880])
        elif header == 'Питання Методика / результат Таблиця Статус':
            # Keep experiment identifiers readable in the appendix registry.
            base._set_table_geometry(table,[1500,2350,900,4610])
    doc.core_properties.title='Системний аналіз та інтелектуальні технології консистентного моделювання процесів ієрархічної генерації відеоданих'
    doc.core_properties.author='Леонід Русланович Кулик'
    doc.core_properties.subject='Поточний український рукопис; незавершені експерименти позначено явно'
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    doc.save(OUTPUT)
    receipt={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (CACHE/'source_hashes.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(OUTPUT)


if __name__=='__main__': main()
