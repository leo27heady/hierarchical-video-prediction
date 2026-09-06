"""Render every exported PDF page, make contact sheets and geometry/text checks."""
from pathlib import Path
import json
import hashlib
import pymupdf
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/ukrainian_pdf/inspection'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pdf_path=ROOT/'output/pdf/DISSERTATION_UKRAINIAN_HIERARCHY_2026-09-06.pdf'
    pdf=pymupdf.open(pdf_path)
    needles=['Рисунок 3.18','Рисунок 3.19','Рисунок 3.22','Таблиця 4.6','Таблиця 4.7','Таблиця 4.8','Таблиця 4.10','Таблиця 4.18','Таблиця 4.19','Таблиця 4.21','Рисунок 4.12','Рисунок 4.13','Рисунок 4.16','(3.55)','Таблиця И.1']
    found={key:[] for key in needles}; outside=[]; blanks=[]
    for i,page in enumerate(pdf):
        text=page.get_text()
        for key in needles:
            if key in text:found[key].append(i+1)
        if len(text.strip())<5:blanks.append(i+1)
        for b in page.get_text('blocks'):
            if b[0]<-1 or b[1]<-1 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1:
                outside.append({'page':i+1,'bbox':list(b[:4]),'text':str(b[4])[:100]})
        page.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(OUT/f'page-{i+1:03}.png')
    for start in range(0,len(pdf),24):
        sheet=Image.new('RGB',(1100,6*410),'#cccccc'); draw=ImageDraw.Draw(sheet)
        for j in range(start,min(start+24,len(pdf))):
            im=Image.open(OUT/f'page-{j+1:03}.png');im.thumbnail((265,380))
            x=((j-start)%4)*275;y=((j-start)//4)*410
            sheet.paste(im,(x,y+25));draw.text((x+5,y+4),str(j+1),fill='black')
        sheet.save(OUT/f'contact-{start//24+1:02}.jpg')
    result={'pages':len(pdf),'pdf_sha256':hashlib.sha256(pdf_path.read_bytes()).hexdigest(),'key_pages':found,'outside_page_blocks':outside,'near_blank_pages':blanks,
            'replacement_characters':sum(p.get_text().count(chr(65533)) for p in pdf),
            'renderer':'Word PDF export + PyMuPDF; packaged DOCX renderer unavailable: LibreOffice absent'}
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
