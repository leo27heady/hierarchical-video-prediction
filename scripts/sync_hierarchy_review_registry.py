"""Refresh current caption registries and diagram hashes without moving prose."""
from pathlib import Path
import json,re,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/dissertation/condensed_translated_humanized_ukrainian'
def main():
    guide=ROOT/'docs/dissertation/DISSERTATION_REVIEWER_PROVENANCE_AND_REPRODUCTION_GUIDE_2026-08-29.md'
    text=guide.read_text(encoding='utf-8')
    # Retain the baseline registry's experiment/artifact details when captions
    # move. A caption-only registry would discard useful reproduction evidence.
    baseline=subprocess.run(['git','show','1454a01:'+guide.relative_to(ROOT).as_posix()],cwd=ROOT,check=True,capture_output=True).stdout.decode('utf-8')
    maps=json.loads((ROOT/'docs/reviews/hierarchy_integration_2026-09-06/numbering_map.json').read_text(encoding='utf-8'))
    inventory=json.loads((ROOT/'docs/reviews/hierarchy_integration_2026-09-06/experiment_evidence_inventory.json').read_text(encoding='utf-8'))
    details={}
    for key,a,b in [('table','## 8. Повний реєстр таблиць','## 9. Повний реєстр рисунків'),('figure','## 9. Повний реєстр рисунків','## 10. Артефакти')]:
        part=baseline.split(a,1)[-1].split(b,1)[0]
        for line in part.splitlines():
            cells=[c.strip() for c in line.strip('|').split('|')]
            if cells and re.fullmatch(r'[1-4А-ЯІЇЄ]\.\d+',cells[0]):
                details[(key,maps[key].get(cells[0],cells[0]))]='; '.join(cells[2:])
    chunks=[]
    for n,word,title in [(8,'Таблиця','таблиць'),(9,'Рисунок','рисунків')]:
        content=f'## {n}. Повний реєстр {title}\n\n| Номер | Назва | Джерело / умови доказу |\n|---|---|---|\n'
        for p in sorted(SRC.glob('0*.md')):
            for id,caption in re.findall(rf'(?m)^\*\*{word} ([1-4А-ЯІЇЄ]\.\d+) [–-] (.*?)\*\*',p.read_text(encoding='utf-8')):
                key='table' if n==8 else 'figure'
                evidence=details.get((key,id),'авторська схема / явне місце результату; реєстр походження та покриття')
                if key=='table':
                    row=next((r for r in inventory if r['table']==id),None)
                    if row:evidence=f"{row['key']}; {row['artifact'] or 'очікується / умови в experiment_evidence_inventory.json'}"
                content+=f'| {id} | {caption.replace("|","/")} | `{p.name}`; {evidence.replace("|","/")} |\n'
        chunks.append(content)
    text=text[:text.index('## 8. Повний реєстр таблиць')]+'\n'.join(chunks)+'\n'+text[text.index('## 10. Артефакти'):]
    guide.write_text(text,encoding='utf-8')
    ledger=ROOT/'figures/restructured/integrated_figure_provenance.json';rows=json.loads(ledger.read_text(encoding='utf-8'))
    for row in rows:
        if row.get('generator')=='scripts/build_hierarchy_diagrams.py':
            asset=ROOT/row['file'];row.update(sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),bytes=asset.stat().st_size,generator_sha256=hashlib.sha256((ROOT/row['generator']).read_bytes()).hexdigest())
    ledger.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Caption registries and native diagram hashes synchronized.')
if __name__=='__main__':main()
