"""Structural and evidence-coverage gate for the authoritative Ukrainian manuscript."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/dissertation/condensed_translated_humanized_ukrainian'
EXPECTED={'FRT','ALC','CEX','STS','LVL','OVR','CONF','MECH','VAL300','ROLL64','PARITY','RESOURCE','SCREEN','QUALITY','COMP'}

def heading_errors(text):
    ids=re.findall(r'(?m)^#{2,4} ([1-4]\.\d+(?:\.\d+)?) ',text)
    errors=[f'duplicate heading: {id}' for id in sorted(set(ids)) if ids.count(id)>1]
    for id in ids:
        if id.count('.')==2 and id.rsplit('.',1)[0] not in ids:errors.append(f'heading has no parent: {id}')
    for parent in sorted(set(id.rsplit('.',1)[0] for id in ids)):
        children=[int(id.rsplit('.',1)[1]) for id in ids if id.rsplit('.',1)[0]==parent]
        if children!=list(range(1,len(children)+1)):errors.append(f'nonsequential children: {parent}: {children}')
    return errors

def object_references(text, stem):
    """Include every member of a list/range, not only its first number."""
    number=r'[1-4А-ЯІЇЄ]\.\d+'
    groups=re.findall(rf'{stem}\s+({number}(?:\s*(?:,|і|й|та|–|-)\s*{number})*)',text)
    return [id for group in groups for id in re.findall(number,group)]

def check():
    texts={p.name:p.read_text(encoding='utf-8') for p in sorted(SRC.glob('0*.md'))}
    combined='\n'.join(texts.values());errors=[]
    for name,text in texts.items():errors.extend(f'{name}: {e}' for e in heading_errors(text))
    headings=set(re.findall(r'(?m)^#{2,4} ([1-4]\.\d+(?:\.\d+)?) ',combined))
    for ref in re.findall(r'§{1,2}([1-4]\.\d+(?:\.\d+)?)',combined):
        if ref not in headings:errors.append(f'undefined section reference: {ref}')
    for stem,word in [(r'[Рр]исун\w*','Рисунок'),(r'[Тт]аблиц\w*','Таблиця')]:
        ids=set(re.findall(rf'(?m)^\*\*{word} ([1-4А-ЯІЇЄ]\.\d+) ',combined))
        for ref in object_references(combined,stem):
            if ref not in ids:errors.append(f'undefined {word} reference: {ref}')
    c4=texts['05_CHAPTER_4.md'];c3=texts['04_CHAPTER_3.md']
    anchors=re.findall(r'<!-- evidence:(\w+) -->',c4)
    if set(anchors)!=EXPECTED or len(anchors)!=len(EXPECTED):errors.append('missing or duplicate experiment anchors')
    inv=json.loads((ROOT/'docs/reviews/hierarchy_integration_2026-09-06/experiment_evidence_inventory.json').read_text(encoding='utf-8'))
    if {r['key'] for r in inv}!=EXPECTED or len(inv)!=len(EXPECTED):errors.append('inventory families differ from required set')
    for row in inv:
        key=row['key'];marker=f'<!-- evidence:{key} -->'
        if marker not in c4:continue
        before,after=c4.split(marker,1)
        actual=re.findall(r'(?m)^### (4\.\d+\.\d+) ',before)[-1]
        table=re.search(r'\*\*Таблиця (4\.\d+)',after)[1]
        if actual!=row['results_section'] or table!=row['table']:errors.append(f'{key}: stale location')
        if row['methods_section'] not in headings:errors.append(f'{key}: missing methods')
        if not row['result_slots']:errors.append(f'{key}: no detailed result slots')
        for cell in row['result_slots']:
            if cell['status']=='перевірено' and (not cell.get('artifact') or not (ROOT/cell['artifact']).is_file()):errors.append(f'{key}: verified cell without artifact')
    if c4.index('<!-- evidence:FRT -->')>c4.index('## 4.10 '):errors.append('structural evidence follows synthesis')
    if 'результати цих порівнянь наведено в §4.8' not in c3:errors.append('structural methods bridge does not point to structural results')
    for phrase in ['Поточна перевірка системного правила: завершені траєкторії','Нова експериментальна гілка','Протокол і незаповнені результати остаточної архітектури']:
        if phrase in combined:errors.append(f'chronological insertion remains: {phrase}')
    if re.search(r'(?m)^\*\*(?:Рисунок|Таблиця) [34]\.X',combined):errors.append('temporary caption number remains')
    counts={k:sum(1 for row in inv for cell in row['result_slots'] if cell['status']==k) for k in ['перевірено','очікується']}
    print(json.dumps({'families':len(inv),'detailed_result_slots':counts,'headings':len(headings)},ensure_ascii=True))
    for e in dict.fromkeys(errors):print('ERROR:',e)
    print('FAIL' if errors else 'HIERARCHY STRUCTURE AND COVERAGE PASS')
    return bool(errors)

if __name__=='__main__':sys.exit(check())
