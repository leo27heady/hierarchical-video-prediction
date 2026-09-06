"""One-time numbering, evidence inventory and source/provenance migration."""
from pathlib import Path
import re,json,hashlib,itertools
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/dissertation/condensed_translated_humanized_ukrainian'
OUT=ROOT/'docs/reviews/hierarchy_integration_2026-09-06'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p3=SRC/'04_CHAPTER_3.md';p4=SRC/'05_CHAPTER_4.md'
    c3=p3.read_text(encoding='utf-8');c4=p4.read_text(encoding='utf-8')
    assert '4.XALC' in c4,'Migration already applied'
    # Place genuine empty figure frames at the corresponding evidence blocks.
    def panel(name,number,caption):return f'![{caption}](../../../figures/restructured/{name})\n\n**Рисунок {number} – {caption}.** Це явно незаповнене місце для результатів, а не графік вимірювань. Змінні, порівняння та необхідні дані підписано в рамці; числові криві або прогнози додаються лише після перевірки артефактів.\n\n'
    c4=c4.replace('### 4.8.3 ',panel('hierarchy_structural_result_slots.svg','4.XS','Якість за зміни місткості та геометрії контексту')+'### 4.8.3 ')
    c4=c4.replace('## 4.9 Якість',panel('hierarchy_diagnostic_result_slots.svg','4.XD','Зіставлені приклади та умовна діагностика самоподачі')+'## 4.9 Якість')
    c4=c4.replace('<!-- evidence:COMP -->',panel('hierarchy_integrated_result_slots.svg','4.XI','Прогнозні приклади для двох задач')+'<!-- evidence:COMP -->')
    # Preserve verified reused FRT values rather than blanking known quantities.
    for old,new in [
        ('| Плоска недавня | очікується | очікується |','| Плоска недавня | 0,02225 | 0,02084 |'),
        ('| Лінійна за віком | очікується | очікується |','| Лінійна за віком | 0,02027 | 0,02215 |'),
        ('| Геометрична за віком | очікується | очікується |','| Геометрична за віком | 0,01957 | 0,01967 |')]:c4=c4.replace(old,new)
    # Remove development chronology, retaining scientific temporal descriptions.
    replacements={
        'Нова матриця додатково використовує':'Структурна матриця використовує',
        'Розширена матриця контекстів':'Матриця структурних порівнянь',
        'Остаточна інтегрована архітектура замінює сталі регіони':'Архітектура з адаптивним пакуванням доповнює дослідження сталих регіонів',
        'Походження принципів остаточної інтегрованої моделі':'Принципи побудови ієрархічного предиктора',
        'Остаточна реалізація':'Реалізація предиктора',
        'остаточний інтегрований':'адаптивний інтегрований',
        'остаточної інтегрованої':'адаптивної інтегрованої',
        'попередня інтегрована':'інтегрована зі сталими ділянками',
        'попередній прототип':'прототип зі сталими ділянками',
        'попереднього прототипу':'прототипу зі сталими ділянками',
        'другорядна попередня межа':'окрема перевірка межі',
        'історичних пар':'контрольованих пар',
        'історичної пари':'контрольованої пари',
        'історичні E2–E8':'контрольовані E2–E8',
        'нової матриці':'структурної матриці',
        'у фінальній редакції':'у дослідженні',
    }
    for a,b in replacements.items():c3=c3.replace(a,b);c4=c4.replace(a,b)
    c3=c3.replace('розділ 4 подає наявний доказовий ланцюг та окремо подає завершені валідаційні результати структурної матриці та явні місця для ще не отриманих інтегрованих вимірювань','розділ 4 послідовно зіставляє досяжність, структуру пакета, прогнозну якість та ресурсну вартість; висновки залежать від статусу відповідних вимірювань')
    # No blanket claim of one architecture. Scientific roles are the connective tissue.
    c4=c4.replace('це не робить архітектуру єдиною експериментальною лінією','це не ототожнює протоколи').replace('Це не робить архітектуру єдиною експериментальною лінією','Це не ототожнює протоколи')
    c4=c4.replace('Саме так зберігається єдина системна історія без удаваного твердження, що всі статті досліджували одну незмінну архітектуру або що коректна композиція вже гарантує емпіричну перевагу.','Контрольовані пакети, предиктор зі сталими ділянками й адаптивне ядро досліджують різні інтерфейси спільної задачі. Перенесення висновку між ними потребує відповідного зіставлення, а коректна композиція сама по собі не встановлює переваги якості.')
    p3.write_text(c3,encoding='utf-8');p4.write_text(c4,encoding='utf-8')
    # Build order-based caption maps once; update contextual references only.
    maps={'figure':{},'table':{}}
    for p in [p3,p4]:
        text=p.read_text(encoding='utf-8');chapter='3' if p==p3 else '4'
        for key,word in [('figure','Рисунок'),('table','Таблиця')]:
            ids=re.findall(rf'(?m)^\*\*{word}\s+({chapter}\.[A-Za-z0-9]+)\s+[–-]',text)
            assert len(ids)==len(set(ids)),(word,ids)
            maps[key].update({old:f'{chapter}.{i}' for i,old in enumerate(ids,1)})
    for p in SRC.glob('0*.md'):
        text=p.read_text(encoding='utf-8')
        for key,stem in [('figure',r'(?:[Рр]исун\w*|[Рр]ис\.)'),('table',r'(?:[Тт]аблиц\w*|[Тт]абл\.)')]:
            text=re.sub(rf'({stem}\s+)([34]\.[A-Za-z0-9]+)',lambda m:m[1]+maps[key].get(m[2],m[2]),text)
        p.write_text(text,encoding='utf-8')
    (OUT/'numbering_map.json').write_text(json.dumps(maps,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Ledger entries keyed by actual source image link/caption, including subpanels.
    image_ids={}
    for p in SRC.glob('0*.md'):
        pending=[]
        for line in p.read_text(encoding='utf-8').splitlines():
            m=re.search(r'!\[[^]]*\]\(([^)]+)\)',line)
            if m:pending.append((p.parent/m[1]).resolve().relative_to(ROOT).as_posix())
            m=re.match(r'\*\*Рисунок\s+([1-4ВЕ]\.\d+)\s',line)
            if m:
                for f in pending:image_ids[f]=m[1]
                pending=[]
    ledgers=[ROOT/'docs/figures/restructured'/x for x in ['figure_provenance.json','ch2_ch4_figure_provenance.json','existing_figure_provenance.json']]+[ROOT/'figures/restructured/integrated_figure_provenance.json']
    known=set()
    for ledger in ledgers:
        rows=json.loads(ledger.read_text(encoding='utf-8'))
        for row in rows:
            file=Path(row.get('file','')).as_posix();known.add(file)
            if file in image_ids:
                old=row.get('figure_id');row['figure_id']=image_ids[file]
                if old!=row['figure_id']:row['caption_attribution']=f"Ukrainian Figure {row['figure_id']}; asset unchanged, narrative order updated 2026-09-06"
        ledger.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    ledger=ledgers[-1];rows=json.loads(ledger.read_text(encoding='utf-8'))
    generator=ROOT/'scripts/build_hierarchy_diagrams.py'
    for file,id in image_ids.items():
        if file in known:continue
        asset=ROOT/file
        rows.append(dict(figure_id=id,file=file,asset_class='author-generated-placeholder' if 'slots' in file else 'author-generated-conceptual',sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),bytes=asset.stat().st_size,generator=str(generator.relative_to(ROOT)).replace('\\','/'),generator_sha256=hashlib.sha256(generator.read_bytes()).hexdigest(),source='Hierarchy protocol definitions, not measured outcomes',caption_attribution=f'Ukrainian Figure {id}; author-generated',rights_status='author-generated'))
    ledger.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Generate coverage inventory with complete dimensions, not fictitious metrics.
    text=p4.read_text(encoding='utf-8')
    method={'FRT':'3.6.3','ALC':'3.6.3','CEX':'3.6.4','STS':'3.6.4','LVL':'3.6.5','OVR':'3.6.5','CONF':'3.6.6','MECH':'3.6.6','VAL300':'3.6.6','ROLL64':'3.6.6','PARITY':'3.7.7','RESOURCE':'3.7.7','SCREEN':'3.7.7','QUALITY':'3.7.7','COMP':'3.7.7'}
    dimensions={
        'FRT':{'budget':[320,768,1152,1792,2432,3072],'policy':['flat','linear','geometric'],'seed':[0,1]},
        'ALC':{'policy':['flat','fine_heavy','balanced','coarse_heavy','linear','geometric','reach'],'seed':[0,1]},
        'CEX':{'coarse_positions':[0,4,8,16,32,64],'seed':[0,1]},
        'STS':{'K':[64,32,16,8],'placement':['recent','uniform','geometric'],'seed':[0,1]},
        'LVL':{'levels':[1,2,3],'seed':[0,1]},'OVR':{'overlap':['nested','exclusive','residual'],'seed':[0,1]},
        'CONF':{'role':['best_flat_1792','best_hierarchy_1792','memory_pareto_within_5pct'],'seed':[0,1]},
        'MECH':{'role':['best_flat_1792','best_hierarchy_1792','failure_hierarchy_1792'],'seed':[0,1]},
        'VAL300':{'shortlist_slot':[1,2,3],'seed':[0,1]},'ROLL64':{'model':['flat','hierarchy'],'seed':[0,1],'horizon':[4,16,32,64]},
        'PARITY':{'dataset':['mmnist','taxibj'],'precision':['fp32','intended_training_precision'],'pair':['eager/checkpoint','eager/reversible'],'quantity':['output','input_gradient','named_parameter_gradients']},
        'RESOURCE':{'dataset':['mmnist','taxibj'],'packing':['flat_dense','hier_static','hier_hybrid'],'execution':['eager','checkpoint','reversible'],'precision':['fp32','intended_training_precision'],'phase':['inference','training'],'scope':['whole_model','core_operation']},
        'SCREEN':{'trajectory':['mmnist_hybrid','taxibj_hybrid','mmnist_flat','mmnist_static']},
        'QUALITY':{'run':['INT_MM_S00','INT_MM_S01','INT_TB_S00','INT_TB_S01']},'COMP':{'dataset':['mmnist','taxibj']}}
    rows=[]
    for key in method:
        pos=text.index(f'<!-- evidence:{key} -->')
        heads=re.findall(r'(?m)^### (4\.\d+\.\d+) ',text[:pos]);section=heads[-1]
        table=re.search(r'\*\*Таблиця (4\.\d+)',text[pos:])[1]
        dims=dimensions[key]
        cells=[dict(zip(dims,v),status='очікується',artifact=None) for v in itertools.product(*dims.values())]
        row=dict(key=key,methods_section=method[key],results_section=section,table=table,dimensions=dims,result_slots=cells,status='умовний дослід' if key in ['ROLL64','COMP'] else 'очікується',dependency='ALC validation selection' if key in ['LVL','OVR'] else 'frozen shortlist' if key in ['CONF','MECH','VAL300','ROLL64'] else 'CUDA parity and bounded screen' if key=='QUALITY' else None,metric='MSE h=4 per protocol' if key in ['FRT','ALC','CEX','STS','LVL','OVR','CONF','VAL300'] else 'defined in corresponding methods and table caption',artifact=None)
        if key=='FRT':
            row.update(status='перевірено',artifact='docs/reviews/a4_completed_snapshot_2026-09-05/snapshot.json')
            for cell in cells:cell.update(status='перевірено',artifact=row['artifact'])
        if key=='ALC':
            row['status']='частково перевірено: три повторно використані політики FRT'
            for cell in cells:
                if cell['policy'] in ['flat','linear','geometric']:cell.update(status='перевірено',artifact='docs/reviews/a4_completed_snapshot_2026-09-05/snapshot.json')
        if key=='PARITY':row['cpu_evidence']='24 CPU model/data tests passed on 2026-09-05; CUDA slots below are not populated by CPU results'
        row['admissible_claim']='Descriptive only until corresponding protocol evidence is verified; no fabricated favorable outcomes'
        rows.append(row)
    (OUT/'experiment_evidence_inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    appendix=SRC/'08_APPENDICES.md'
    content=appendix.read_text(encoding='utf-8')
    content+='''\n## Додаток И – Покриття експериментальних питань

Реєстр пов’язує методику з місцем результату. Повні комбінації умов та їхній статус збережено у `docs/reviews/hierarchy_integration_2026-09-06/experiment_evidence_inventory.json`. «Очікується» означає відсутність перевіреного вимірювання; умовні досліди не є обіцянкою запуску. Одне навчання може обслуговувати кілька ролей, тому кількість комірок не дорівнює кількості незалежних моделей.

**Таблиця И.1 – Реєстр покриття експериментальних питань.** Основні таблиці компактні; цей реєстр визначає повні виміри для заповнення й не додає навчальних запусків.

| Питання | Методика / результат | Таблиця | Статус |
|---|---|---|---|
'''
    for row in rows:
        content+=f"| {row['key']} | {row['methods_section']} / {row['results_section']} | {row['table']} | {row['status']} |\n"
    content+='''
Ресурсний реєстр охоплює обидві задачі, три способи пакування, три режими ядра, FP32 і точність навчання, застосування й навчальний крок, повну модель та операції ядра. Кожний запис передбачає allocated/reserved baseline і peak у байтах, додатковий пік, типи тензорів, унікальні сховища, saved tensors, параметри/градієнти/стан оптимізатора та час повторів. Непідтримувана точність позначається незастосовною з причиною, а не нульовою витратою. Для QUALITY повний артефакт містить результати за горизонтами, нормалізацію, розбиття, контрольну точку й приклади. Для PARITY зберігаються абсолютні та відносні розбіжності й допуски кожної величини.
'''
    appendix.write_text(content,encoding='utf-8')
    # Rebuild the reviewer table/figure registries from the authoritative captions.
    guide=ROOT/'docs/dissertation/DISSERTATION_REVIEWER_PROVENANCE_AND_REPRODUCTION_GUIDE_2026-08-29.md'
    g=guide.read_text(encoding='utf-8')
    ts='## 8. Повний реєстр таблиць\n\n| Номер | Назва | Джерело |\n|---|---|---|\n'
    fs='## 9. Повний реєстр рисунків\n\n| Номер | Підпис | Джерело |\n|---|---|---|\n'
    for p in sorted(SRC.glob('0*.md')):
        content=p.read_text(encoding='utf-8')
        for id,caption in re.findall(r'(?m)^\*\*Таблиця ([1-4А-ЯІЇЄ]\.\d+) [–-] (.*?)\*\*',content):ts+=f'| {id} | {caption.replace("|","/")} | `{p.name}` |\n'
        for id,caption in re.findall(r'(?m)^\*\*Рисунок ([1-4ВЕ]\.\d+) [–-] (.*?)\*\*',content):fs+=f'| {id} | {caption.replace("|","/")} | `{p.name}`; реєстр походження активів |\n'
    g=g[:g.index('## 8. Повний реєстр таблиць')]+ts+'\n'+fs+'\n'+g[g.index('## 10. Артефакти'):]
    g+='\n\n## Структурне узгодження 6 вересня 2026 року\n\nПоточна карта: §3.6 — структура контексту; §3.7 — предиктор і контракти; §4.8 — структурні результати; §4.9 — якість і ресурси; §4.10 — синтез; §4.11 — реалізація; §4.12 — обмеження. Попередні списки прив’язок нижче/вище зберігають походження активів; чинні номери визначають реєстри 8–9 та `docs/reviews/hierarchy_integration_2026-09-06/numbering_map.json`.\n'
    guide.write_text(g,encoding='utf-8')
    counts={str(ch):{kind:len(re.findall(rf'(?m)^\*\*{word} {ch}\.\d+ ',(p3 if ch==3 else p4).read_text(encoding='utf-8'))) for kind,word in [('figures','Рисунок'),('tables','Таблиця')]} for ch in [3,4]}
    print(json.dumps(counts));print('Inventory families:',len(rows))

if __name__=='__main__':main()
