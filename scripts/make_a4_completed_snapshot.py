"""Read-only frozen snapshot of completed A4 logs and diagnostic validation panels.

Selection fixed before reading panel errors: FRT B1152 flat_recent vs geometric
age, seed 0, step 30000, horizon 4. Choose maximal advantage of either arm and
maximal minimum error, with distinct rows. These are selected diagnostics among
the saved 16 validation examples, never representative held-out evidence.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT.parent / 'hierarchical-video-prediction-path-b/artifacts/v2/article4_primary_30k'
OUT = ROOT / 'docs/reviews/a4_completed_snapshot_2026-09-05'
SOURCE = ROOT / 'docs/dissertation/condensed_translated_humanized_ukrainian'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((OUT/'snapshot.json').read_text()) if (OUT/'snapshot.json').exists() else None
    frozen_ids = {r['run_id'] for r in previous['runs']} if previous else None
    rows = []
    for run in sorted((LIVE / 'runs').iterdir()):
        status_path = run / 'status.json'
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get('status') != 'complete' or status.get('step') != 30000:
            continue
        if frozen_ids is not None and status['experiment_id'] not in frozen_ids:
            continue
        manifest = json.loads((run / 'run_manifest.json').read_text())
        assert manifest['git_commit'] == '7a5979615f5354564a793083b673cf1ed9ec77be'
        assert not manifest['git_dirty_at_launch']
        assert status['config_hash'] == manifest['config_hash']
        for name in ('training_metrics', 'validation_metrics'):
            assert sha(run / (name + '.jsonl')) == status[name + '_sha256'], run.name
        milestones = list((run / 'milestones').glob('*30000*'))
        assert len(milestones) == 1, milestones
        assert sha(milestones[0]) == status['final_milestone_sha256']
        metrics = [json.loads(line) for line in (run / 'validation_metrics.jsonl').read_text().splitlines()]
        final = next(r for r in metrics if r['step'] == 30000)
        assert np.isfinite(final['horizon4_foreground_mse'])
        cfg = manifest['config']
        rows.append(dict(run_id=status['experiment_id'], path=str(run), config_hash=status['config_hash'],
                         family=cfg['family'], budget=cfg['budget_tokens'], policy=cfg['policy'],
                         seed=cfg['seed'], horizon4_foreground_mse=final['horizon4_foreground_mse'],
                         pixel_mse=final['pixel_mse'], checkpoint_sha256=status['final_milestone_sha256'],
                         validation_sha256=status['validation_metrics_sha256']))
    receipt = {'utc': previous['utc'] if previous else datetime.now(timezone.utc).isoformat(), 'analyzer_sha256': sha(Path(__file__)),
               'evidence_class': 'completed training; partial campaign; fixed-step validation only',
               'completed_runs': len(rows), 'runs': rows}
    (OUT / 'snapshot.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    frt = [r for r in rows if r['family'] == 'FRT']
    groups = sorted({(r['budget'], r['policy']) for r in frt})
    table = ['| Бюджет | Політика | MSE переднього плану h=4, ініц. 0 | Ініц. 1 |',
             '|---:|---|---:|---:|']
    labels = {'flat_recent':'плоска недавня', 'balanced_recent':'збалансована недавня',
              'geometric_age':'геометрична за віком', 'geometric':'геометрична за віком',
              'linear_age':'лінійна за віком'}
    for budget, policy in groups:
        pair = {r['seed']: r for r in frt if r['budget'] == budget and r['policy'] == policy}
        def value(seed):
            return f"{pair[seed]['horizon4_foreground_mse']:.5f}".replace('.', ',') if seed in pair else 'очікується'
        table.append(f'| {budget} | {labels.get(policy, policy)} | {value(0)} | {value(1)} |')
    text = f'''### 4.9.3 Поточна перевірка системного правила: завершені траєкторії

На 5 вересня 2026 року перевірено {len(rows)} завершених навчань із 86 запланованих; для кожного звірено хеші журналів і контрольної точки 30 000 кроків. Таблиця 4.10 містить усі завершені комірки фронтиру FRT, а не відібрані найкращі результати. Це валідаційний зріз незавершеної кампанії: він не замінює відкладене підтвердження й не змінює висновків історичних E2–E8.

**Таблиця 4.10 – Завершені FRT-навчання за спільного бюджету 30 000 кроків.** Менша MSE краща; дві ініціалізації наведено окремо. Оцінка використовує фіксовані 64 вибірки з поверненням, а не всі 300 валідаційних кліпів.

''' + '\n'.join(table) + '''

Первинне ранжування використовує MSE переднього плану на четвертому горизонті: усередині пакета сума квадратичних помилок ділиться на число еталонних пікселів переднього плану, після чого усереднюються чотири пакети. Натомість підтверджувальний оцінювач зберігає окремі помилки кліпів і горизонтів. Ці агрегації не тотожні за різної площі об’єктів. Порожня маска в наявному оцінювачі дає нуль через знаменник, обмежений знизу одиницею; кількість таких випадків слід подавати окремо. Нова версія парного аналізу вирівнює ідентифікатори кліпів, використовує лише горизонт 4 для основної оцінки та повторно вибирає цілі кліпи. Позитивна різниця «плоска мінус ієрархічна» сприятлива для ієрархії. Інтервали для зафіксованих моделей не оцінюють розподіл між навчаннями; без заздалегідь визначеного допуску висновок про не-гіршість не формується.

Незавершені ALC/CEX/STS і залежні LVL/OVR-комірки, повний валідаційний прохід короткого списку та відкладене підтвердження залишаються незаповненими до появи перевірених артефактів. Очікуваний позитивний напрям є гіпотезою, а не значенням для таблиці.

'''
    chapter = SOURCE / '05_CHAPTER_4.md'
    content = chapter.read_text(encoding='utf-8')
    anchor = '## 4.10 Набір програмних засобів та практичне застосування'
    assert anchor in content
    # Artifact generation must never reinsert a chronological subsection.
    # Narrative placement and numbering are owned by the manuscript/inventory.
    update_manuscript = False
    if update_manuscript:
        chapter.write_text(content.replace(anchor, text + anchor), encoding='utf-8')
    # Use only saved final-step panels: no model inference and no CUDA access.
    candidates = [r for r in frt if r['budget'] == 1152 and r['seed'] == 0]
    flat = next(r for r in candidates if r['policy'] == 'flat_recent')
    hier = next(r for r in candidates if 'geom' in r['policy'])
    files = [Path(r['path']) / 'panel_data/panel_step030000.npz' for r in (flat, hier)]
    a, b = [np.load(p) for p in files]
    for key in ('sampled_clips', 'sampled_starts', 'target', 'last'):
        np.testing.assert_array_equal(a[key], b[key])
    target = a['target'][:, 3, 0].astype(float)
    pa, pb = a['prediction'][:, 3, 0].astype(float), b['prediction'][:, 3, 0].astype(float)
    ea, eb = ((pa-target)**2).mean((1,2)), ((pb-target)**2).mean((1,2))
    chosen = []
    for score in (ea-eb, eb-ea, np.minimum(ea,eb)):
        chosen.append(next(int(i) for i in np.argsort(-score, kind='stable') if int(i) not in chosen))
    titles = ['Останній контекст', 'Еталон, h=4', 'Плоский', 'Ієрархічний', '|Помилка| плоска', '|Помилка| ієрарх.']
    fig, axes = plt.subplots(3,6,figsize=(13,7), constrained_layout=True)
    for row, index in enumerate(chosen):
        data = [a['last'][index,0],target[index],pa[index],pb[index],abs(pa[index]-target[index]),abs(pb[index]-target[index])]
        for col, pixels in enumerate(data):
            axes[row,col].imshow(pixels, cmap='magma' if col>=4 else 'gray', vmin=0,vmax=1)
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(titles[col],fontsize=10)
        axes[row,0].set_ylabel(f"кліп {a['sampled_clips'][index]}\nt={a['sampled_starts'][index]-1}",fontsize=10)
    figure = ROOT / 'figures/restructured/ch4_fig4_14_completed_validation_examples.png'
    fig.savefig(figure,dpi=180); plt.close(fig)
    receipt['qualitative'] = {'panel_files': [str(p) for p in files], 'hashes':[sha(p) for p in files],
                             'rows':chosen, 'clip_ids':a['sampled_clips'][chosen].tolist(),
                             'flat_mse':ea[chosen].tolist(),'hierarchical_mse':eb[chosen].tolist()}
    (OUT/'snapshot.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    panel = '''![Зіставлені приклади завершених FRT-моделей](../../../figures/restructured/ch4_fig4_14_completed_validation_examples.png)

**Рисунок 4.15 – Діагностичні приклади двох завершених FRT-моделей за бюджету 1 152 токени, ініціалізації 0 та 30 000 кроків.** Показано останнє спостереження й четвертий прогнозований кадр. Рядки вибрано серед 16 збережених валідаційних прикладів за найбільшою перевагою ієрархії, найбільшою перевагою плоского пакета та найбільшою спільною помилкою без повторення рядків. Це навмисно вибрані випадки, а не оцінка їхньої частоти. Яскравість кадрів і абсолютних помилок має спільний діапазон [0; 1]. Ідентифікатори кліпів і часу подано на рисунку; склад давнішої історії задає відповідний план пакета, а не лише зображений останній кадр.

'''
    content = chapter.read_text(encoding='utf-8')
    if update_manuscript:
        chapter.write_text(content.replace(anchor, panel + anchor),encoding='utf-8')
    print(json.dumps({'completed':len(rows),'frt':len(frt),'figure':str(figure)}))


if __name__ == '__main__':
    main()
