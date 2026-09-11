"""Dated, idempotent manuscript migration from a preserved September 11 baseline."""
from pathlib import Path
import json,re,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'docs/dissertation/condensed_translated_humanized_ukrainian'
OUT=ROOT/'docs/reviews/integrated_evidence_2026-09-11';FIG=ROOT/'figures/restructured'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def num(x,n=5):return f'{x:.{n}f}'.replace('.',',')
def main():
    backup=ROOT/'tmp/integrated_evidence_20260911/before';backup.mkdir(parents=True,exist_ok=True)
    for p in SRC.glob('0*.md'):
        if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
    a=read(OUT/'audit.json');text=(backup/'05_CHAPTER_4.md').read_text(encoding='utf-8')
    resource='''**Таблиця 4.19 – Повний ресурсний профіль предиктора.** Навчальний крок, адаптивне пакування, BF16-autocast; Moving MNIST: пакет 8, TaxiBJ: пакет 16. Медіани п’яти повторів після двох прогрівань; пам’ять у GiB, час у секундах.

| Задача / режим | Пік allocated / reserved | Додатковий пік / час |
| --- | --- | --- |
'''
    names={'eager':'звичайний','checkpoint':'контрольні точки','reversible':'реверсивний'}
    for profile in a['profiles']:
        if profile['precision']!='bf16':continue
        for c in profile['cases']:
            if not c['training'] or c['routing']!='hybrid':continue
            s=c['summary'];ds='Moving MNIST' if profile['dataset']=='mmnist' else 'TaxiBJ'
            resource+=f"| {ds}: {names[c['execution_mode']]} | {num(s['absolute_peak_allocated_bytes']/2**30,3)} / {num(s['absolute_peak_reserved_bytes']/2**30,3)} | {num(s['incremental_peak_allocated_bytes']/2**30,3)} / {num(c['median_seconds'],4)} |\n"
    resource+='''
Повний профіль охоплює 72 умови: дві задачі, FP32 і BF16, плоске, статичне та адаптивне пакування, три режими ядра, застосування й навчальний крок. Вимірювання виконано на RTX 4060 Ti зі сталими початковими вагами, ненульовою вихідною проєкцією та синтетичним входом. Кожний повтор створює стан AdamW заново з нульовою швидкістю навчання; це контрольований ресурсний зонд, а не час епохи повного навчання з Adam. Діагностичний граф вивільняється до вимірювання; базова allocated-пам’ять між трьома режимами адаптивного пакета однакова: 0,0255 GiB для Moving MNIST і 0,0240 GiB для TaxiBJ.

У BF16 реверсивне ядро зменшує повний allocated-пік відносно звичайного з 0,764 до 0,673 GiB для Moving MNIST і з 0,174 до 0,156 GiB для TaxiBJ — приблизно на 12 % і 10 %. У FP32 відповідні піки становлять 1,106→0,945 та 0,250→0,218 GiB. Отже, виграш внутрішнього ядра зберігається на рівні предиктора, але його обмежують інші активації, декодер і робочі буфери. Це не коефіцієнт скорочення всього навчального процесу за довільного горизонту.

Час має окрему невизначеність: для реверсивного TaxiBJ у BF16 п’ять повторів дали 0,052–0,197 с. За такого розкиду медіани не обґрунтовують стійкого ранжування швидкодії. Реєстр зберігає всі повтори й точні байти; окремі операції ядра цим повним профілем не виміряні. За BF16-autocast упаковані токени залишаються FP32, а груба обхідна карта — BF16. Тому аналітичне скорочення кількості токенів не підміняє вимірювання байтів.

'''
    start=text.index('**Таблиця 4.19');end=text.index('### 4.9.5',start);text=text[:start]+resource+text[end:]
    quality='''**Таблиця 4.21 – Якість ієрархічного предиктора.** Тестова вибірка, контрольна точка з мінімальною валідаційною MSE. MSE та MAE усереднено за пікселями й каналами у нормалізованих одиницях завантажувача.

| Задача / горизонт | Ініціалізація | MSE / MAE | SSIM / PSNR, dB |
| --- | --- | --- | --- |
'''
    for q in a['quality']:
        m=q['result']['test_metrics'];ds='Moving MNIST, 10→10' if '_MM_' in q['run'] else 'TaxiBJ, 4→4'
        quality+=f"| {ds} | {q['run'][-1]} | {num(m['pixel_mse'],6)} / {num(m['pixel_mae'],6)} | {num(m['ssim'],4)} / {num(m['psnr'],2)} |\n"
    quality+='''
Moving MNIST оцінено на 10 000 тестових кліпах, TaxiBJ — на 500; усі прогнозні кадри входять до підсумку. MSE і MAE обчислено без обрізання прогнозу. Для SSIM та PSNR прогноз і ціль обмежено інтервалом [0; 1], після чого показники усереднено за кадрами. Moving MNIST нормалізовано діленням на 255, TaxiBJ — перетворенням (x+1)/2; зворотного переходу до фізичних одиниць потоку немає. Кадрова сума MSE дорівнює піксельній MSE, помноженій на 4096 для Moving MNIST і на 2048 для двоканального TaxiBJ. Їх не слід змішувати з foreground MSE структурної матриці TRB.

Кожна ініціалізація Moving MNIST завершила 200 епох, або 250 000 оновлень; TaxiBJ — 100 епох, або 115 000 оновлень. Вибрано епохи 198 і 199 для Moving MNIST та 55 і 64 для TaxiBJ, рахуючи від одиниці. Відповідні валідаційні мінімуми становлять 0,010104 і 0,010279 та 0,0002652 і 0,0002662. Тест виконано один раз після вибору. Під час навчання змішана політика використовує адаптивний вибір із ймовірністю 0,75 та статичну квоту інакше; під час оцінювання вибір завжди адаптивний.

Близькість двох ініціалізацій підтримує відтворюваність працездатності композиції за цим протоколом, але не встановлює переваги над повністю навченим плоским або статичним контролем. Для Moving MNIST піксельна MSE від першого до десятого кадру зростає з 0,00698 до 0,01279 та з 0,00708 до 0,01299. Це прямий прогноз десяти кадрів, а не довгий рекурсивний прогін. Первинні артефакти містять криві валідації й похибки за горизонтами; покліпові тестові похибки не збережено, тому довірчі інтервали з агрегованих чисел не відновлюються.

![Прогнозні приклади для двох задач](../../../figures/restructured/hierarchy_integrated_result_slots.svg)

**Рисунок 4.16 – Прогнозні приклади для двох задач.** Перший кліп фіксованої тестової вибірки, ініціалізація 0, контрольна точка з мінімальною валідаційною втратою; приклади не відбирали за якістю. Угорі Moving MNIST: останній вхідний кадр та горизонти 1, 5 і 10. Унизу TaxiBJ: горизонти 1 та 4 для обох каналів потоку. Для цілі й прогнозу спільна шкала [0; 1]; абсолютна помилка також показана у шкалі [0; 1]. Канали TaxiBJ не є ідентичностями рухомих об’єктів. Приклади ілюструють локалізацію похибки; агреговану оцінку наведено в таблиці 4.21.

'''
    start=text.index('**Таблиця 4.21');end=text.index('<!-- evidence:COMP -->',start);text=text[:start]+quality+text[end:]
    # Actual horizon values are inserted directly from audited arrays.
    mm=[q['result']['test_metrics']['horizon_pixel_mse'] for q in a['quality'][:2]]
    text=text.replace('з 0,00708 до 0,01299',f"з {num(mm[1][0])} до {num(mm[1][-1])}")
    text=text.replace('Позначка «очікується» в таблицях і явно незаповнені рамки рисунків означають відсутність перевірених вимірювань.', 'Позначка «очікується» в таблицях означає відсутність перевірених вимірювань.')
    text=text.replace('Первинні артефакти містять криві валідації й похибки за горизонтами;', 'Криві валідації й похибки за горизонтами наведено на рисунку И.1;')
    (SRC/'05_CHAPTER_4.md').write_text(text,encoding='utf-8')
    p=SRC/'04_CHAPTER_3.md';text=(backup/p.name).read_text(encoding='utf-8').replace('CUDA-профіль очікується','повний CUDA-профіль наведено в §4.9.4; окремі операції ядра не виміряні');p.write_text(text,encoding='utf-8')
    p=SRC/'06_CONCLUSIONS.md';text=(backup/p.name).read_text(encoding='utf-8').replace('Повні ресурсні піки потребують повторного вимірювання після усунення утримання діагностичного графа, а якість — завершення незалежних повних навчань.','Повні навчання завершено у двох ініціалізаціях на кожній задачі: тестова піксельна MSE становить 0,009894–0,010035 для Moving MNIST і 0,0002763–0,0002792 для TaxiBJ у нормалізованих одиницях. Контрольований повний CUDA-профіль у BF16 показав зменшення allocated-піку реверсивного режиму приблизно на 12 % і 10 % відповідно; часові повтори TaxiBJ мають значний розкид. Це підтверджує прогнозну працездатність та обмежений ресурсний виграш композиції. Без повністю навченого зіставного контролю ці результати не встановлюють переваги адаптивного членства чи найкращої кінцевої якості.');p.write_text(text,encoding='utf-8')
    p=SRC/'08_APPENDICES.md';text=(backup/p.name).read_text(encoding='utf-8').replace('| RESOURCE | 3.7.7 / 4.9.4 | 4.19 | очікується: повторний профіль після контролю вивільнення графа |','| RESOURCE | 3.7.7 / 4.9.4 | 4.19 | повна модель: перевірено; операції ядра: очікується |').replace('| QUALITY | 3.7.7 / 4.9.5 | 4.21 | очікується |','| QUALITY | 3.7.7 / 4.9.5 | 4.21 | перевірено |');p.write_text(text,encoding='utf-8')
    invpath=ROOT/'docs/reviews/hierarchy_integration_2026-09-06/experiment_evidence_inventory.json';inv=read(invpath)
    artifact=(OUT/'audit.json').relative_to(ROOT).as_posix()
    for row in inv:
        if row['key'] not in ['RESOURCE','QUALITY']:continue
        row['artifact']=artifact;row['status']='перевірено' if row['key']=='QUALITY' else 'частково перевірено'
        for cell in row['result_slots']:
            accepted=row['key']=='QUALITY' or cell['scope']=='whole_model'
            cell.update(status='перевірено' if accepted else 'очікується',artifact=artifact if accepted else None)
            cell['reason']='Перевірені завершені артефакти' if accepted else 'Окремі операції ядра не виміряні повним профілем'
    write(invpath,inv)
    plot_examples();plot_curves(a)
    shutil.copy2(OUT/'learning_and_horizon_curves.png',FIG/'integrated_learning_horizon_curves.png')
    p=SRC/'08_APPENDICES.md'
    with p.open('a',encoding='utf-8') as f:
        f.write('\n\nКриві на рисунку И.1 доповнюють підсумкові оцінки таблиці 4.21: вони показують перебіг оптимізації та зміну похибки в межах прямого прогнозу. Кожна лінія відповідає окремій ініціалізації, без усереднення між навчаннями.\n\n![Криві навчання й тестові горизонти](../../../figures/restructured/integrated_learning_horizon_curves.png)\n\n**Рисунок И.1 – Валідаційні траєкторії та похибка за горизонтом.** Угорі — валідаційна піксельна MSE за епохами, логарифмічна вісь ординат; унизу — тестова піксельна MSE за горизонтом прямого прогнозу для вибраних контрольних точок. Moving MNIST та TaxiBJ мають окремі шкали; зображення не є зіставленням складності задач.\n')
    p=FIG/'integrated_figure_provenance.json';ledger=read(p)
    for row in ledger:
        if row.get('figure_id')=='4.16':
            asset=FIG/'hierarchy_integrated_result_slots.svg'
            row.update(asset_class='author-generated-artifact-derived',source=artifact,generator='scripts/integrate_integrated_evidence_20260911.py',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),bytes=asset.stat().st_size,caption_attribution='Ukrainian Figure 4.16; fixed first test examples, verified checkpoints, 2026-09-11')
    write(p,ledger)
def plot_examples():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(6,5,figsize=(8.3,7.8),layout='constrained')
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    for offset,ds in [(0,'mmnist'),(3,'taxibj')]:
        a=np.load(OUT/f'examples/{ds}.npz');x=a['inputs'][0];y=a['targets'][0];p=a['predictions'][0]
        specs=[(0,0),(4,0),(9,0)] if ds=='mmnist' else [(0,0),(3,0),(0,1),(3,1)]
        for i in range(3):
            axes[offset+i,0].axis('off')
            axes[offset+i,0].text(.5,.5,['Еталон','Прогноз','|Помилка|'][i],ha='center',va='center')
        if ds=='mmnist':
            specs=[(-1,0)]+specs
        for col,(h,ch) in enumerate(specs,1):
            for row in range(3):
                ax=axes[offset+row,col]
                if h<0:
                    if row==0:ax.imshow(x[-1,ch],cmap='gray',vmin=0,vmax=1)
                    else:ax.axis('off')
                else:
                    value=[y[h,ch],p[h,ch],np.abs(p[h,ch]-y[h,ch])][row]
                    ax.imshow(value,cmap='gray' if ds=='mmnist' else 'viridis',vmin=0,vmax=1)
                if row==0:ax.set_title('Останній вхід' if h<0 else f'h={h+1}'+(f', канал {ch}' if ds=='taxibj' else ''),fontsize=9)
        axes[offset,0].set_title('Moving MNIST' if ds=='mmnist' else 'TaxiBJ',fontsize=10)
    fig.savefig(FIG/'hierarchy_integrated_result_slots.svg');plt.close(fig)
    p=FIG/'hierarchy_integrated_result_slots.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
def plot_curves(a):
    fig,ax=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for q in a['quality']:
        col=0 if '_MM_' in q['run'] else 1
        logs=[json.loads(s) for s in (OUT/f"quality/{q['run']}/metrics.jsonl").read_text().splitlines()];v=[x for x in logs if x['event']=='validation'];label='Ініціалізація '+q['run'][-1]
        ax[0,col].plot([x['epoch']+1 for x in v],[x['val_loss'] for x in v],label=label)
        h=q['result']['test_metrics']['horizon_pixel_mse'];ax[1,col].plot(range(1,len(h)+1),h,'o-',label=label)
    for col,ds in enumerate(['Moving MNIST','TaxiBJ']):
        ax[0,col].set(title=ds,xlabel='Епоха',ylabel='Валідаційна pixel MSE',yscale='log');ax[0,col].legend()
        ax[1,col].set(xlabel='Горизонт прямого прогнозу',ylabel='Тестова pixel MSE');ax[1,col].legend()
    fig.savefig(OUT/'learning_and_horizon_curves.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
