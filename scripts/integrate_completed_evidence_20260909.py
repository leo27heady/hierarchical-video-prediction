"""Integrate verified measurements; retain rejected and unfinished result slots."""
from pathlib import Path
import json,re,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/dissertation/condensed_translated_humanized_ukrainian'
E=ROOT/'docs/reviews/completed_evidence_2026-09-09'
FIG=ROOT/'figures/restructured'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def number(v,n=5):return f'{v:.{n}f}'.replace('.',',')
def pair(v):return ' / '.join(number(x) for x in v)
def main():
    audit=read(E/'audit.json');rows=read(E/'all_final_development_metrics.json')['rows']
    audit['resource_accepted']=False
    audit['resource_exclusion']='Original saved-tensor hook retains autograd graph; growing baseline allocations invalidate total-peak comparisons. Requires corrected remeasurement.'
    audit['auditor_sha256']=hashlib.sha256((ROOT/'scripts/audit_completed_evidence_20260909.py').read_bytes()).hexdigest()
    (E/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    before=ROOT/'tmp/evidence_20260909/before'
    before.mkdir(parents=True,exist_ok=True)
    for p in SRC.glob('0*.md'):
        if not (before/p.name).exists():shutil.copy2(p,before/p.name)
    c4=(before/'05_CHAPTER_4.md').read_text(encoding='utf-8')
    def values(predicate):
        selected=sorted([r for r in rows if predicate(r)],key=lambda r:r['spec']['seed'])
        assert len(selected)==2,[r['experiment_id'] for r in selected]
        return [r['metrics']['horizon4_foreground_mse'] for r in selected]
    def policy(p):return values(lambda r:r['spec']['policy']==p and (r['spec']['family']=='ALC' or r['spec']['family']=='FRT' and r['context_tokens']==1792))
    def table(key,headers,data):
        nonlocal c4
        marker=f'<!-- evidence:{key} -->';start=c4.index(marker)
        match=re.search(r'(?m)^\|.*(?:\n\|.*)*',c4[start:]);assert match,key
        body='\n'.join('| '+' | '.join(row)+' |' for row in [headers,['---']*len(headers),*data])
        c4=c4[:start+match.start()]+body+c4[start+match.end():]
    def paragraph(prefix,text):
        nonlocal c4
        pattern=r'(?m)^'+re.escape(prefix)+r'[^\n]*'
        c4,n=re.subn(pattern,lambda m:text,c4);assert n==1,(prefix,n)
    alc=[('Плоска недавня','flat_recent'),('Перевага деталей','fine_heavy'),('Збалансована недавня','balanced_recent'),
         ('Перевага грубого рівня','coarse_heavy'),('Лінійна за віком','linear_age'),('Геометрична за віком','geometric_age'),('Орієнтована на досяжність','reach_optimized')]
    # Inspect actual policy spelling rather than infer it from a run ID.
    available={r['spec']['policy'] for r in rows}
    if 'reach_optimized' not in available:alc[-1]=(alc[-1][0],'reach_opt')
    table('ALC',['Політика','MSE h=4, ініц. 0','Ініц. 1'],[[label,*[number(v) for v in policy(p)]] for label,p in alc])
    paragraph('Поки сімейство не перевірено', 'Серед семи політик ALC найменшу середню помилку має геометричний розподіл за віком (0,01962), тоді як перевага грубого рівня дає 0,03227. Отже, сама широка підтримка не компенсує невдалого розподілу деталей. Для плоскої, лінійної та геометричної політик повторно використано контрольні точки FRT; вони не є додатковими незалежними навчаннями.')
    cex=[]
    for coarse in (0,4,8,16,32,64):
        v=values(lambda r:r['spec']['family']=='CEX' and r['spec']['allocation_args']['coarse_times']==coarse)
        cex.append(v)
    table('CEX',['Грубих позицій','Токени / досяжність','MSE h=4, 0','Ініц. 1'],[[str(c),f'{384+16*c} / {6+c}',*[number(v) for v in vs]] for c,vs in zip((0,4,8,16,32,64),cex)])
    paragraph('Відсутність подальшого зменшення', 'Розширення до 64 грубих позицій знижує середню помилку з 0,04528 до 0,04213, приблизно на 7%, за зростання пакета з 384 до 1 408 токенів. На проміжних розмірах зміни малі та не монотонні. Це обмежена користь доданої підтримки за її явної ресурсної ціни, а не доведена універсальна перевага довшого контексту.')
    sts={}
    for k in (64,32,16,8):
        for placement in ('recent','uniform','geometric'):
            sts[k,placement]=values(lambda r:r['spec']['family']=='STS' and r['spec']['allocation_args']['quota']==k and placement in r['spec']['allocation_args']['temporal_policy'])
    table('STS',['K / моментів','Недавнє, 0 / 1','Рівномірне, 0 / 1','Геометричне, 0 / 1'],[[f'{k} / {256//k}',*[pair(sts[k,p]) for p in ('recent','uniform','geometric')]] for k in (64,32,16,8)])
    paragraph('Кожна заповнена комірка', 'За сталої квоти найменшу середню помилку має K=16 з геометричним розміщенням у 16 моментах (0,01860). Чотири повні детальні моменти K=64 не є найкращими; проте подальше розрідження до K=8 також не покращує результат. План K=64 з недавнім розміщенням семантично збігається з орієнтованим на досяжність ALC; розбіжність повторних виконань не означає іншого механізму. Цей статичний контраст не встановлює переваги навчуваного членства (§4.9).')
    paragraph('За однакового бюджету додатковий масштаб', 'За однакового бюджету додатковий масштаб забирає токени в решти рівнів. За замороженим правилом §3.6.5 опорною політикою LVL обрано geometric_age, а для OVR — balanced_recent серед допустимих для всіх схем перекриття. Вибір виконано за розвитком до відкладеної перевірки.')
    lvl=values(lambda r:r['spec']['family']=='LVL')
    table('LVL',['Рівні','Фізичний розподіл','MSE h=4, 0 / 1'],[['1','28 × 64 деталей',pair(policy('flat_recent'))],['2','Геометрична ALC',pair(policy('geometric_age'))],['3','384 верхніх + 1 408 решти',pair(lvl)]])
    ovr=[policy('balanced_recent'),values(lambda r:r['spec']['family']=='OVR' and r['spec']['overlap']=='exclusive'),values(lambda r:r['spec']['family']=='OVR' and r['spec']['overlap']=='residual')]
    table('OVR',['Представлення','MSE h=4, ініц. 0','Ініц. 1'],[[label,*[number(v) for v in vals]] for label,vals in zip(['Вкладене','Взаємовиключне','Залишкове'],ovr)])
    paragraph('Нульові або змішані ефекти', 'Три рівні дають середню помилку 0,02553 проти 0,01962 у дворівневої опори: додавання масштабу за цього бюджету не допомогло. У OVR вкладена схема (0,02371) краща за взаємовиключну (0,02639) та залишкову (0,02644) у середньому за двома ініціалізаціями. Залишкова схема має змішаний напрям щодо взаємовиключної. Отже, усунення часової повторюваності або додавання масштабу не гарантує виграшу.')
    roles=audit['roles'];comparisons=audit['comparisons']
    role_names=[('Плоска, B=1792','best_flat_1792'),('Геометрична ALC','best_hierarchy_1792'),('STS K16, геометрична','memory_pareto_within_5pct')]
    table('CONF',['Роль','MSE h=4, 0 / 1','Δ плоска − роль, 0 / 1'],[[label,pair([s['h4'] for s in roles[key]['seeds']]),'Опора' if key=='best_flat_1792' else pair([c['flat_minus_hierarchy'] for c in comparisons[key]])] for label,key in role_names])
    ci=[]
    for label,key in role_names[1:]:
        ci.append(label+': '+ '; '.join(f"ініц. {seed} [{number(c['adjusted_percentile_interval'][0])}; {number(c['adjusted_percentile_interval'][1])}]" for seed,c in enumerate(comparisons[key])))
    paragraph('Допуск відбору 5%', 'Для чотирьох описових парних контрастів застосовано 10 000 повторних вибірок кліпів і поправку Бонферроні на сімейний рівень 0,05. Скориговані процентильні інтервали: '+'. '.join(ci)+'. Геометрична ALC має додатний інтервал лише за ініціалізації 0; STS K16 — за обох. Ресурсна роль також має B=1792, тому її назва не означає фактичної економії токенів. Допуск відбору 5% не є межею статистичної не-гіршості. Порожні маски оброблялися нульовим чисельником і знаменником не меншим за один; їхній окремий лічильник у первинних підтверджувальних масивах не збережено.')
    val={r['run_id']:r['horizon4_foreground_mse'] for r in audit['validation300']}
    table('VAL300',['Фіксована конфігурація','Кліпів','MSE h=4, 0 / 1'],[[label,'300',pair([val[s['run_id']] for s in roles[key]['seeds']])] for label,key in role_names])
    paragraph('Оцінюється не більш як три конфігурації', 'Усі три заморожені конфігурації оцінено без повторного відбору. STS K16 зберігає нижчу помилку за обох ініціалізацій; геометрична ALC також нижча за плоску на цих 300 кліпах, хоча у відкладеній перевірці її напрям не відтворився для ініціалізації 1. Абсолютні значення не ототожнюються з відбірковими: тут використано кожний кліп один раз і середнє покліпових похибок, тоді як розвитковий журнал агрегує пікселі переднього плану в пакетах із 64 вибірками з поверненням.')
    mech=[('Плоска','best_flat_1792'),('Геометрична ALC','best_hierarchy_1792'),('Перевага грубого рівня','failure_hierarchy_1792')]
    table('MECH',['Роль','MSE повторного входу, 0 / 1','Ентропія уваги, 0 / 1'],[[label,pair([s['reentry_foreground_mse'] for s in roles[key]['seeds']]),pair([s['attention_entropy'] for s in roles[key]['seeds']])] for label,key in mech])
    paragraph('Приклади добираються за оголошеним правилом', 'Помилка повторного входу агрегована за 76 позначеними записами «кліп–горизонт» із 2 000; це не 76 незалежних кліпів. Ентропія уваги нормована та усереднена за шарами одного фіксованого пробного вікна, а не за всіма 500 кліпами. Груба політика має меншу помилку повторного входу, але гіршу загальну h=4-помилку за геометричну ALC; це вказує на різні вимоги до підтримки та деталізації. У STS K16 помилка повторного входу становить 0,03690 / 0,04538. Увага є діагностикою, а не причинним доказом використання історії.')
    table('PARITY',['Умова','Вихід / градієнти','Статус'],[['CPU FP32','Ненульова проєкція; залежності','Перевірено'],['CUDA FP32, обидві задачі','atol 3×10⁻⁵; rtol 4×10⁻⁴','Пройдено'],['CUDA BF16, обидві задачі','atol 0,01; rtol 0,08','Пройдено']])
    paragraph('CPU-перевірки не дають', 'На CUDA зіставлено звичайне ядро, контрольні точки та реверсивне ядро за однакових ваг і входів, пакетів 8 для Moving MNIST та 16 для TaxiBJ. Виходи, входові й активні параметричні градієнти пройшли перевірку з наведеними допусками. Це допуски тесту, а не виміряні максимальні розбіжності: первинний звіт зберіг факт проходження, але не їхні максимуми. Використано фіксовану ненульову проєкцію для числової проби, а не навчений прогноз як свідчення якості.')
    paragraph('Основна таблиця стосується', 'Ресурсне зіставлення ще не прийнято: контроль артефактів виявив наростання базових алокацій між режимами через утримання графа діагностичним перехоплювачем збережених тензорів. Такі повні піки не є зіставними, тому числові комірки залишено незаповненими до повторного вимірювання з перевіркою вивільнення. Це не скасовує незалежної перевірки паритету. Повний профіль має враховувати ваги, градієнти, оптимізатор, грубу обхідну карту, індекси й фактичні типи: за BF16-autocast упаковані токени у перевіреній реалізації залишаються FP32. Аналітичне скорочення кількості токенів не підміняє вимірювання байтів.')
    screen_names=[('Moving MNIST, адаптивний','SCREEN_MM_FULL'),('TaxiBJ, адаптивний','SCREEN_TB_FULL'),('Moving MNIST, плоский','SCREEN_MM_FLAT'),('Moving MNIST, статичний','SCREEN_MM_STATIC')]
    table('SCREEN',['Задача / пакет','Валідаційна MSE / норма градієнта','Оновлень/с / статус'],[[label,number(audit['screens'][key]['validation_loss'],6)+' / '+number(audit['screens'][key]['last_grad_norm'],3),number(audit['screens'][key]['updates_per_second'],2)+' / 200 кроків'] for label,key in screen_names])
    paragraph('За результатами цих перевірок', 'Усі чотири короткі траєкторії завершили 200 оновлень із скінченними втратами й ненульовими градієнтами; тестову вибірку не оцінювали. У таблиці наведено повну валідаційну MSE, останню записану норму градієнта та медіанну швидкість без першого інтервалу. Графіки навчання, точність BF16, розбиття й правило вибору контрольної точки зафіксовано для чотирьох повних навчань. Вони перевіряють прогнозну працездатність композиції; близькі втрати коротких контролів не доводять рівності кінцевої якості.')
    c4=re.sub(r'\*\*Рисунок 4\.13[^\n]*', '**Рисунок 4.13 – Залежність помилки від розподілу й геометрії контексту.** ALC і CEX показано для обох ініціалізацій; дві карти STS мають спільну шкалу. K означає число детальних комірок на момент, а число моментів дорівнює 256/K. Це розвиткова MSE переднього плану на горизонті 4 за 30 000 оновлень; вона не є відкладеною оцінкою.', c4)
    c4=re.sub(r'\*\*Рисунок 4\.14[^\n]*', '**Рисунок 4.14 – Помилка за горизонтом і парні відкладені контрасти.** Угорі середні покліпові похибки на 500 захищених кліпах, усереднені за двома ініціалізаціями; внизу окремі парні h=4-різниці та інтервали з поправкою на чотири контрасти. Додатна різниця сприятлива для ієрархічної політики. Горизонти 1–4 є прямим прогнозом, а не 64-кадровою самоподачею.', c4)
    (SRC/'05_CHAPTER_4.md').write_text(c4,encoding='utf-8')
    conclusion=(before/'06_CONCLUSIONS.md').read_text(encoding='utf-8')
    paragraph_add='У структурному зіставленні за 1 792 токенів найменшу розвиткову помилку має геометричне розміщення 16 деталей у кожному з 16 моментів. На 500 захищених кліпах воно зменшує h=4-помилку щодо плоскої опори на 0,01169 та 0,01026 за двома ініціалізаціями; обидва скориговані інтервали додатні. Геометрична ALC не відтворює переваги за другою ініціалізацією. Додавання третього рівня та заміна вкладеного подання взаємовиключним або залишковим не дали середнього виграшу. Отже, підтримку історії слід поєднувати з відповідним розміщенням деталей, а не максимізувати число рівнів.'
    conclusion=conclusion.replace('Отже, загальний ресурс і конкретний розподіл потрібно інтерпретувати разом.','Отже, загальний ресурс і конкретний розподіл потрібно інтерпретувати разом.\n\n'+paragraph_add)
    conclusion=conclusion.replace('вони не замінюють зіставних вимірювань CUDA-пам’яті й якості.','паритет також пройдено на CUDA у FP32 і BF16. Чотири 200-крокові траєкторії підтвердили скінченні втрати та дієві градієнти без оцінювання тестової вибірки. Повні ресурсні піки потребують повторного вимірювання після усунення утримання діагностичного графа, а якість — завершення незалежних повних навчань.')
    (SRC/'06_CONCLUSIONS.md').write_text(conclusion,encoding='utf-8')
    inventory=read(ROOT/'docs/reviews/hierarchy_integration_2026-09-06/experiment_evidence_inventory.json')
    accepted={'FRT','ALC','CEX','STS','LVL','OVR','CONF','MECH','VAL300','PARITY','SCREEN'}
    for group in inventory:
        for cell in group['result_slots']:
            if group['key'] in accepted:
                cell.update(status='перевірено',artifact='docs/reviews/completed_evidence_2026-09-09/audit.json')
            elif group['key']=='RESOURCE':cell['reason']='Повторне вимірювання: відхилено профіль з утриманням графа'
    (ROOT/'docs/reviews/hierarchy_integration_2026-09-06/experiment_evidence_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Scientific figures replace the two corresponding empty frames in place.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    labels=['Плоска','Детальна','Збалансована','Груба','Лінійна','Геометрична','Досяжність']
    for seed,marker in [(0,'o'),(1,'x')]:
        axes[0,0].plot([policy(p)[seed] for _,p in alc],range(7),marker,ls='none',label=f'Ініц. {seed}')
        axes[0,1].plot([0,4,8,16,32,64],[v[seed] for v in cex],marker+'-',label=f'Ініц. {seed}')
    axes[0,0].set_yticks(range(7),labels);axes[0,0].invert_yaxis();axes[0,0].set(title='ALC: сталий бюджет 1 792',xlabel='MSE переднього плану h=4');axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='CEX: розширення підтримки',xlabel='Грубих позицій',ylabel='MSE переднього плану h=4');axes[0,1].legend(fontsize=8)
    for seed in (0,1):
        ax=axes[1,seed];matrix=np.array([[sts[k,p][seed] for p in ('recent','uniform','geometric')] for k in (64,32,16,8)])
        im=ax.imshow(matrix,vmin=.018,vmax=.038,cmap='YlOrRd',aspect='auto')
        ax.set_xticks(range(3),['Недавнє','Рівномірне','Геометричне']);ax.set_yticks(range(4),['64 / 4','32 / 8','16 / 16','8 / 32'])
        ax.set(title=f'STS: ініціалізація {seed}',ylabel='K / моментів')
        for i in range(4):
            for j in range(3):ax.text(j,i,number(matrix[i,j],3),ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=list(axes[1]),label='MSE h=4',shrink=.8)
    fig.savefig(FIG/'hierarchy_structural_result_slots.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8.5,6.4),layout='constrained')
    for label,key in [*role_names,('Груба ALC','failure_hierarchy_1792')]:
        vals=[np.load(E/'confirmation'/f"{s['run_id']}.npz")['foreground_mse'].mean(axis=0) for s in roles[key]['seeds']]
        axes[0].plot(range(1,5),np.mean(vals,axis=0),'o-',label=label)
    axes[0].set(xticks=range(1,5),xlabel='Горизонт прямого прогнозу',ylabel='MSE переднього плану');axes[0].legend(fontsize=8,ncol=2)
    labels=[];y=0
    for label,key in role_names[1:]:
        for seed,c in enumerate(comparisons[key]):
            lo,hi=c['adjusted_percentile_interval'];v=c['flat_minus_hierarchy']
            axes[1].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4)
            labels.append(f'{label}, ініц. {seed}');y+=1
    axes[1].axvline(0,color='black',lw=1);axes[1].set(yticks=range(4),yticklabels=labels,xlabel='MSE плоскої − MSE політики, h=4');axes[1].invert_yaxis()
    fig.savefig(FIG/'hierarchy_diagnostic_result_slots.svg');plt.close(fig)
    for name in ('hierarchy_structural_result_slots.svg', 'hierarchy_diagnostic_result_slots.svg'):
        path = FIG / name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    print('Integrated verified tables, conclusions, inventory and two scientific figures.')

if __name__=='__main__':main()
