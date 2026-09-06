"""Correct Figure 3.20 and register the completed-run diagnostic figure."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]


def main():
    fig, ax=plt.subplots(figsize=(10,11))
    ax.set(xlim=(0,10),ylim=(0,11)); ax.axis('off')
    def box(x,y,w,h,text,color='#e7eff6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=color,ec='#34495e'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(points,color='#34495e',style='-'):
        for a,b in zip(points[:-2],points[1:-1]): ax.plot([a[0],b[0]],[a[1],b[1]],color=color,ls=style)
        ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='->',color=color,linestyle=style,lw=1.6))
    ax.text(5,10.7,'Інтегрований предиктор: значення та обхідні шляхи',ha='center',fontsize=14,weight='bold')
    box(2.8,9.5,4.4,.65,'Спостережувані кадри  B × T × C × H × W')
    box(2.8,8.25,4.4,.75,'Спільний згортковий кодувальник\nдетальна сітка: крок 4; груба: крок 8')
    arrow([(5,9.5),(5,9)])
    box(.9,6.85,3.2,.8,'Детальні кандидати\nT × H × W / 16', '#fff0d9')
    box(5.8,6.85,3.1,.8,'Повна груба карта\nT × H × W / 64', '#e3f3e7')
    arrow([(4,8.25),(2.5,7.65)]); arrow([(6,8.25),(7.3,7.65)])
    box(.9,5.3,3.9,1,'Маршрутизатор\nзалишкова енергія + зміна +\nнавчувана оцінка; квота K', '#fff0d9')
    arrow([(2.5,6.85),(2.5,6.3)])
    arrow([(6,6.85),(4.8,5.85)])
    ax.text(5.2,6.5,'груба опора',fontsize=9,rotation=-35)
    box(.9,3.95,3.9,.8,'gather K деталей + усі грубі токени\nспільні ознаки оцінки та координат')
    arrow([(2.7,5.3),(2.7,4.75)])
    arrow([(7.8,6.85),(7.8,4.65),(4.8,4.35)])
    box(.9,2.5,3.9,.8,'Transformer-ядро\nзвичайне / контрольні точки / реверсивне','#fde4e0')
    arrow([(2.7,3.95),(2.7,3.3)])
    box(5.8,2.5,3.1,1,'Майбутні запити\nостання груба карта +\nкоординати майбутнього','#e3f3e7')
    arrow([(8.9,7.25),(9.35,7.25),(9.35,3),(8.9,3)],style='--')
    box(2.6,1.05,4.8,.8,'Cross-attention → збільшення ×8\nзалишок + останній спостережуваний кадр')
    arrow([(2.7,2.5),(4,1.85)]); arrow([(7.3,2.5),(6,1.85)])
    arrow([(2.8,9.85),(.35,9.85),(.35,1.45),(2.6,1.45)],color='#986522',style='--')
    ax.text(.18,5.6,'Останній кадр: залишковий обхід',rotation=90,ha='center',fontsize=9,color='#986522')
    arrow([(5,1.05),(5,.65)])
    ax.text(5,.4,'Повнокадровий прогноз  B × Tₒᵤₜ × C × H × W',ha='center',fontsize=11)
    fig.tight_layout()
    path=ROOT/'figures/restructured/ch3_fig3_20_packed_integrated_architecture.svg'
    fig.savefig(path); fig.savefig(path.with_suffix('.png'),dpi=180); plt.close(fig)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    ledger=ROOT/'figures/restructured/integrated_figure_provenance.json'
    records=json.loads(ledger.read_text(encoding='utf-8'))
    row=next(x for x in records if x['file']==path.relative_to(ROOT).as_posix())
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
               generator='scripts/refine_review_figures.py',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               modification_status='corrected coarse router and decoder/residual bypasses; 2026-09-05')
    ledger.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
