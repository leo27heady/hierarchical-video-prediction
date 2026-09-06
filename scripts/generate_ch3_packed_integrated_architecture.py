from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "restructured" / "ch3_fig3_20_packed_integrated_architecture.svg"


SVG = r'''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="900" viewBox="0 0 1800 900">
  <defs>
    <style>
      .title { font: 700 32px 'DejaVu Sans', sans-serif; fill: #172033; }
      .box-title { font: 700 20px 'DejaVu Sans', sans-serif; fill: #172033; }
      .body { font: 16px 'DejaVu Sans', sans-serif; fill: #334155; }
      .small { font: 14px 'DejaVu Sans', sans-serif; fill: #475569; }
      .boundary { font: 700 16px 'DejaVu Sans', sans-serif; fill: #7c2d12; }
      .arrow { stroke: #334155; stroke-width: 3; fill: none; marker-end: url(#arrow); }
      .dashed { stroke: #64748b; stroke-width: 2.5; stroke-dasharray: 9 7; fill: none; marker-end: url(#arrow); }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>
    </marker>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.15"/>
    </filter>
  </defs>

  <rect width="1800" height="900" fill="#f8fafc"/>
  <text x="900" y="52" text-anchor="middle" class="title">Фізично упакована інтегрована архітектура</text>

  <rect x="35" y="150" width="185" height="120" rx="14" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="128" y="188" text-anchor="middle" class="box-title">Вхід OpenSTL</text>
  <text x="128" y="220" text-anchor="middle" class="body">B × Tᵢₙ × C × H × W</text>
  <text x="128" y="248" text-anchor="middle" class="small">спостережувані кадри</text>

  <rect x="270" y="135" width="200" height="150" rx="14" fill="#e0f2fe" stroke="#0284c7" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="370" y="176" text-anchor="middle" class="box-title">Спільна згорткова</text>
  <text x="370" y="202" text-anchor="middle" class="box-title">піраміда</text>
  <text x="370" y="235" text-anchor="middle" class="body">stride 4 → fine</text>
  <text x="370" y="261" text-anchor="middle" class="body">stride 8 → coarse</text>

  <rect x="525" y="90" width="185" height="105" rx="14" fill="#fef3c7" stroke="#d97706" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="618" y="128" text-anchor="middle" class="box-title">Fine-кандидати</text>
  <text x="618" y="159" text-anchor="middle" class="body">TᵢₙHW / 16</text>
  <text x="618" y="182" text-anchor="middle" class="small">просторово-часовий об’єм</text>

  <rect x="525" y="245" width="185" height="105" rx="14" fill="#dcfce7" stroke="#16a34a" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="618" y="283" text-anchor="middle" class="box-title">Coarse-основа</text>
  <text x="618" y="314" text-anchor="middle" class="body">TᵢₙHW / 64</text>
  <text x="618" y="337" text-anchor="middle" class="small">повне покриття</text>

  <rect x="760" y="75" width="210" height="145" rx="14" fill="#ffedd5" stroke="#ea580c" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="865" y="112" text-anchor="middle" class="box-title">Маршрутизатор</text>
  <text x="865" y="144" text-anchor="middle" class="body">залишкова енергія</text>
  <text x="865" y="170" text-anchor="middle" class="body">часова зміна</text>
  <text x="865" y="196" text-anchor="middle" class="body">навчувана поправка</text>

  <rect x="1015" y="90" width="170" height="105" rx="14" fill="#fed7aa" stroke="#c2410c" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="1100" y="128" text-anchor="middle" class="box-title">Top-k + gather</text>
  <text x="1100" y="158" text-anchor="middle" class="body">рівно K деталей</text>
  <text x="1100" y="181" text-anchor="middle" class="small">мінімум на кадр</text>

  <rect x="760" y="265" width="210" height="120" rx="14" fill="#ede9fe" stroke="#7c3aed" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="865" y="302" text-anchor="middle" class="box-title">Фізичні метадані</text>
  <text x="865" y="334" text-anchor="middle" class="body">x, y, час, sₓ, sᵧ</text>
  <text x="865" y="360" text-anchor="middle" class="body">рівень 0 / 1 / 2</text>

  <rect x="1015" y="265" width="170" height="120" rx="14" fill="#ddd6fe" stroke="#6d28d9" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="1100" y="302" text-anchor="middle" class="box-title">Fourier-ознаки</text>
  <text x="1100" y="334" text-anchor="middle" class="body">sin / cos</text>
  <text x="1100" y="360" text-anchor="middle" class="small">+ embedding рівня</text>

  <rect x="1235" y="150" width="175" height="130" rx="14" fill="#ccfbf1" stroke="#0f766e" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="1323" y="187" text-anchor="middle" class="box-title">PackedContext</text>
  <text x="1323" y="219" text-anchor="middle" class="body">усі coarse</text>
  <text x="1323" y="245" text-anchor="middle" class="body">+ K fine</text>
  <text x="1323" y="268" text-anchor="middle" class="small">+ метадані маршруту</text>

  <rect x="1450" y="125" width="310" height="180" rx="16" fill="#fee2e2" stroke="#dc2626" stroke-width="3.5" filter="url(#shadow)"/>
  <text x="1605" y="167" text-anchor="middle" class="box-title" style="font-size:18px">Реверсивне Transformer-ядро</text>
  <text x="1605" y="202" text-anchor="middle" class="body">адитивне зв’язування F / G</text>
  <text x="1605" y="230" text-anchor="middle" class="body">звичайний або reversible backward</text>
  <text x="1605" y="260" text-anchor="middle" class="boundary">єдина реверсивна межа</text>
  <text x="1605" y="287" text-anchor="middle" class="small">та сама пряма функція</text>

  <rect x="820" y="555" width="250" height="125" rx="14" fill="#e0e7ff" stroke="#4f46e5" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="945" y="594" text-anchor="middle" class="box-title">Майбутні coarse-запити</text>
  <text x="945" y="627" text-anchor="middle" class="body">TₒᵤₜHW / 64</text>
  <text x="945" y="655" text-anchor="middle" class="small">додатний час, рівень 2</text>

  <rect x="1160" y="530" width="270" height="175" rx="14" fill="#cffafe" stroke="#0891b2" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="1295" y="570" text-anchor="middle" class="box-title">Декодер майбутнього</text>
  <text x="1295" y="605" text-anchor="middle" class="body">cross-attention до пам’яті</text>
  <text x="1295" y="633" text-anchor="middle" class="body">3 × збільшення роздільності</text>
  <text x="1295" y="661" text-anchor="middle" class="body">повнокадровий залишок</text>
  <text x="1295" y="688" text-anchor="middle" class="small">нереверсивний модуль</text>

  <rect x="1510" y="555" width="240" height="125" rx="14" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)"/>
  <text x="1630" y="594" text-anchor="middle" class="box-title">Вихід OpenSTL</text>
  <text x="1630" y="627" text-anchor="middle" class="body">B × Tₒᵤₜ × C × H × W</text>
  <text x="1630" y="655" text-anchor="middle" class="small">останній кадр + залишок</text>

  <path class="arrow" d="M220 210 H270"/>
  <path class="arrow" d="M470 185 C495 185 495 143 525 143"/>
  <path class="arrow" d="M470 235 C495 235 495 297 525 297"/>
  <path class="arrow" d="M710 143 H760"/>
  <path class="arrow" d="M970 143 H1015"/>
  <path class="dashed" d="M618 195 V220 H865 V265"/>
  <path class="dashed" d="M618 350 V410 H865 V385"/>
  <path class="arrow" d="M970 325 H1015"/>
  <path class="arrow" d="M1185 143 H1210 V185 H1235"/>
  <path class="arrow" d="M710 297 V445 H1210 V245 H1235"/>
  <path class="dashed" d="M1185 325 H1210 V260 H1235"/>
  <path class="arrow" d="M1410 215 H1450"/>
  <path class="arrow" d="M1605 305 V465 H1400 V530"/>
  <path class="dashed" d="M618 350 V490 H945 V555"/>
  <path class="arrow" d="M1070 618 H1160"/>
  <path class="arrow" d="M1430 618 H1510"/>

  <rect x="35" y="770" width="1715" height="85" rx="12" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>
  <text x="65" y="804" class="body">Суцільні стрілки — значення токенів; пунктирні — фізичні метадані або опорна груба геометрія.</text>
  <text x="65" y="834" class="body">Поза червоною межею активації зберігаються звичайно; виграш повної GPU-пам’яті встановлюється лише вимірюванням.</text>
</svg>
'''


def main() -> None:
    # The historical SVG above is retained as source history. Use the corrected
    # generator so this public entry point cannot restore missing bypasses.
    from refine_review_figures import main as corrected
    corrected()


if __name__ == "__main__":
    main()
