"""Remove editorial scaffolding and normalize the dissertation sources to English.

This is an idempotent assembly transform for the authoritative Markdown sources. It does not
alter numerical claims, citations, tables, equations, or figure paths.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "dissertation" / "rewrite"

FILES = [
    "00_FRONT_MATTER.md",
    "01_VSTUP.md",
    "02_ROZDIL1.md",
    "03_ROZDIL2.md",
    "04_ROZDIL3.md",
    "05_ROZDIL4.md",
    "06_VYSNOVKY.md",
]

START_MARKERS = {
    "02_ROZDIL1.md": "# РОЗДІЛ 1 —",
    "03_ROZDIL2.md": "# РОЗДІЛ 2 —",
    "04_ROZDIL3.md": "# РОЗДІЛ 3 —",
    "05_ROZDIL4.md": "# РОЗДІЛ 4 —",
    "06_VYSNOVKY.md": "# ВИСНОВКИ",
}

END_MARKERS = {
    "02_ROZDIL1.md": "\n## Figure inventory",
    "03_ROZDIL2.md": "\n## Figure inventory",
    "04_ROZDIL3.md": "\n## Dispositions of the OLD",
    "05_ROZDIL4.md": "\n## Working notes",
    "06_VYSNOVKY.md": "\n## Working notes",
}

EXACT_HEADINGS = {
    "# 00 — FRONT MATTER (rewrite, phase 1)": "# Front Matter",
    "# 01 — ВСТУП (rewrite, phase 1)": "# Introduction",
    "# РОЗДІЛ 1 — СИСТЕМНИЙ АНАЛІЗ ПРОЦЕСІВ ІЄРАРХІЧНОЇ ГЕНЕРАЦІЇ ВІДЕОДАНИХ ТА КРИТЕРІЇВ ЇХ КОНСИСТЕНТНОСТІ":
        "# Chapter 1 — Systems Analysis of Hierarchical Video Generation and Its Consistency Criteria",
    "# РОЗДІЛ 2 — МАТЕМАТИЧНІ МЕТОДИ ОЦІНЮВАННЯ НЕВИЗНАЧЕНОСТІ ТА РЕСУРСОЕФЕКТИВНОГО МОДЕЛЮВАННЯ БАГАТОКРОКОВИХ ВІДЕОПРОЦЕСІВ":
        "# Chapter 2 — Mathematical Methods for Uncertainty Estimation and Resource-Efficient Modelling of Multi-Step Video Processes",
    "# РОЗДІЛ 3 — ІЄРАРХІЧНА МАТЕМАТИЧНА МОДЕЛЬ ГЕНЕРАЦІЇ ВІДЕОПОСЛІДОВНОСТЕЙ ІЗ ДОВГОТРИВАЛИМ БАГАТОПОТОКОВИМ КОНТЕКСТОМ":
        "# Chapter 3 — A Hierarchical Mathematical Model for Video Generation with Long Multi-Stream Context",
    "# РОЗДІЛ 4 — ЕКСПЕРИМЕНТАЛЬНЕ ДОСЛІДЖЕННЯ ТА ПРАКТИЧНЕ ЗАСТОСУВАННЯ":
        "# Chapter 4 — Experimental Evaluation and Practical Application",
    "# ВИСНОВКИ": "# General Conclusions",
    "## 2. АНОТАЦІЯ (UA) / ABSTRACT (EN) — canonical replacement text": "## 2. Abstract",
    "## replacement fields": "## 1. Title Page",
    "### new structure per DISS_DIGEST §6": "### Chapter-by-Chapter Summary",
    "## 3. СПИСОК ПУБЛІКАЦІЙ ЗДОБУВАЧА — replacement note": "## 3. Applicant's Publications",
    "## 4. ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ ТА СКОРОЧЕНЬ — replacement list":
        "## 4. Symbols and Abbreviations",
    "## 5. ЗМІСТ (table of contents skeleton — new chapter titles per DISS_DIGEST §6)":
        "## 5. Table of Contents",
    "## Обґрунтування вибору теми дослідження (Justification / relevance)":
        "## Justification and Relevance",
    "## Зв'язок роботи з науковими програмами, планами, темами":
        "## Relationship to Research Programmes, Plans and Topics",
    "## Мета і задачі дослідження": "## Aim and Research Tasks",
    "## Об'єкт і предмет дослідження": "## Object and Subject of Research",
    "## Методи дослідження": "## Research Methods",
    "## Наукова новизна отриманих результатів": "## Scientific Novelty",
    "## Положення, що виносяться на захист": "## Propositions Submitted for Defence",
    "## Практичне значення отриманих результатів": "## Practical Significance",
    "## Особистий внесок здобувача": "## Applicant's Personal Contribution",
    "## Апробація результатів дисертації": "## Validation and Presentation of Results",
    "## Публікації": "## Publications",
    "## Структура та обсяг дисертації": "## Dissertation Structure and Length",
    "## 2.1 Метод оцінювання просторово-часової невизначеності відеоданих на основі ентропійних показників":
        "## 2.1 Entropy-Based Estimation of Spatio-Temporal Video Uncertainty",
    "## 2.2 Забезпечення структурної когерентності даних за допомогою багатовимірного позиційного кодування":
        "## 2.2 Structural Coherence through Multidimensional Positional Encoding",
    "## 2.3 Метод оптимізації обчислювальних ресурсів складної системи при моделюванні багатокрокових часових процесів":
        "## 2.3 Resource Optimization for Multi-Step Temporal Modelling",
    "## 2.4 Математичне обґрунтування часонезалежного зворотного розповсюдження помилки на основі оборотних перетворень":
        "## 2.4 Time-Agnostic Backpropagation through Reversible Transformations",
    "### Узагальнення на стохастичні рекурентності та межі застосовності оборотних обчислень":
        "### Extension to Stochastic Recurrences and Limits of Reversible Computation",
    "## 2.5 Висновки до розділу": "## 2.5 Chapter Conclusions",
    "## 3.1 Формалізація ієрархічного прихованого простору станів":
        "## 3.1 Formalization of the Hierarchical Latent State Space",
    "### 3.1.1 Дворівнева декомпозиція стану та заморожений автокодувальник":
        "### 3.1.1 Two-Level State Decomposition and Frozen Autoencoder",
    "### 3.1.2 Передача контекстних сигналів між рівнями: від перехресної уваги до спільної уваги":
        "### 3.1.2 Cross-Level Context Transfer: from Cross-Attention to Joint Attention",
    "### 3.1.3 Односпрямоване керування з відсіканням градієнтних зв'язків":
        "### 3.1.3 One-Way Control with Detached Gradient Paths",
    "## 3.2 Генеративне ядро: віконна дифузія у прихованому просторі":
        "## 3.2 Generative Core: Windowed Latent Diffusion",
    "### 3.2.1 Токенна послідовність та покомпонентне кондиціонування":
        "### 3.2.1 Token Sequence and Component-Wise Conditioning",
    "### 3.2.2 Трансформерний остов з адаптивною нормалізацією":
        "### 3.2.2 Transformer Backbone with Adaptive Normalization",
    "### 3.2.3 Віконна дифузійна цільова функція (v-параметризація)":
        "### 3.2.3 Windowed Diffusion Objective (v-Parameterization)",
    "### 3.2.4 Детерміноване вибіркове відновлення (DDIM)":
        "### 3.2.4 Deterministic Sampling (DDIM)",
    "### 3.2.5 Регресійне контрольне ядро": "### 3.2.5 Regression Control Head",
    "## 3.3 Алгоритм авторегресійного розгортання з кільцевою буферною пам'яттю":
        "## 3.3 Autoregressive Rollout with Ring-Buffer Memory",
    "### 3.3.1 Рекурсія розгортання": "### 3.3.1 Rollout Recurrence",
    "### 3.3.2 Часова маршрутизація: чесне відображення блочно-каузального принципу":
        "### 3.3.2 Temporal Routing and Block-Causal Semantics",
    "### 3.3.3 Аналіз пам'яті та обчислювальної складності":
        "### 3.3.3 Memory and Computational Complexity",
    "### 3.3.4 Конфігурації системи та контрольні варіанти зі зрівняним бюджетом":
        "### 3.3.4 System Configurations and Matched-Budget Controls",
    "### 3.3.5 Узагальнення на сімейство ієрархічних конфігурацій":
        "### 3.3.5 Generalization to a Family of Hierarchical Configurations",
    "## 3.4 Тестовий стенд горизонту пам'яті та протокол оцінювання зі стратифікацією за повторним входженням":
        "## 3.4 Memory-Horizon Testbed and Re-Entry-Stratified Evaluation",
    "### 3.4.1 Циклічний світ з інерційним панорамуванням":
        "### 3.4.1 Cyclic World with Inertial Panning",
    "### 3.4.2 Точні мітки складності повторного входження":
        "### 3.4.2 Exact Re-Entry Difficulty Labels",
    "### 3.4.3 Метрики: маскований передній план, покадрово, після декодування":
        "### 3.4.3 Decoded Frame, Foreground and Object Metrics",
    "### 3.4.4 Статистичний протокол, опорні рівні та межі":
        "### 3.4.4 Statistical Protocol, References and Boundaries",
    "## 3.5 Висновки до розділу": "## 3.5 Chapter Conclusions",
    "## 4.1 Інформаційна база, апаратне середовище та протокол вимірювань":
        "## 4.1 Data, Hardware and Measurement Protocol",
    "### 4.1.1 Родина тестових стендів: дві робочі точки":
        "### 4.1.1 Testbed Family: Two Operating Points",
    "### 4.1.2 Синтетичний набір методу оборотних перетворень ([A2])":
        "### 4.1.2 Synthetic Dataset of the Reversible-Transformation Study ([A2])",
    "### 4.1.3 Міст до стандартного стенда OpenSTL/MMNIST":
        "### 4.1.3 Bridge to the OpenSTL/Moving-MNIST Benchmark",
    "### 4.1.4 Апаратне та програмне середовище":
        "### 4.1.4 Hardware and Software Environment",
    "### 4.1.5 Статистичний протокол та загрози валідності":
        "### 4.1.5 Statistical Protocol and Threats to Validity",
    "## 4.2 Механізмне дослідження за зрівняних обчислень: вибір генеративного ядра":
        "## 4.2 Matched-Compute Mechanism Study: Selecting the Generative Core",
    "### 4.2.1 Спільний випробувальний стенд механізмів":
        "### 4.2.1 Shared Mechanism Testbed",
    "### 4.2.2 Одновкроковий режим оракульного покриття: дифузія виграє кожну колонку":
        "### 4.2.2 Single-Step Oracle-Coverage Regime: Diffusion Wins Every Column",
    "### 4.2.3 Розворот у розгортанні: регресія виграє режим best-of-1":
        "### 4.2.3 Rollout Reversal: Regression Wins Best-of-1",
    "### 4.2.4 Двобічне дослідження енергетичного ядра: охарактеризований негативний результат":
        "### 4.2.4 Two-Sided Energy-Head Study: A Characterized Negative Result",
    "### 4.2.5 Висновок щодо генеративного ядра":
        "### 4.2.5 Generative-Core Conclusion",
    "## 4.3 Одновкроковий гейт пам'яті та розгортальне дослідження зі зрівняним бюджетом":
        "## 4.3 Single-Step Memory Gate and Matched-Budget Rollout Study",
    "### 4.3.1 Попередній гейт: здатність пригадування виникає лише за більшого бюджету оптимізації":
        "### 4.3.1 Preliminary Gate: Recall Emerges Only after Sufficient Optimization",
    "### 4.3.2 Матриця розгортання": "### 4.3.2 Rollout Matrix",
    "### 4.3.3 Головне парне порівняння за двома посівами: виграш, що вмикається за межею горизонту":
        "### 4.3.3 Main Two-Seed Pair: Advantage beyond the Flat Memory Horizon",
    "### 4.3.4 flat-long: Парето-фронт за 2.7× токенів":
        "### 4.3.4 flat-long: the 2.7×-Token Pareto Reference",
    "### 4.3.5 Робастність окремого порівняння до голови та її межі":
        "### 4.3.5 Prediction-Head Robustness and Its Boundary",
    "### 4.3.6 Доповнення фронтиру токен-бюджету: зіставлені партнери на верхній та нижній межі":
        "### 4.3.6 Completing the Token-Budget Frontier with Matched Upper and Lower Partners",
    "## 4.4 Форми ієрархічної абстракції: просторова, часова, глибинна":
        "## 4.4 Forms of Hierarchical Abstraction: Spatial, Temporal and Depth",
    "### 4.4.1 Просторовий факторіал: економічний, залежний від ядра вердикт":
        "### 4.4.1 Spatial Factorial: an Economical, Head-Dependent Verdict",
    "### 4.4.2 Часове сімейство та рухомий край смуги":
        "### 4.4.2 Temporal Family and the Moving Tested Boundary",
    "### 4.4.3 Грубий рівень чи просто проліт? Контроль зі стрибковою повнороздільною історією":
        "### 4.4.3 Coarse Level or Reach Alone? Sparse Full-Resolution Control",
    "### 4.4.4 Що зберігає кожен рівень: зондування засліпленням":
        "### 4.4.4 What Each Level Preserves: Blinding Probes",
    "### 4.4.5 Зведена порівняльна таблиця: глибина, ядра, зібрана сітка":
        "### 4.4.5 Consolidated Comparison: Depth, Heads and the Assembled Grid",
    "## 4.5 Дослідження межі часової здатності (робоча точка Б)":
        "## 4.5 Temporal-Capability Boundary Study (Operating Point B)",
    "### 4.5.1 Епізод аліасингу: діагностована відмова, а не відкинутий запуск":
        "### 4.5.1 Aliasing Episode: Diagnosed Failure, Not a Discarded Run",
    "### 4.5.2 Геометрія: проліт пам'яті проти ефективного періоду циклу":
        "### 4.5.2 Geometry: Memory Reach versus Effective Loop Period",
    "### 4.5.3 Пригадування існує: розширене оцінювання та внутрішній контроль":
        "### 4.5.3 Recall Exists: Extended Evaluation and Internal Control",
    "### 4.5.4 Порогова гіпотеза прольоту та її одновкрокова перевірка за попередньо зафіксованим правилом":
        "### 4.5.4 Reach-Threshold Hypothesis and Preregistered Single-Step Test",
    "### 4.5.5 Вирішальна розгортальна пара та зареєстрована негативна реплікація":
        "### 4.5.5 Decisive Rollout Pair and Registered Negative Replication",
    "### 4.5.6 Методологічні уроки дослідження межі":
        "### 4.5.6 Methodological Lessons from the Boundary Study",
    "## 4.6 Узагальнення обраної конфігурації на стандартні та синтетичні набори даних":
        "## 4.6 Generalization across Standard and Synthetic Datasets",
    "### 4.6.1 Обрана конфігурація за режимами висновування":
        "### 4.6.1 Selected Configuration by Inference Regime",
    "### 4.6.2 Міст до стандартного стенда OpenSTL Moving-MNIST: виміряний негативний контроль":
        "### 4.6.2 OpenSTL Moving-MNIST Bridge: Measured Negative Control",
    "### 4.6.3 Синтетичні стенди [A1]/[A2]: 2D-обертання та 3D-полікуби":
        "### 4.6.3 Synthetic [A1]/[A2] Testbeds: 2D Rotations and 3D Polycubes",
    "### 4.6.4 MMNIST-CIFAR": "### 4.6.4 Moving-MNIST-CIFAR",
    "### 4.6.5 TaxiBJ: реальні дані просторово-часового прогнозування — ВИМІРЯНО (2026-07-07)":
        "### 4.6.5 TaxiBJ: Measured Real Spatio-Temporal Forecasting Data",
    "### 4.6.6 KTH-64 — майбутнє розширення": "### 4.6.6 KTH-64 — Future Extension",
    "### 4.6.7 Межі виміряного узагальнювального твердження":
        "### 4.6.7 Boundaries of the Measured Generalization Claim",
    "### 4.6.8 Незалежне підтвердження на недоторканій вибірці LoopMom":
        "### 4.6.8 Independent Confirmation on an Untouched LoopMom Holdout",
    "### 4.6.9 Робастність до бюджету семплювання та випадковості DDIM":
        "### 4.6.9 Robustness to Sampling Budget and DDIM Randomness",
    "### 4.6.10 Робастність до незалежно навченої латентної репрезентації":
        "### 4.6.10 Robustness to an Independently Trained Latent Representation",
    "### 4.6.11 Привілейований позиційно-індексований еталон явного доступу":
        "### 4.6.11 Privileged Pose-Indexed Explicit-Access Reference",
    "## 4.7 Впровадження результатів дослідження":
        "## 4.7 Implementation and Use of the Research Results",
    "## 4.8 Висновки до розділу": "## 4.8 Chapter Conclusions",
}


def _translate_parenthetical_heading(line: str) -> str:
    if not line.startswith("#"):
        return line
    match = re.match(r"^(#{1,3})\s+(.+?)\s+\(([^()]*)\)\s*$", line)
    if not match:
        return line
    level, left, english = match.groups()
    if not re.search(r"[A-Za-z]", english):
        return line
    number = re.match(r"^(\d+(?:\.\d+)*)\s+", left)
    prefix = f"{number.group(1)} " if number else ""
    return f"{level} {prefix}{english}"


def finalize(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    name = path.name

    if name in START_MARKERS:
        start = text.find(START_MARKERS[name])
        if start < 0:
            # Already finalized.
            start = 0
        text = text[start:]
    else:
        # Front matter and introduction retain their first heading but not the editorial blockquote.
        text = re.sub(r"\A(#[^\n]+\n)\n(?:>.*\n)+(?:\n)?", r"\1\n", text)

    end_marker = END_MARKERS.get(name)
    if end_marker and end_marker in text:
        text = text.split(end_marker, 1)[0].rstrip() + "\n"

    # Genre/localization notes are archival instructions, not dissertation prose.
    text = re.sub(r"\s*\[UA-PASS:.*?\]", "", text, flags=re.DOTALL)
    # Remove an editorial blockquote placed directly below a document/chapter title. Algorithm
    # blockquotes inside Chapter 3 are preserved because they are not at the start.
    text = re.sub(r"\A(#[^\n]+\n)\n(?:>.*\n)+(?:\n)?", r"\1\n", text)
    # Standalone italic parentheticals are rewrite/disposition notes, not scientific prose.
    text = re.sub(r"\n\*\(.*?\)\*\s*\n", "\n", text, flags=re.DOTALL)
    # Earlier rewrite passes also left editorial parentheticals attached to scientific
    # sentences. Remove only notes carrying an explicit provenance/disposition marker; retain
    # scientific parentheticals such as equation references and the Task 1--7 mapping.
    editorial_markers = (
        r"DISS_DIGEST|decision Q|old eq\.?|replaces? the old|retargeted|"
        r"rewrit(?:e|ten)|\bFIX\b|\bNEW equation\b|forward reference|"
        r"paradigm ordering|preserved from|misattribution|original genre|"
        r"register and application|packaging reframe|per decision|generalized from|"
        r"contradiction wording|reference retargeted|de-energized"
    )
    text = re.sub(
        rf"\s*\*\((?=[^)]*(?:{editorial_markers})).*?\)\*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = text.replace("\n>\n", "\n")

    lines = []
    for line in text.splitlines():
        line = EXACT_HEADINGS.get(line, line)
        line = _translate_parenthetical_heading(line)
        lines.append(line.rstrip())
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    for name in FILES:
        finalize(SOURCE / name)
    print("DISSERTATION_SOURCE_FINALIZATION_DONE")


if __name__ == "__main__":
    main()
