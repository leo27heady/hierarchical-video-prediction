# Довідник рецензента: походження тверджень, таблиць, формул і рисунків дисертації

**Дата зрізу:** 29 серпня 2026 р.
**Репозитарій:** `C:\Users\leoni\Documents\projects\hierarchical-video-prediction`
**Базовий коміт під час укладання:** `87a181c767d1899cf4dc53c41f8968fae2ec5164` (`87a181c`, `refactor(llm): formulas and tables`)
**Основний текст:** `docs/dissertation/condensed_translated_humanized_ukrainian/`
**Статус цього документа:** навігаційний і доказовий довідник. Він не замінює первинні артефакти, статті, код або реєстр тверджень.

> Важливо. На момент створення довідника робоча директорія містила незакомічені зміни у файлах `00_FRONT_MATTER.md`, `01_INTRODUCTION.md`, `02_CHAPTER_1.md` і `03_CHAPTER_2.md`. Тому новий рецензент повинен аналізувати поточні файли робочої директорії, а не лише стан коміту `87a181c`.

## 1. Навіщо існує цей файл

Документ дає новому агенту достатній контекст, щоб розпочати перевірку дисертації з нуля і не переплутати:

- емпіричний результат із пояснювальною схемою;
- авторське математичне означення з формулою, запозиченою з літератури;
- перевірену модель із запропонованим, але не перевіреним розширенням;
- центральний доказ четвертої статті з окремою дисертаційною перевіркою технічної здійсненності;
- кількість токенів або корисний обсяг стану з піковою пам’яттю GPU;
- відкладене підтвердження з діагностикою розроблення, робастністю або зовнішньою межею.

Під «кожним твердженням» у цьому довіднику розуміється кожне **змістове наукове твердження або сімейство тверджень**, а не службові сполучники й окремі редакційні речення. Буквальна відповідність кожного речення швидко застаріла б після мовної правки. Натомість нижче кожний підрозділ прив’язано до моделі, джерела даних, класу доказу й допустимого висновку; кожну нумеровану таблицю, формулу та ілюстрацію внесено до окремого реєстру.

## 2. Авторитетні файли та рекомендований порядок читання

### 2.1 Поточний рукопис

Читати в такому порядку:

1. `docs/dissertation/condensed_translated_humanized_ukrainian/00_FRONT_MATTER.md`;
2. `docs/dissertation/condensed_translated_humanized_ukrainian/01_INTRODUCTION.md`;
3. `docs/dissertation/condensed_translated_humanized_ukrainian/02_CHAPTER_1.md`;
4. `docs/dissertation/condensed_translated_humanized_ukrainian/03_CHAPTER_2.md`;
5. `docs/dissertation/condensed_translated_humanized_ukrainian/04_CHAPTER_3.md`;
6. `docs/dissertation/condensed_translated_humanized_ukrainian/05_CHAPTER_4.md`;
7. `docs/dissertation/condensed_translated_humanized_ukrainian/06_CONCLUSIONS.md`;
8. `docs/dissertation/condensed_translated_humanized_ukrainian/07_REFERENCES.md`;
9. `docs/dissertation/condensed_translated_humanized_ukrainian/08_APPENDICES.md`.

Англійський стислий варіант у `docs/dissertation/condensed_english/` корисний для встановлення первинного значення речень, але поточним об’єктом редагування є український текст. Старші каталоги `rewrite/`, `restructured_english/` і `final/` містять історію змісту та матеріал для перевірки повноти; вони не мають автоматично перезаписувати новішу стислу редакцію.

### 2.2 Плани, що пояснюють поточну структуру

- `docs/dissertation/DISSERTATION_RESTRUCTURE_AND_WRITING_BLUEPRINT_2026-08-10.md` – структура, обсяг, вимоги до стилю, ілюстрацій і доказового тону.
- `docs/dissertation/DISSERTATION_SINGLE_PROBLEM_AND_INTEGRATION_REVIEW_PLAN_2026-08-23.md` – перехід до однієї центральної проблеми та трьох обмежень.
- `docs/dissertation/DISSERTATION_SURGICAL_SINGLE_PROBLEM_AND_INTEGRATED_FEASIBILITY_REFACTOR_PLAN_2026-08-23.md` – точкові зміни й місце інтегрованої перевірки.
- `docs/CONTENT_FINALIZATION_CLAIM_MATRIX_2026-07-19.md` – дозволене і заборонене формулювання основних результатів.
- `docs/CLAIM_LEDGER.md` – заморожений науковий реєстр четвертої статті.

### 2.3 Статті й публічні реалізації

| Гілка | Публікація або рукопис | Код / дані | Роль у дисертації |
|---|---|---|---|
| Огляд консистентності | Kulyk, Mokin, 2024, DOI `10.31649/1997-9266-2024-175-4-75-85`, посилання [133] | опубліковані ілюстрації вилучено з `docs/figures/restructured/sources/Kulyk_Mokin_2024_official.pdf` | поняття консистентності, генеративні інтерфейси та історичний контекст; не експериментальна опора головної межі |
| TempVerseFormer | *Temporal Modeling with Reversible Transformers*, Neural Computing and Applications 38, article 30 (2026), DOI `10.1007/s00521-025-11707-3` | `https://github.com/leo27heady/TempVerseFormer` | пам’ять активацій і реверсивне навчання довгих часових ланцюгів |
| V-BLT / RoPE-ND | Kulyk, Mokin, *Scaling Video Prediction with Spatio-Temporal Patches*, 2025, DOI `10.31649/1997-9266-2025-182-5-129-139` | `https://github.com/leo27heady/visionBLT`; дані: `https://github.com/leo27heady/simple-shape-dataset-toolbox` | ієрархічна локально-глобальна обробка, байтова ціль, RoPE-ND та діагностична ентропія |
| Ієрархічний контекст | `docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md` | цей репозитарій, насамперед `hvp_rev/`, `hvp/`, `experiments/`, `artifacts/` | центральне matched-budget дослідження межі «досяжність–точність відтворення» |
| Інтегрований прототип | тільки дисертаційна перевірка здійсненності; до четвертої статті не входить | `hvp_rev/fixed_region_revformer.py`, `hvp_rev/packed_simvp.py`, `hvp_rev/reversible_rollout.py` і пов’язані скрипти | демонстрація технічної сумісності фіксованого грубо-детального стану, SimVP і реверсивного розгортання |

## 3. Центральна логіка, яку не можна пошкодити редакторською правкою

Одна центральна проблема роботи – **довгогоризонтна консистентність ієрархічного прогнозування відео за обмежених ресурсів**. Її основний архітектурний аспект – як зберегти віддалену релевантну історію за обмеженого бюджету контексту. Три взаємопов’язані обмеження вимірюються окремо:

1. `M_context/state`: ієрархічне або багатошвидкісне пакетування змінює обсяг і форму збереженої історії;
2. `M_activations`: TempVerseFormer відтворює проміжні стани замість зберігання всієї стрічки активацій;
3. просторова інформаційна ємність: V-BLT показує фіксовану локально-глобальну ієрархію та ентропійний діагностичний сигнал, але не підтверджує динамічний маршрутизатор.

Головний емпіричний висновок четвертої статті: **ієрархія купує досяжність; плоский контекст купує точність відтворення**. Це умовне правило для перевіреної області, а не універсальна перевага архітектури.

Допустима сильна арифметична заява: за однакової номінальної 48-кадрової досяжності `48 coarse + 6 fine = 1 152` і `48 coarse + 2 fine = 896` контекстних токенів проти `48 fine = 3 072`, тобто у 2,67 і 3,43 раза менше. Заборонено перетворювати це на «та сама якість», «у стільки ж разів менша VRAM» або множити на коефіцієнт економії активацій.

Інтегрований прототип підтверджує сумісність принципів, але:

- має одну навчальну ініціалізацію;
- використовує Moving-MNIST 10→10 і фіксовані ділянки;
- не відтворює 48-кадрову межу LoopMom;
- не входить до четвертої статті;
- не доводить перевагу ентропійної динамічної маршрутизації;
- не перетворює окремі дисертаційні гілки на одну всебічно перевірену архітектуру.

## 4. Класи походження тверджень

| Код | Тип твердження | Що є первинним джерелом | Як перевіряти |
|---|---|---|---|
| `LIT` | оглядове або методологічне | посилання у `07_REFERENCES.md`, насамперед [1–174] | перевірити, що джерело підтримує саме це речення, а не лише близьку тему |
| `PUB` | результат раніше опублікованої статті автора | PDF статті, публічний репозитарій, опублікована таблиця або рисунок | не «покращувати» числа й не приписувати причинність конфігураційному порівнянню |
| `DEF` | авторське означення, облік або математична постановка | поточний текст, реалізований інтерфейс і код | перевірити одиниці, індекси та відповідність реалізації; зовнішнє посилання не завжди потрібне |
| `DER` | алгебраїчне виведення або прямий розрахунок | попередні означення та арифметика | перерахувати незалежно; це не результат навчання моделі |
| `EMP` | заморожений експериментальний результат | JSON-артефакт, маніфест, журнал і аналізатор | числа мають збігатися з артефактом і класом доказу |
| `SYN` | авторський синтез кількох результатів | сукупність `LIT`, `PUB`, `DEF` або `EMP` | оцінити логічність переходу й не видавати синтез за окремий експеримент |
| `ILL` | пояснювальна ілюстрація або приклад | авторська схема або вибраний приклад | не використовувати як кількісний доказ |
| `ADMIN` | адміністративні відомості | офіційні дані автора та установи | не виводити з Git, статей або припущень |

## 5. Моделі, набори даних і експериментальні програми

### 5.1 Моделі

| Позначення в тексті | Фактичний об’єкт | Навчальна / прогнозна ціль | Що доведено | Що не доведено |
|---|---|---|---|---|
| TempVerseFormer | VAE + часовий трансформер з адитивним реверсивним станом і фіксованим вікном переходу | MSE/LPIPS/KL у прихованому просторі та реконструкції згідно зі статтею і кодом | реконструкція повторюваного стану; зменшення активаційної пам’яті в перевірених режимах; зовнішні OpenSTL результати | що весь тракт VAE→VAE є оборотним; що реверсивність покращує якість; що вона спричиняє результати LoopMom |
| V-BLT | локальний кодер → глобальний трансформер → локальний байтовий декодер, фіксовані просторово-часові патчі, RoPE-ND | категоріальна перехресна ентропія для 256 значень байта | працездатність повної конфігурації на 2-D/3-D обертаннях; просторово структурована ентропія | причинний внесок кожної компоненти; калібрована невизначеність; виграш динамічного ентропійного поділу |
| Основний WindowDiT-предиктор | заморожений дворівневий кодек (`4×4` coarse, `8×8` fine, 16 каналів) + один joint-attention DiT для чотирикадрової цілі | `v`-prediction diffusion або регресійна SmoothL1 голова | matched-budget межа контекстних пакетів; робастність і межі | універсальна перевага ієрархії; однакова якість при меншій кількості токенів; природне відео |
| EBT-діагностика | адаптований Energy-Based Transformer зі скалярною енергією та градієнтним спуском за кандидатом майбутнього | ранжування енергією й оптимізація кандидата | енергія навчається ранжувати, але холодний спуск не досягає потрібної різкості в перевіреному бюджеті | що EBT загалом непридатні; що порівняння є matched-capacity з основним DiT |
| Інтегрований прототип | фіксоване пакування `16 coarse + 8 regional fine` → адаптер → SimVP → реверсивний часовий ланцюг | регресійне прогнозування Moving-MNIST | технічна сумісність; якість однієї конфігурації; same-model BPTT/reversible профіль пам’яті | підтверджувальна нова архітектурна перевага; динамічна маршрутизація; багатонасіннєва робастність |

### 5.2 Дані

| Набір | Джерело / генератор | Доказова роль |
|---|---|---|
| Процедурні 2-D фігури й 3-D полікуби | `simple-shape-dataset-toolbox`; рівняння 2.1–2.3 | контрольована динаміка TempVerseFormer і V-BLT; не довгогоризонтна пам’ять |
| Moving-MNIST 10→10 | OpenSTL / локальний генератор | компетентність і порівняння інтегрованого прототипу; короткий горизонт |
| TaxiBJ, KTH, MMNIST-CIFAR | рецепти TempVerseFormer / OpenSTL | зовнішня працездатність опублікованої часової моделі |
| LoopMom | `hvp_rev` і скрипти генерації; тороїдальний світ 192×64 з вікном 64×64 | центральна точна спостережуваність, повторний вхід і парно-специфічні популяції досяжності |
| Long Moving-MNIST з постійними ID | Phase C v2 | зовнішня межа: просторове покриття/траєкторія проти ідентичності |

### 5.3 Зареєстровані програми четвертої статті

| Код | Питання | Первинний артефакт | Аналіз / валідація | Термінальний статус |
|---|---|---|---|---|
| E2 | untouched reach/frontier | `artifacts/confirmation_e2/results.json` | `scripts/analyze_e2_confirmation.py`, `scripts/validate_e2_completion.py` | завершено |
| E4/E4b | голова × форма пакета | `artifacts/e4_head_package/results.json`, `artifacts/e4b_head_package/results.json` | `scripts/analyze_e4_factorial.py`, `scripts/analyze_e4b_factorial.py` | завершено |
| E5a | реплікація пізньої self-fed межі | `artifacts/e5a_trust_replication/results.json` | `scripts/analyze_e5a_trust_replication.py`, `scripts/validate_e5a_completion.py` | пізню межу не відтворено |
| E7c | sampler/NFE robustness | `artifacts/e7c_sampler_nfe/results.json` | `scripts/analyze_e7c_sampler_nfe.py`, `scripts/validate_e7c_completion.py` | завершено |
| E7a | незалежний кодек | `artifacts/e7a_codec_robustness/results.json` | `scripts/analyze_e7a_codec_robustness.py`, `scripts/validate_e7a_completion.py` | завершено; верхня точка невизначена family-wise |
| E8a | явний pose-indexed raw-access еталон | `artifacts/e8a_pose_mosaic/results.json` | `scripts/analyze_e8a_pose_mosaic.py`, `scripts/validate_e8a_completion.py` | завершено; привілейований еталон |
| E6 / Phase C v2 | постійні об’єкти | `artifacts/phasec_v2/results.json` | `scripts/analyze_phasec_v2.py`, `scripts/validate_phasec_completion.py` | завершено; змішана зовнішня межа |
| E9 | контрфактичний світ | `experiments/authorization.json` | utility gate у канонічному реєстрі | `SKIPPED_POST_FREEZE_NO_INCREMENTAL_UTILITY` |

Повна машинна прив’язка й SHA-256: `artifacts/rft_claim_artifact_index.json`; релізний зріз: `artifacts/release_manifest_2026-07-19.json`.

## 6. Карта змістових тверджень за розділами

### 6.1 Вступні матеріали, вступ і висновки

| Місце | Змістове твердження | Клас і джерело | Межа перевірки |
|---|---|---|---|
| `00_FRONT_MATTER`, анотація | тема, мета, опис розділів, основні результати, новизна й практичне значення | `SYN` з розділів 1–4 та `EMP` з артефактів | кожне число має дослівно узгоджуватися з розділом 4; адміністративні поля лишаються `ADMIN` |
| Вступ, актуальність | довгогоризонтна помилка накопичується; повторний вхід потребує збереженої історії; одного показника недостатньо | `LIT` + `SYN`, деталізовано в 1.1–1.2 | не стверджувати, що будь-яка довга генерація обов’язково відмовляє однаково |
| Вступ, три обмеження | контекст, активації та просторова ємність є різними витратами одного процесу | `DEF` + `SYN`; формули 1.5, 2.6 та 3.3–3.5 | не додавати ці величини без спільних одиниць і не множити коефіцієнти економії |
| Вступ, центральний фокус | ієрархічний контекст є головним емпіричним об’єктом; TempVerseFormer/V-BLT усувають суміжні обмеження | `SYN` з планів реструктуризації | це не твердження про один спільно навчений pipeline |
| Мета і завдання | шість завдань формують одну проблематику; інтеграція є перевіркою сумісності, не сьомою незалежною роботою | авторська постановка `DEF` | зміна кількості завдань потребує синхронізації новизни, висновків і анотації |
| Методи | системний аналіз, алгебраїчна інверсія, пакетний облік, LoopMom, кластерний bootstrap, max-statistic | `LIT`, `DEF`, `EMP` | опис статистики має відповідати формулам 3.41–3.45 |
| Наукова новизна | новизна належить методології, контрольованим втручанням, RoPE-ND/TempVerseFormer та перевіреній межі | `PUB` + `EMP` + авторська атрибуція | не заявляти винайдення ієрархічного відеомоделювання або universal SOTA |
| Практичне значення | генератори даних, fail-closed експерименти, метрики, пакування, API і прототип | код, тести, `docs/RELEASE_QA_REPORT_2026-07-18.md` | дослідницький статус не дорівнює виробничій готовності |
| Загальні висновки 1–6 | стислий підсумок системної постановки, основного предиктора, оцінювання, TempVerseFormer, V-BLT та інтеграції | ті самі джерела, що відповідні розділи | висновки не мають містити нового числа або сильнішої причинної заяви |

### 6.2 Розділ 1

| Підрозділ | Основна заява | Походження |
|---|---|---|
| 1.1.1 | відео є частковим спостереженням прихованого динамічного стану; контекстний оператор формує доступний стан моделі | `LIT` системного аналізу + авторська формалізація 1.1–1.3 |
| 1.1.2 | self-fed rollout змінює майбутній розподіл входів і накопичує помилки | `LIT` про exposure bias/авторегресію + означення 1.4 |
| 1.1.3 | якість є вектором критеріїв, а пам’ять має кілька компонентів | авторський `DEF/SYN`, формули 1.5–1.7 |
| 1.2.1–1.2.5 | консистентність не зводиться до MSE/SSIM/LPIPS/FID/FVD; потрібні метрики підтримки, ID, траєкторії, зайвого вмісту й міжрівневої відповідності | огляд [133], метрична література [94, 127–129] і авторські означення 1.8–1.11 |
| 1.3.1 | BPTT зберігає стрічку активацій, яка в першому наближенні росте з горизонтом | `LIT` [28–32] + асимптотика 1.12; точні результати лише в розділі 2 |
| 1.3.2 | фіксована сітка витрачає однакову кількість токенів на нерівномірно складні області | `SYN` із V-BLT та формули 1.13; не доказ виграшу динамічного router |
| 1.3.3 | коротке детальне вікно може не адресувати визначальну минулу подію | авторське означення reach 1.14; експериментально перевірено через LoopMom у розділах 3–4 |
| 1.3.4–1.5 | центральна проблема розкладається на обмеження, для яких потрібні різні втручання й різні докази | `SYN` із усіх трьох статей і claim matrix |

### 6.3 Розділ 2

| Підрозділ | Основна заява | Походження |
|---|---|---|
| 2.1 | процедурні 2-D/3-D послідовності дають відомий стан і контроль руху | `PUB`, `simple-shape-dataset-toolbox`, формули 2.1–2.3 |
| 2.2.1–2.2.4 | адитивний реверсивний блок дозволяє реконструювати повторюваний часовий стан і локально перебудувати граф зворотного проходу | TempVerseFormer, формули 2.4–2.15; межа реверсивності явно зафіксована таблицею 2.2 |
| 2.2.5–2.2.7 | модель відтворює контрольовані закономірності й OpenSTL дані; пам’ять майже стала в опублікованих точках, час зростає | `PUB` зі статті TempVerseFormer, таблиці 2.3–2.6, рисунки 2.9–2.11 |
| 2.2.8 | стохастичний перехід може бути оборотним лише за відтворення шуму; довга числова стійкість потребує окремої перевірки | дисертаційне `DER/DEF`, формули 2.16–2.27; не основний опублікований доказ |
| 2.2.9 | checkpointing і reversibility мають різні області доцільності | `LIT/SYN`, таблиця 2.7 |
| 2.3.1–2.3.4 | V-BLT реалізує фіксований local-global-local тракт, причинну байтову ціль і RoPE-ND | `PUB`, код visionBLT, формули 2.28–2.39 |
| 2.3.5 | ентропія категоріального виходу є просторовим діагностичним сигналом | `PUB` + Shannon [9], формули 2.40–2.41 | не називати каліброваною невизначеністю або реалізованим контролером |
| 2.3.6 | опублікована конфігурація конкурентна на процедурних даних, але має змішаний 3-D рейтинг | таблиці 2.9–2.10, рисунок 2.18 | не виділяти причинний внесок RoPE-ND/patching без ablation |
| 2.3.7 | адаптивний поділ на області є майбутньою оптимізаційною постановкою | `DEF`, формула 2.42, рисунок 2.19 | позначати як неперевірену пропозицію |

### 6.4 Розділ 3

| Підрозділ | Основна заява | Походження |
|---|---|---|
| 3.1 | пакет визначається рівнем, кроком, кількістю слотів і лагом; reach і fidelity є різними властивостями | авторські означення 3.1–3.6, реалізація `ArmSpec`/контекстних пакетів |
| 3.2 | заморожений двохрівневий кодек формує 16 coarse і 64 fine токени на кадр; один Transformer спільно обробляє контекст і ціль | `hvp_rev/rhvd.py`, конфігурації експериментів, формули 3.7–3.18 |
| 3.3 | основна голова є latent diffusion з `v`-prediction і DDIM; регресія є контрольним вихідним модулем | код моделі й навчання, формули 3.17–3.28 | не повертати застарілий опис epsilon-prediction |
| 3.4 | self-fed rollout оновлює кільцеві coarse/fine буфери; токени, payload і total sequence length обліковуються окремо | код rollout/context, формули 3.29–3.32 |
| 3.5.1–3.5.3 | LoopMom створює точну подію виходу/повторного входу; популяція reach difference залежить від конкретної пари пакетів | генератор LoopMom, `DATA_CARD_LOOPMOM.md`, формули 3.33–3.38 |
| 3.5.4–3.5.5 | повні маскові й persistent-object метрики розділяють пропуски, зайві компоненти, ID і траєкторію | реалізації метрик у `hvp_rev`, формули 3.39–3.40 |
| 3.5.6 | bootstrap повторно відбирає цілі кліпи; max-statistic дає одночасні інтервали | статистична реалізація й аналізатори, формули 3.41–3.45 |
| 3.6 | гіпотези визначають очікувані напрями до перегляду відкладених результатів | протоколи й `experiments/authorization.json` |
| 3.7 | фіксований 24-токенний стан, SimVP і реверсивне розгортання утворюють feasibility prototype | нові модулі інтеграції, формули 3.46–3.53 | не підмінює головну модель і не доводить dynamic entropy routing |

### 6.5 Розділ 4

| Підрозділ | Основна заява | Первинне джерело |
|---|---|---|
| 4.1 | програми мають різні класи доказів і ліцензії порівняння | `APPENDIX_CH4_CONFIGURATION_AND_EVIDENCE_REGISTRY.md`, authorization ledger |
| 4.2.1 | diffusion краща в однокроковому best-of-8 тесті, regression дешевша й іноді краща self-fed | заморожені механізм-досліди; таблиця 4.1 та додаток Е |
| 4.2.2 | EBT ранжує кандидатів, але перевірений спуск не досягає достатньої різкості | `logs_rhcp/ebt_scaled_s1.log`, `logs_rhcp/ebt_scaled_s2.log`; рисунок 4.4 |
| 4.3 | у парно визначеній смузі reach hierarchy має +0,0980/+0,0914, а повний набір метрик підтримує механізм | `artifacts/confirmation_e2/results.json` |
| 4.4 | підтверджено hierarchy при 896/1 152, 1 792 невизначено, flat при 2 432/3 072; 2,67–3,43× стосується токенів | той самий E2 артефакт + прямий розрахунок токенів |
| 4.5 | форма пакета взаємодіє з головою й seed; diffusion temporal package стабільний у п’яти seeds, regression heterogeneous | E4/E4b артефакти |
| 4.6 | напрями зберігаються при NFE/sampler і незалежному codec, але верхні family-wise інтервали іноді невизначені | E7c та E7a артефакти |
| 4.7.1 | ранній ефект відтворився, пізня «trust boundary» не відтворилась | E5a |
| 4.7.2 | pose-indexed raw K48 є нижчою за помилкою привілейованою діагностикою | E8a; не token-matched рейтинг |
| 4.7.3 | E6: hierarchy підтримує support/continuity, flat краще ID; абсолютна якість низька | Phase C v2 |
| 4.7.4 | короткі OpenSTL/TaxisBJ тести є competence boundary, не доказом long-range memory | опубліковані/локальні короткі тести |
| 4.8 | інтегрований прототип має MSE 28,72, MAE 80,27 і same-model memory benefit 2,58× при 40 кадрах | зовнішні D:-артефакти інтеграції, наведені в §10.3 |
| 4.9 | метод обирається за обмежувальним ресурсом; дві осі економії виміряні окремо | `SYN` з розділів 2–4 |
| 4.10–4.11 | кодова база, практична роль і загрози валідності | репозитарій, QA-звіти, обмеження експериментів |

## 7. Повний реєстр формул

Формули набиралися безпосередньо в Markdown; окремого скрипту, що «генерує формулу», немає. У колонці «реалізація / основа» вказано, звідки взялася математична конструкція. Під час виправлення позначень потрібно змінювати також усі посилання на номер і, для реалізованих інтерфейсів, відповідний код або навпаки.

### 7.1 Формули розділу 1

| № | Значення | Тип | Реалізація / основа |
|---|---|---|---|
| 1.1 | рівняння прихованого стану та спостереження | `LIT/DEF` | стандартна системна модель; не конкретна нейромережа |
| 1.2 | умовний розподіл майбутнього вікна за станом контексту й керуванням | `DEF` | загальний інтерфейс усіх предикторів дисертації |
| 1.3 | багатокрокова оптимізація з вагами втрат | `DEF` | авторська постановка, не фактична єдина loss-функція всіх моделей |
| 1.4 | self-fed авторегресивний розподіл | `DEF` | реалізований принцип rollout основної моделі |
| 1.5 | розклад повної пам’яті системи | `SYN/DEF` | компоненти обліковуються окремо; продовжено у 2.6 і 3.31–3.32 |
| 1.6 | вектор критеріїв консистентності | `DEF` | узагальнення маскових, ID, trajectory та cross-level метрик |
| 1.7 | багатокритеріальна ціль за ресурсних обмежень | `DEF` | концептуальна оптимізація, не виконаний scalarized experiment |
| 1.8 | вектор object-level якостей | `DEF` | попередник persistent-object метрик 3.39–3.40 та E6 |
| 1.9 | середня похибка траєкторії центроїда | `DEF` | реалізована E6-метрика trajectory error |
| 1.10 | частки пропущеного та зайвого переднього плану | `DEF` | маскова семантика основних метрик LoopMom |
| 1.11 | міжрівнева невідповідність coarse/fine декодувань | `DEF` | концептуальна метрика; не headline endpoint четвертої статті |
| 1.12 | `Θ(KA)` для стрічки BPTT | `LIT/DER` | асимптотика; виміряні дані TempVerseFormer наведено окремо |
| 1.13 | кількість токенів фіксованої просторово-часової сітки | `DER` | геометрична арифметика патчів V-BLT |
| 1.14 | номінальна досяжність найдавнішої прив’язки | `DEF` | спрощений вступ до точного пакета 3.1–3.5 |

### 7.2 Формули розділу 2

| № | Значення | Тип | Модель / джерело |
|---|---|---|---|
| 2.1 | детермінований перехід процедурного стану; renderer визначено в тексті | `DEF/PUB` | `simple-shape-dataset-toolbox` |
| 2.2 | рівномірне 2-D обертання | `DEF/PUB` | процедурний набір 2-D фігур |
| 2.3 | тривимірний кутовий перехід | `DEF/PUB` | процедурні 3-D полікуби |
| 2.4 | наближене лінійне зростання активацій BPTT | `DER/PUB` | TempVerseFormer baseline |
| 2.5 | обчислювальна ціна звичайного BPTT | `DER` | прямий і зворотний прохід одного transition core |
| 2.6 | повний розклад пікової пам’яті | `SYN/DEF` | уточнює, що tape не є всією VRAM |
| 2.7 | перший additive-coupling вихід | `LIT/PUB` | TempVerseFormer reversible block |
| 2.8 | другий additive-coupling вихід | `LIT/PUB` | TempVerseFormer reversible block |
| 2.9 | відновлення другого вхідного потоку | `DER/PUB` | точна алгебраїчна інверсія 2.8 |
| 2.10 | відновлення першого вхідного потоку | `DER/PUB` | точна алгебраїчна інверсія 2.7 |
| 2.11 | одиничний визначник Якобі additive coupling | `DER` | алгебраїчна властивість блоку; не гарантія числової стійкості |
| 2.12 | повторне застосування reversible transition | `DEF/PUB` | часовий ланцюг TempVerseFormer |
| 2.13 | вихідна проєкція й VAE decoder | `DEF/PUB` | нереверсивна межа повного тракту |
| 2.14 | loss, описана в статті | `PUB` | TempVerseFormer paper |
| 2.15 | loss, реалізована в коді | `PUB/code audit` | репозитарій TempVerseFormer; окремо від 2.14 через різницю paper/code |
| 2.16 | рекуренція adjoint для стану | `DER` | reverse-mode differentiation через реконструйований transition |
| 2.17 | сума градієнтів параметрів за часовими переходами | `DER` | TempVerseFormer reversible backward |
| 2.18 | додаткова обчислювальна ціна reversible replay | `DER` | пояснює memory–compute trade-off |
| 2.19 | стохастичний перехід із шумом | `DEF` | дисертаційне розширення, не основний paper experiment |
| 2.20 | інверсія стохастичного переходу за відомого шуму | `DER` | потребує відтворення RNG |
| 2.21 | перше оновлення stochastic additive coupling | `DEF` | розширення TempVerseFormer |
| 2.22 | друге оновлення stochastic additive coupling | `DEF` | розширення TempVerseFormer |
| 2.23 | відновлення другого потоку stochastic coupling | `DER` | потребує відтворення RNG |
| 2.24 | відновлення першого потоку stochastic coupling | `DER` | перевірка відтворення стану |
| 2.25 | розклад пам’яті відтворення на anchors, RNG і gates | `DER` | три альтернативні контракти RNG описано в тексті |
| 2.26 | пам’ять anchor schedule `O(√T D_s)` | `DER` | запропонований компроміс, не headline вимірювання |
| 2.27 | верхня Lipschitz-межа residual map | `DER/LIT` | діагностика стійкості; не доводить contraction |
| 2.28 | число фіксованих spatiotemporal patches | `DER/PUB` | V-BLT |
| 2.29 | локальне кодування патча | `DEF/PUB` | V-BLT local encoder |
| 2.30 | глобальний причинний Transformer з RoPE-ND | `DEF/PUB` | V-BLT global model |
| 2.31 | локальні 256-класові byte logits | `DEF/PUB` | V-BLT local decoder |
| 2.32 | причинна часова attention mask | `DEF/PUB` | V-BLT implementation |
| 2.33 | softmax-розподіл 256 значень байта | `LIT/PUB` | категоріальний вихід V-BLT |
| 2.34 | byte cross-entropy | `LIT/PUB` | навчальна ціль V-BLT |
| 2.35 | комплексна форма 1-D RoPE | `LIT` | RoPE [107] |
| 2.36 | залежність phase dot product від відносного зсуву | `DER/LIT` | RoPE |
| 2.37 | дійсна матрична форма обертання | `DER/LIT` | еквівалент 2.35–2.36 |
| 2.38 | конкатенація окремих часової, вертикальної й горизонтальної осей | `PUB/DEF` | авторський RoPE-ND у V-BLT |
| 2.39 | осьовий розклад attention score | `DER/PUB` | RoPE-ND |
| 2.40 | Shannon entropy у бітах | `LIT` | 256-класовий byte distribution |
| 2.41 | entropy у натах і перетворення одиниць | `DER` | фактичний код візуалізації V-BLT |
| 2.42 | оптимізаційна постановка dynamic region partition | `DEF` | **гіпотетичне майбутнє розширення**, не реалізований V-BLT result |


### 7.3 Формули розділу 3

| № | Значення | Тип | Реалізація / джерело |
|---|---|---|---|
| 3.1 | контекстний пакет як множина потоків `(level, stride, slots, newest lag)` | `DEF` | `ArmSpec`/контекстні конфігурації основної моделі |
| 3.2 | лаги та фактичні часові прив’язки потоку | `DEF` | явна anchor geometry |
| 3.3 | кількість контекстних токенів | `DER/DEF` | package validation і таблиця В.8 |
| 3.4 | кількість цільових токенів; повну довжину послідовності визначено в тексті | `DER/DEF` | WindowDiT input sequence |
| 3.5 | номінальна reach пакета | `DER/DEF` | anchor geometry; не retrieval success |
| 3.6 | концептуальна декомпозиція risk на недосяжність і втрату fidelity | `SYN/DEF` | пояснює гіпотезу, не fitted probability model |
| 3.7 | парний вихід coarse/fine замороженого кодера; форми тензорів визначено в тексті | `DEF/code` | заморожений дворівневий codec |
| 3.8 | незалежні coarse/fine decoders | `DEF/code` | codec interface |
| 3.9 | stochastic branch loss кодека | `DEF/code` | навчання codec, не predictor loss |
| 3.10 | channel-wise нормування latent maps | `DEF/code` | frozen codec statistics |
| 3.11 | проєкція однієї latent cell у Transformer token | `DEF/code` | WindowDiT input embedding |
| 3.12 | конкатенація позначених послідовностей контексту $\mathcal C_t$ і цілі $\mathcal T_t$ | `DEF/code` | exact token sequence; розгортання обох підпослідовностей наведено в тексті, а не всередині одного перевантаженого виразу |
| 3.13 | level/role/frame embeddings | `DEF/code` | WindowDiT token metadata |
| 3.14 | diffusion-step/action conditioning | `DEF/code` | target conditioning; context має zero diffusion step |
| 3.15 | RMS-normalized joint self-attention | `LIT/DEF` | Transformer implementation |
| 3.16 | stop-gradient до старих predictions при truncated rollout training | `DEF/code` | межа gradient graph |
| 3.17 | conditional joint distribution coarse/fine target window | `DEF` | основна diffusion model |
| 3.18 | forward noising equation | `LIT/code` | latent diffusion |
| 3.19 | нормований cosine noise schedule з implementation clip | `LIT/code` | cosine form із [149], `s=0.008`; межі $10^{-4}$ і $0.9999$ є запобіжником реалізації |
| 3.20 | `v`-prediction target | `LIT/code` | **binding executed objective** |
| 3.21 | coarse+fine MSE у `v`-space | `DEF/code` | diffusion training loss |
| 3.22 | adaptive LayerNorm modulation | `LIT/code` | DiT/AdaLN-Zero |
| 3.23 | gated attention residual update | `LIT/code` | WindowDiT block |
| 3.24 | gated MLP residual update | `LIT/code` | WindowDiT block |
| 3.25 | DDIM-реконструкція `z0` | `LIT/code` | deterministic sampler |
| 3.26 | DDIM-оцінка шуму | `LIT/code` | deterministic sampler |
| 3.27 | DDIM наступний стан | `LIT/code` | sampler update |
| 3.28 | SmoothL1 coarse+fine regression loss | `DEF/code` | regression control head |
| 3.29 | coarse/fine ring-buffer update | `DEF/code` | self-fed rollout, `(Nc,Nf)=(48,6)` у main arm |
| 3.30 | exact anchor index and lag | `DEF/code` | context geometry validator |
| 3.31 | latent payload bytes | `DER` | не peak VRAM; таблиця В.8 |
| 3.32 | quadratic attention approximation | `LIT/DER` | theoretical sequence-cost comparison, not profiler result |
| 3.33 | LoopMom toroidal viewport motion | `DEF/code` | LoopMom generator |
| 3.34 | world columns covered by last `m` frames | `DEF/code` | analytic observability |
| 3.35 | foreground world columns in target | `DEF/code` | threshold 0.5 |
| 3.36 | множина нового переднього плану; частку визначено в тексті | `DEF/code` | reach-difference population construction |
| 3.37 | частка доступних для відповіді стовпців; медіанний вік визначено в тексті | `DEF/code` | pair-specific diagnostic |
| 3.38 | registered pair-specific reach population | `DEF/code` | E2 filter; do not replace with a universal fixed band without checking pair |
| 3.39 | узагальнена маскова MSE; конкретні маски й правила порожніх множин визначено в тексті | `DEF/code` | frame metrics |
| 3.40 | вартість пари для Hungarian matching; coverage та identity визначено в тексті | `DEF/code` | Hungarian assignment and E6 endpoints |
| 3.41 | record-weighted paired mean effect | `DEF/code` | statistical analyzer |
| 3.42 | whole-clip bootstrap replicate | `DEF/code` | clips are sampling units |
| 3.43 | bootstrap standard error | `DER/code` | simultaneous inference |
| 3.44 | studentized max statistic | `DER/code` | registered family correction |
| 3.45 | simultaneous confidence interval | `DER/code` | family-wise inference |
| 3.46 | softmax-ваги усередині fixed region; variance score визначено в тексті | `DEF/code` | integrated fixed-region packer; not V-BLT entropy router |
| 3.47 | regional token from mean + weighted fine summary | `DEF/code` | `hvp_rev/fixed_region_revformer.py` |
| 3.48 | 24×16 packed state = 384 scalars/frame | `DER/code` | integrated prototype |
| 3.49 | packed SimVP transition | `DEF/code` | `packed_simvp.py` + reconstructor |
| 3.50 | перше пряме рівняння additive reversible rollout | `DEF/code` | `reversible_rollout.py` |
| 3.51 | друге пряме рівняння additive reversible rollout | `DEF/code` | `reversible_rollout.py` |
| 3.52 | відновлення другого попереднього компонента | `DER/code` | exact algebraic inverse |
| 3.53 | відновлення першого попереднього компонента | `DER/code` | same module; numerical drift measured separately |


### 7.4 Формули розділу 4

| № | Значення | Тип | Джерело |
|---|---|---|---|
| 4.1 | гармонізація знаку ефектів: positive favors registered candidate | `DEF` | усі аналізатори четвертої статті |
| 4.2 | кількість network evaluations для regression і diffusion | `DER` | виконані heads; regression 1 NFE, diffusion `K` NFE |
| 4.3 | gradient descent за EBT energy | `LIT/DEF` | EBT diagnostic implementation |
| 4.4 | weighted reach-band paired effect | `DEF/code` | `artifacts/confirmation_e2/results.json` і `analyze_e2_confirmation.py` |
| 4.5 | context-token gain relative to 48 fine frames | `DER` | таблиця 4.3; arithmetic only |

## 8. Повний реєстр таблиць

У поточній редакції 40 нумерованих таблиць. Більшість таблиць набрано вручну в Markdown з названого джерела. Скрипт аналізу створює JSON або проміжну Markdown-таблицю, але остаточне розміщення й українські підписи редагувалися вручну. Якщо окремого генератора немає, це прямо зазначено; у такому разі числа потрібно звіряти з первинним PDF/JSON, а не з іншою копією таблиці в рукописі.

### 8.1 Таблиці розділів 1–2

| № | Зміст | Модель / результат | Джерело та відтворення |
|---|---|---|---|
| 1.1 | терміни ресурсів та інтерфейси керування | системний синтез | авторська таблиця; формули 1.5, 2.6, 3.3–3.5; окремого генератора немає |
| 1.2 | критерії консистентності, помилки й сліпі зони метрик | огляд + endpoint taxonomy | [94, 127–129, 133] та метрики LoopMom/E6; авторська синтезована таблиця |
| 2.1 | ролі наборів даних | усі гілки | авторська evidence-role таблиця; звірити з §2.1 і 3.5 |
| 2.2 | межа реверсивності TempVerseFormer | TempVerseFormer | аудит статті та публічного коду; ручна таблиця, не експериментальний результат |
| 2.3 | FID/MSE/SSIM для режимів обертання | TempVerseFormer, TempFormer, Transformer, LSTM | опублікована стаття TempVerseFormer; точкові оцінки без нових CI; окремого локального table generator немає |
| 2.4 | OpenSTL порівняння | TempVerseFormer і baselines | опублікована таблиця статті; публічний repo для повторного запуску, але не для припущення byte-identical history |
| 2.5 | пам’ять GPU на 1/4/16 кроках | TempVerseFormer і baselines | опублікована стаття/рисунок; ручне перенесення; далекі точки до 1024 не мають exact raw coordinates |
| 2.6 | BPTT/checkpoint/replay на 10–160 кроках | детерміноване багаторівневе recurrent core | дисертаційне локальне профілювання RTX 4060 Ti; окремого immutable T1–T4 receipt не збережено, тому таблиця є supporting, а не центральним frozen claim |
| 2.7 | вибір часової стратегії | BPTT, checkpointing, reversibility | авторський синтез з літератури й вимірювань; без генератора |
| 2.8 | компоненти V-BLT і рівень доказовості | V-BLT | paper/code audit; ручна таблиця розмежування implemented/proposed |
| 2.9 | model size і GFLOPs V-BLT/ViViT/UNet-3D | V-BLT paper | опубліковані точкові оцінки; рисунок 2.18 створює `scripts/make_restructured_ch2_ch4_figures.py` з цих чисел |
| 2.10 | MSE/SSIM/PSNR на 2-D/3-D | V-BLT paper | опублікована таблиця; mixed 3-D ranking зберігати |

### 8.2 Таблиці розділу 3

| № | Зміст | Основа | Джерело та відтворення |
|---|---|---|---|
| 3.1 | поля контекстного пакета | основний context model | формули 3.1–3.5, конфігурації/`ArmSpec`; ручний словник інтерфейсу |
| 3.2 | frozen latent interface | дворівневий codec | реалізація codec у `hvp_rev/rhvd.py` та executed configs; ручна специфікація |
| 3.3 | набори даних і evidence role | усі експерименти | авторська карта; не ранжування difficulty |

### 8.3 Таблиці розділу 4

| № | Зміст | Первинний результат | Скрипт / перевірка |
|---|---|---|---|
| 4.1 | regression vs diffusion mechanism selection | frozen one-step mechanism study | числа з terminal mechanism receipts; expanded table Е.1; рисунок 4.1 задає інтерфейси, а повний монтаж Е.1 лишається якісною ілюстрацією |
| 4.2 | full E2 reach-band endpoint suite | `artifacts/confirmation_e2/results.json` | `scripts/analyze_e2_confirmation.py`; кінцеві числа перевіряє claim/text QA |
| 4.3 | 48-frame token/payload arithmetic | definitions, not model scores | direct calculation from 16/64 tokens and 16 coefficients; no result analyzer required |
| 4.4 | five-seed head × package factorial | E4/E4b | `scripts/analyze_e4_factorial.py`, `scripts/analyze_e4b_factorial.py`; artifact paths in §5.3 |
| 4.5 | E6 relative and absolute endpoints | `artifacts/phasec_v2/results.json` | `scripts/analyze_phasec_v2.py`, `scripts/validate_phasec_completion.py` |
| 4.6 | integrated quality vs OpenSTL | `simvp_packed_pixel_finetune_s0_metrics.json`, `openstl_baseline_comparison.json` | `scripts/train_fixed_region_adapter.py`, `scripts/finetune_simvp_on_packed_context.py`; external baselines partly published/local, so no causal rank claim |
| 4.7 | integrated peak incremental memory by horizon | `integrated_reversible_openstl_memory.json` | `scripts/profile_integrated_reversible_openstl.py`; same-model BPTT/reversible comparison |

### 8.4 Таблиці додатків

| № | Зміст | Походження | Правило оновлення |
|---|---|---|---|
| А.1 | публікації, результати, внесок | статті [A1–A4] + авторська attribution | статус [A4] і склад авторів потребують даних автора; інтеграцію не приписувати статті |
| Б.1 | адміністративні записи | `ADMIN` | заповнювати лише за офіційними документами автора/ВНТУ |
| В.1 | comparison licences | evidence registry | синхронізувати з `APPENDIX_CH4_CONFIGURATION_AND_EVIDENCE_REGISTRY.md` |
| В.2 | 26 executed LoopMom arms | configs, result registry | ручне компактне представлення повної registry; row IDs LM01–LM26 стабільні |
| В.3 | structurally absent/closed cells | authorization ledger | не позначати як pending; E9 лишається skipped |
| В.4 | component/configuration matrix | experiments + artifacts | ручний join; не трактувати як full factorial |
| В.5 | evidence map for context shape | E2/E4/B1 diagnostics | comparison licence є частиною результату |
| В.6 | applicability matrix | synthesis of chapters 2–4 | не додавати universal recommendations without evidence |
| В.7 | experiment codes, questions, evidence classes | authorization/evidence registry | `experiments/authorization.json`, `APPENDIX_CH4_CONFIGURATION_AND_EVIDENCE_REGISTRY.md`; перенесено з головного тексту як довідковий індекс |
| В.8 | exact package/context/target/total-token and payload accounting | усі LoopMom arms та B1 | конфігураційний реєстр + формули 3.3–3.5, 3.31–3.32; повний довідковий контракт, перенесений із §3.4.3 |
| Г.1 | human-readable ↔ machine-readable evidence index | artifact paths and hashes | primary navigation table; update when filenames/hashes change |
| Д.1 | full simultaneous intervals for sampler/codec | E7c/E7a | analysis scripts in §5.3; positive sign follows registered candidate, including flat at 2 432 |
| Д.2 | повний реєстр метрик, популяцій і порожніх випадків | LoopMom/E6 evaluators | код метрик у `hvp_rev`, data card, формули 3.39–3.45; довідковий контракт реалізації |
| Д.3 | exact discrete frontier results with simultaneous intervals | E2 | `analyze_e2_confirmation.py`; `scripts/make_rft_final_tables.py`; не інтерполювати між перевіреними бюджетами |
| Е.1 | expanded mechanism-selection results | frozen mechanism study | supporting material; NFE sensitivity uses smaller frozen pool and cannot be pooled with main evaluation |
| Ж.1 | public repositories | GitHub repos [171–173] | record exact commit/environment for any new reproduction |
| Ж.2 | integrated software components/tests | current worktree | `fixed_region_revformer.py`, `packed_simvp.py`, `reversible_rollout.py`; tests named in table |
| Ж.3 | descriptive memory vs OpenSTL | `openstl_baseline_comparison.json` + integrated profiles | only first two rows are same-model causal contrast; horizons differ for external baselines |

## 9. Повний реєстр рисунків

У головному тексті рівно 77 нумерованих рисунків: 25 у розділі 1, 19 у розділі 2, 19 у розділі 3 і 14 у розділі 4. Додатки містять ще чотири рисунки: довідкові В.1–В.2 та повні монтажі Е.1–Е.2. Отже, весь рукопис містить 81 підпис рисунка. Композитні рисунки 1.6 і 2.17 містять по два файли, але мають один номер. Точні SHA-256, розміри, спосіб вилучення та права зберігаються у:

- `docs/figures/restructured/figure_provenance.json` – розділ 1;
- `docs/figures/restructured/ch2_ch4_figure_provenance.json` – нові й опубліковані активи розділів 2–4;
- `docs/figures/restructured/existing_figure_provenance.json` – раніше наявні активи репозитарію;
- `figures/restructured/integrated_figure_provenance.json` – рисунки 3.18, 3.19 і 4.12.

Не змінювати опубліковані рисунки TempVerseFormer і V-BLT під виглядом косметичної правки. Для них дозволено змінити масштаб, зовнішні поля та підпис у дисертації, але не внутрішній зміст. Власні концептуальні схеми можна регенерувати, якщо збережено їхню семантику й оновлено provenance.

### 9.1 Рисунки розділу 1

| № | Файл | Що показує і на чому ґрунтується | Створення / доказовий статус |
|---|---|---|---|
| 1.1 | `docs/figures/restructured/ch1_fig1_1_system_boundary.png` | system boundary формул 1.1–1.3 | `scripts/make_restructured_dissertation_figures.py`; авторська `ILL/DEF`, не дані |
| 1.2 | `.../ch1_fig1_2_hidden_viewport.png` | прихований світ, viewport і occlusion | той самий скрипт; концепт LoopMom/POMDP |
| 1.3 | `.../ch1_fig1_3_teacher_self_fed.png` | teacher-forced vs self-fed recurrence | той самий скрипт; пояснення формули 1.4 |
| 1.4 | `.../ch1_fig1_4_resource_locations.png` | parameters/optimizer/activations/context/transient | той самий скрипт; схема розкладу 1.5 |
| 1.5 | `docs/figures/fig31_consistency_landscape.png` | паралельна перевірка прогнозів за незалежними критеріями консистентності | `scripts/make_dissertation_foundation_figures.py`; умовна матриця, не траєкторія оптимізації і не емпірична шкала |
| 1.6 | `.../published_2024_fig1_sdxl.png` + `.../ch1_fig1_6b_video_prediction_example.png` | складений рисунок: опубліковані авторські SDXL-приклади та умовний рух/зникнення/повторна поява | першу панель вилучено `make_restructured_dissertation_figures.py` з official PDF, CC BY 4.0; другу створено тим самим скриптом; обидві `ILL` |
| 1.7 | `.../ch1_fig1_6_consistency_taxonomy.png` | overlapping consistency criteria | той самий скрипт; синтез дисертації й [133] |
| 1.8 | `.../published_2024_fig5_visual_perception.png` | local visual input vs integrated scene understanding | вилучено з [133]; author-created, концептуально за [174], CC BY 4.0 |
| 1.9 | `.../ch1_fig1_8_failure_montage.png` | різні довгогоризонтні відмови | `make_restructured_dissertation_figures.py`; illustrative montage |
| 1.10 | `.../ch1_fig1_8_trajectory_consistency.png` | smoothness vs trajectory vs re-entry identity | той самий скрипт; формула 1.9 і E6 taxonomy |
| 1.11 | `.../ch1_fig1_9_cross_level_consistency.png` | compatible/conflicting coarse-fine predictions | той самий скрипт; формула 1.11, не measured cross-level experiment |
| 1.12 | `.../ch1_fig1_7_static_to_video.png` | перехід від static control до video criteria | той самий скрипт; synthesis |
| 1.13 | `.../published_2024_fig2_forward_diffusion.png` | forward diffusion | вилучено з [133] без зміни; `make_restructured_dissertation_figures.py` |
| 1.14 | `.../published_2024_fig3_reverse_diffusion.png` | reverse diffusion | так само, original figure 3 |
| 1.15 | `.../published_2024_fig4_diffusion_pipeline.png` | latent-diffusion pipeline | так само, original figure 4 |
| 1.16 | `.../published_2024_fig6_img2img.png` | image-to-image conditioning | так само, original figure 6 |
| 1.17 | `.../published_2024_fig7_controlnet.png` | geometry conditioning/ControlNet | так само, original figure 7 |
| 1.18 | `.../published_2024_fig8_inversion.png` | inversion-based control | так само, original figure 8 |
| 1.19 | `.../published_2024_fig9_stylealigned.jpg` | shared style / attention | так само, original figure 9 |
| 1.20 | `.../ch1_fig1_11_control_interfaces.png` | п’ять interfaces of consistency control | `make_restructured_dissertation_figures.py`; author synthesis |
| 1.21 | `.../ch1_fig1_12_support_identity_bridge.png` | support і identity як незалежні осі | той самий скрипт; мотивує E6, не є E6 data |
| 1.22 | `.../ch1_fig1_15_activation_tape.png` | activation tape growth | той самий скрипт; формула 1.12/TempVerse concept |
| 1.23 | `docs/figures/fig35_fixed_grid_tokenization.png` | fixed grid on nonuniform scene | `scripts/make_dissertation_foundation_figures.py`; V-BLT motivation |
| 1.24 | `docs/figures/fig39_flat_long_context_barrier.png` | relevant history beyond flat window | той самий скрипт; LoopMom motivation |
| 1.25 | `.../ch1_fig1_18_systems_map.png` | bottlenecks, interventions, evidence boundaries | `scripts/make_restructured_ch2_ch4_figures.py`; high-level synthesis |

### 9.2 Рисунки розділу 2

| № | Файл | Основа | Створення / статус |
|---|---|---|---|
| 2.1 | `docs/figures/restructured/ch2_fig2_1_vblt_2d_sequence_published.png` | V-BLT paper, 2-D sequence, PDF p.7 | `scripts/make_restructured_ch2_ch4_figures.py` вилучив published panel; не регенерувати як нові дані |
| 2.2 | `.../ch2_fig2_2_vblt_3d_sequence_published.png` | V-BLT paper, 3-D polycube, PDF p.8 | той самий extraction workflow |
| 2.3 | `.../ch2_fig2_3_dataset_ladder.png` | roles of datasets in dissertation | той самий скрипт; author categorical map |
| 2.4 | `.../ch2_fig2_4_storage_schedules.png` | BPTT/checkpoint/reversible schedules | той самий скрипт; conceptual, equations 2.4–2.15 |
| 2.5 | `.../ch2_fig2_5_tempverse_reversible_block_published.png` | published TempVerse reversible block, paper p.9 | copied from `TempVerseFormer/resources/figures/rev-block-final.png`; preserve internal artwork |
| 2.6 | `.../ch2_fig2_6_tempverse_architecture_published.png` | full TempVerseFormer, paper p.12 | source `tempverse-arch-final.png`; published artwork unchanged |
| 2.7 | `.../ch2_fig2_7_tempverse_chaining_published.png` | temporal chaining, paper p.17 | source `chain-example-final.png`; published artwork unchanged |
| 2.8 | `.../ch2_fig2_8_tempverse_time_agnostic_published.png` | reversible backward procedure, paper p.20 | extracted from supplied PDF by `make_restructured_ch2_ch4_figures.py` |
| 2.9 | `.../ch2_fig2_16_tempverse_combined_fid_published.png` | published combined-motion FID, paper p.31 | extracted published result; точкові криві, no added CI |
| 2.10 | `.../ch2_fig2_18_tempverse_memory_published.png` | published memory vs horizon, paper p.32 | extracted result; exact quantitative claims only for tabulated 1/4/16 points |
| 2.11 | `.../ch2_fig2_19_tempverse_time_published.png` | published time vs horizon, paper p.33 | extracted result; configuration-specific |
| 2.12 | `.../ch2_fig2_20_fixed_patch_flow.png` | V-BLT local-global-local tensor flow | `make_restructured_ch2_ch4_figures.py`; author redrawing from method, not result |
| 2.13 | `.../ch2_fig2_21_vblt_architecture_published.png` | V-BLT paper p.4 | extracted published architecture; preserve |
| 2.14 | `.../ch2_fig2_22_vblt_causal_mask_published.png` | V-BLT paper p.6 | extracted published mask; preserve |
| 2.15 | `docs/figures/fig47_rope_nd.png` | axis-disjoint RoPE-ND subspaces | `scripts/make_dissertation_foundation_figures.py`; explanatory derivation 2.35–2.39 |
| 2.16 | `.../ch2_fig2_24_entropy_pipeline.png` | logits→softmax→entropy→channel aggregation | `make_restructured_ch2_ch4_figures.py`; based on V-BLT code and 2.40–2.41 |
| 2.17 | `.../ch2_fig2_25a_vblt_2d_entropy_published.jpeg` + `.../ch2_fig2_25b_vblt_3d_entropy_published.jpeg` | складений рисунок із 2-D та 3-D ентропійними панелями V-BLT, paper pp. 9–10 | extracted published qualitative results; внутрішній зміст панелей не змінювати |
| 2.18 | `.../ch2_fig2_26_vblt_point_estimates.png` | published values of tables 2.9–2.10 | `make_restructured_ch2_ch4_figures.py`; author-generated visualization of published numbers |
| 2.19 | `.../ch2_fig2_27_measured_vs_proposed.png` | implemented fixed V-BLT vs proposed dynamic partition | той самий скрипт; left side `PUB`, right side explicitly hypothetical `DEF` |

### 9.3 Рисунки розділу 3

| № | Файл | Основа | Створення / статус |
|---|---|---|---|
| 3.1 | `docs/figures/restructured/ch3_fig3_1_reentry_memory.png` | object leaves recent fine context and re-enters | `scripts/make_restructured_ch2_ch4_figures.py`; conceptual bridge to LoopMom |
| 3.2 | `docs/figures/fig40_context_package_comparison.png` | dense flat, spatial hierarchy, temporal subsampling | `scripts/make_dissertation_foundation_figures.py`; package definitions, not result |
| 3.3 | `.../ch3_fig3_3_reach_fidelity_axes.png` | три способи розподілу однакового бюджету між давньою історією та недавньою деталлю | `make_restructured_ch2_ch4_figures.py`; структурне порівняння пакетів, **не емпірична безперервна крива** |
| 3.4 | `.../ch3_fig3_4_codec_dimensions.png` | exact 4×4/8×8 latent shapes and token counts | той самий скрипт; based on frozen codec interface |
| 3.5 | `docs/figures/fig22_token_sequence_layout.png` | exact context/target token ordering | `scripts/make_dissertation_missing_figures.py`; `build_existing_figure_provenance.py` records hash |
| 3.6 | `.../ch3_fig3_6_joint_attention.png` | joint attention in one window vs rollout causality | `make_restructured_ch2_ch4_figures.py`; model structure |
| 3.7 | `.../ch3_fig3_7_gradient_paths.png` | frozen, detached and trained paths | той самий скрипт; формула 3.16/code contract |
| 3.8 | `.../ch3_fig3_8_objective_interfaces.png` | diffusion, regression, EBT target interfaces | той самий скрипт; comparison taxonomy |
| 3.9 | `docs/figures/fig16_context_packaging_architecture.png` | full frozen-codec→WindowDiT pipeline | `scripts/make_rft_core_schematics.py`; active architecture, not superseded macro-first design |
| 3.10 | `.../ch3_fig3_10_ddim_timeline.png` | deterministic DDIM timeline and NFE | `make_restructured_ch2_ch4_figures.py`; формули 3.25–3.27 та текстовий підрахунок NFE |
| 3.11 | `docs/figures/fig23_rollout_ring_buffer.png` | coarse/fine ring-buffer update | `scripts/make_dissertation_missing_figures.py`; формула 3.29 |
| 3.12 | `.../ch3_fig3_12_anchor_geometry.png` | exact anchors at four-frame target boundary | `make_restructured_ch2_ch4_figures.py`; формула 3.30/config registry |
| 3.13 | `.../ch3_fig3_13_token_costs.png` | context/target/total token and payload arithmetic | той самий скрипт; deterministic calculation, not GPU profile |
| 3.14 | `docs/figures/fig15_loopmom_observability_schematic.png` | LoopMom world, viewport and reach band | `scripts/make_rft_core_schematics.py`; analytic observability |
| 3.15 | `.../ch3_fig3_15_reentry_storyboard.png` | multi-frame disappearance/re-entry event | `make_restructured_ch2_ch4_figures.py`; illustrative event |
| 3.16 | `.../ch3_fig3_16_reach_population.png` | pair-specific eligibility filter | той самий скрипт; формули 3.34–3.38 |
| 3.17 | `.../ch3_fig3_17_persistent_assignment.png` | Hungarian matching and persistent ID tracking | той самий скрипт; E6 metric protocol |
| 3.18 | `figures/restructured/ch3_fig3_18_integrated_architecture.svg` | fixed-width packed state, SimVP transition, reversible chain | dissertation-native manual deterministic SVG; no standalone generator found; provenance hash in `integrated_figure_provenance.json` |
| 3.19 | `figures/restructured/ch3_fig3_19_fixed_regions.svg` | 16 coarse cells + 8 fixed fine regions | dissertation-native manual deterministic SVG; based on formulas 3.46–3.48, not dynamic routing |

### 9.4 Рисунки розділу 4

| № | Файл | Дані / модель | Створення / допустиме читання |
|---|---|---|---|
| 4.1 | `docs/figures/fig24_generation_objectives.png` | regression/diffusion/EBT interfaces | `scripts/make_dissertation_missing_figures.py`; protocol schematic |
| 4.2 | `.../ch4_fig4_2_ebt_composite.png` | EBT ranking and cold-descent diagnostics | `scripts/make_restructured_ch2_ch4_figures.py` from `logs_rhcp/ebt_scaled_s1.log` and `ebt_scaled_s2.log`; diagnostic only |
| 4.3 | `docs/figures/fig28_memory_gate_optimization.png` | four development training logs, 6k vs 20k updates | `scripts/make_dissertation_missing_figures.py`; optimization gate, not untouched evidence |
| 4.4 | `.../ch4_fig4_4_reach_effects.png` | E2 full-population vs pair-specific reach effects | `make_restructured_ch2_ch4_figures.py` from frozen E2 values; validate numbers against `artifacts/confirmation_e2/results.json` |
| 4.5 | `docs/figures/article_confirmed_frontier.png` | discrete simultaneous E2 frontier + token arithmetic | `scripts/make_rft_result_figures.py`; E2 artifact; no lines/interpolation implying a fitted crossover |
| 4.6 | `docs/figures/fig17_head_package_seed_heterogeneity.png` | E4b five-seed factorial | `scripts/make_rft_result_figures.py` from `artifacts/e4b_head_package/results.json` |
| 4.7 | `docs/figures/fig19_sampler_nfe_robustness.png` | E7c 20 effects | той самий скрипт from `artifacts/e7c_sampler_nfe/results.json`; two upper intervals unresolved |
| 4.8 | `docs/figures/fig20_alternate_codec_robustness.png` | original vs independently trained codec | той самий скрипт from `artifacts/e7a_codec_robustness/results.json`; conditional on predictor seed 0 |
| 4.9 | `docs/figures/fig18_trust_negative_replication.png` | E5a early/late effects | той самий скрипт from `artifacts/e5a_trust_replication/results.json`; late boundary failed |
| 4.10 | `docs/figures/fig21_pose_indexed_raw_memory.png` | E8a raw K6/K14/K38/K48 reference | той самий скрипт from `artifacts/e8a_pose_mosaic/results.json`; privileged reference, not token-matched model |
| 4.11 | `docs/figures/fig22_phasec_v2_boundary.png` | E6 support/trajectory/coverage/identity | той самий скрипт from `artifacts/phasec_v2/results.json`; external boundary with low absolute performance |
| 4.12 | `figures/restructured/ch4_fig4_12_integrated_memory.svg` | same-model reversible vs BPTT horizon profile | dissertation-native manual SVG from `integrated_reversible_openstl_memory.json` |
| 4.13 | `docs/figures/restructured/ch4_fig4_13_two_resource_axes.png` | context token arithmetic + activation-memory results | `scripts/make_restructured_ch2_ch4_figures.py`; synthesis of separately measured axes; ratios never multiplied |
| 4.14 | `.../ch4_fig4_14_decision_flow.png` | decision flow by limiting resource | той самий скрипт; design rule, not trained classifier or fitted policy |

### 9.5 Рисунки додатків і узгодження нумерації активів

| № | Файл | Призначення | Створення / статус |
|---|---|---|---|
| В.1 | `docs/figures/fig63_executed_configuration_matrix.png` | виконане покриття й структурно відсутні комірки | `scripts/make_dissertation_foundation_figures.py`; довідковий індекс, не effect size |
| В.2 | `docs/figures/restructured/appv_figv_2_equal_reach.png` | точна вартість трьох пакетів за досяжності 48 кадрів | `scripts/make_restructured_ch2_ch4_figures.py`; арифметика токенів і fp16 payload, не VRAM profile |
| Е.1 | `docs/figures/fig29_mechanism_single_step_samples.png` | повний fixed-index монтаж regression/diffusion | `scripts/make_dissertation_missing_figures.py`; якісна ілюстрація, таблиця 4.1 визначає вибір |
| Е.2 | `docs/figures/fig7_samples_claim1.png` | повний fixed-index монтаж чотирьох E2 rollout clips | `scripts/make_samples.py`; якісна ілюстрація, таблиця Д.3 містить агрегований доказ |

Актив, Markdown і provenance інтеграційного профілю тепер узгоджено під номером 4.12 і basename `ch4_fig4_12_integrated_memory.svg`. Дві компактні галереї вилучено з головного тексту; повні монтажі Е.1–Е.2 збережено лише в додатках із поясненням рядків і стовпців.

## 10. Артефакти, скрипти й команди відтворення

### 10.1 Центральна доказова лінія LoopMom

| Етап | Скрипт / код | Вихід |
|---|---|---|
| Генерація захищеної E2-вибірки | `scripts/generate_confirmation_set.ps1`, `scripts/generate_loopmom_confirmation.py` | `artifacts/confirmation_e2/dataset_manifest_2026-07-12.json` і захищені кліпи |
| Запуск E2 | `scripts/run_e2_confirmation.ps1` | вирівняні prediction/metric records для frozen checkpoints |
| Аналіз E2 | `scripts/analyze_e2_confirmation.py` | `artifacts/confirmation_e2/results.json` |
| Перевірка завершеності E2 | `scripts/validate_e2_completion.py`, `scripts/validate_e2_design.py` | completion validation / fail-closed verdict |
| E4/E4b | `run_e4_factorial.ps1`, `run_e4b_factorial.ps1`, `analyze_e4_factorial.py`, `analyze_e4b_factorial.py` | `artifacts/e4_head_package/results.json`, `artifacts/e4b_head_package/results.json` |
| E5a | `run_e5a_trust_replication.ps1`, `analyze_e5a_trust_replication.py` | `artifacts/e5a_trust_replication/results.json` |
| E7c | `run_e7c_sampler_nfe.ps1`, `analyze_e7c_sampler_nfe.py` | `artifacts/e7c_sampler_nfe/results.json` |
| E7a | `run_e7a_codec_robustness.ps1`, `analyze_e7a_codec_robustness.py` | `artifacts/e7a_codec_robustness/results.json` |
| E8a | `run_e8a_reference.ps1`, `evaluate_e8a_pose_mosaic.py`, `analyze_e8a_pose_mosaic.py` | `artifacts/e8a_pose_mosaic/results.json` |
| E6 | `run_phasec_v2.ps1`, `analyze_phasec_v2.py`, `validate_phasec_completion.py` | `artifacts/phasec_v2/results.json`, completion і W&B audit receipts |
| Узагальнені таблиці | `scripts/make_rft_final_tables.py` | manuscript-ready таблиці / consistency inputs |
| Результуючі графіки | `scripts/make_rft_result_figures.py` | frontier, E4b, E5a, E7a/c, E8a, E6 figures |

Запуски зареєстрованих експериментів не потрібно повторювати для редакційної перевірки. Вони fail-closed через `experiments/authorization.json`. Новий запуск має бути зареєстрований як reproduction; не можна просто змінити статус завершеного експерименту.

### 10.2 Генератори ілюстрацій

| Скрипт | Створює |
|---|---|
| `scripts/make_restructured_dissertation_figures.py` | авторські схеми розділу 1, composite 1.6, extraction опублікованих рисунків 2024 року, `figure_provenance.json` |
| `scripts/make_restructured_ch2_ch4_figures.py` | більшість нових схем розділів 2–4, extraction TempVerseFormer/V-BLT panels, графіки з опублікованих point estimates, `ch2_ch4_figure_provenance.json` |
| `scripts/make_dissertation_foundation_figures.py` | foundation figures `fig31`, `fig35`, `fig39`, `fig40`, `fig47`, `fig63` |
| `scripts/make_dissertation_missing_figures.py` | token layout, ring buffer, generation objectives, mechanism samples, memory-gate figure |
| `scripts/make_rft_core_schematics.py` | основна context architecture і LoopMom observability schematic |
| `scripts/make_rft_result_figures.py` | artifact-derived result figures 4.5–4.11 |
| `scripts/make_samples.py` | fixed-index qualitative E2 rollouts (рисунок Е.2) |
| `scripts/build_existing_figure_provenance.py` | hashes/provenance for figures that predate restructured generators |

Базова команда виконується з кореня репозитарію, наприклад:

```powershell
python scripts/make_rft_result_figures.py
python scripts/build_existing_figure_provenance.py
```

Перед масовою регенерацією потрібно переглянути diff активів. Частина скриптів історично знала стару нумерацію, а published panels не повинні бути візуально перемальовані.

### 10.3 Інтегрований прототип і зовнішній архів

Великі checkpoints і частина числових квитанцій зберігаються поза Git у `D:\hierarchical-video-prediction-storage\feasibility\`.

| Файл | SHA-256 | Створює / використовує | Роль |
|---|---|---|---|
| `fixed_region_adapter_blended_mmnist_s0_metrics.json` | `AC9BF2BAA404A023C060434A4C17B99B47F14C9CD7270655DF4062E1A0A3A294` | `scripts/train_fixed_region_adapter.py` | gate просторового reconstructor |
| `simvp_packed_pixel_finetune_s0_metrics.json` | `1C36D65632FA46AF0202F080D72CB486D1C0384B426BE0A845E683F6F44E8E8C` | `scripts/finetune_simvp_on_packed_context.py` | MSE 28,72 / MAE 80,27 на 10 000 clips |
| `openstl_baseline_comparison.json` | `96E3E6672E9D85A1657DB844175D6426D16EE419CE5F3BA85AD6677DE6C8AB9A` | зведено з local profiles і OpenSTL table | таблиця 4.6 і Ж.3; external rows descriptive |
| `integrated_reversible_openstl_memory.json` | `FEDD7547A4C79BAC61C2E009EE742E27C5CE5A81A698508AC20042C4061B0AC9` | `scripts/profile_integrated_reversible_openstl.py` | batch-1 horizon profile, loss/gradient/reconstruction checks |
| `integrated_reversible_openstl_memory_b2.json` | `F8D6095A4DD70F59B18EC96982BC290030A721C6E4DFBF03C9B34CC7BB7A9997` | той самий profiler | matched batch-2 20-frame point |
| `integrated_feasibility_manifest_20260823.json` | `32098422B4383F499A8E9101FFD42872CCAAC44DAFF6227FC2FAFBAAAF0B3CDC` | manifest build during feasibility closure | stable index selected receipts/checkpoints |

Кодовий шлях:

- `hvp_rev/fixed_region_revformer.py` – fixed coarse/fine regions і reconstructor;
- `hvp_rev/packed_simvp.py` – адаптація packed state до SimVP;
- `hvp_rev/reversible_rollout.py` – algebraic replay;
- `scripts/train_fixed_region_adapter.py` – adapter training;
- `scripts/finetune_simvp_on_packed_context.py` – selected quality run;
- `scripts/profile_fixed_region_memory.py` – token-core/OpenSTL one-step profiles;
- `scripts/profile_integrated_reversible_openstl.py` – same-model horizon profile;
- `tests/test_fixed_region_revformer.py`, `tests/test_packed_simvp.py`, `tests/test_reversible_rollout.py` – unit/contract tests.

Рисунок 4.12 створено з `integrated_reversible_openstl_memory.json`. Таблиця 4.7 використовує ті самі значення. Якщо D:-диск недоступний іншому агенту, hashes і числа в рукописі можна перевірити за таблицею Г.1, але повне незалежне відтворення потребує передачі архіву або публічного downloader.

### 10.4 Опубліковані TempVerseFormer і V-BLT матеріали

TempVerseFormer figures 2.5–2.11 походять із:

- `C:\Users\leoni\Documents\projects\TempVerseFormer\resources\figures\` для оригінальних architecture diagrams;
- `C:\Users\leoni\Downloads\Temporal_Modeling_with_Reversible_Transformers__NCAv2_.pdf` для published plots/panels.

V-BLT figures 2.1–2.2, 2.13–2.14 і 2.17 походять із:

- `C:\Users\leoni\Downloads\Article_LKulyk_OMokin_124_2025_mt.pdf`;
- публічного репозитарію `visionBLT` для code-level interpretation.

Абсолютний локальний шлях у provenance є історичним шляхом вилучення, а не вимогою до майбутньої збірки. Для переносимості бажано зберігати source PDF або stable DOI й hash. Не замінювати published numbers новим запуском без чіткого маркування «reproduction».

### 10.5 Перевірка тексту, посилань і збірки

```powershell
python scripts/audit_ukrainian_translation_parity.py
python scripts/audit_compact_dissertation.py
python scripts/check_text_consistency.py --gate
python scripts/check_rft_consistency.py
python scripts/check_manuscript_finalization.py
python scripts/validate_scientific_configs.py
python scripts/release_check.py
```

- `audit_ukrainian_translation_parity.py` порівнює структуру українського й англійського стислих наборів, але не замінює semantic review;
- `check_text_consistency.py --gate` шукає stale claims/statuses;
- `check_rft_consistency.py` перевіряє четверту статтю та headline values;
- `check_manuscript_finalization.py` перевіряє рисунки, placeholders і assembly inputs;
- `validate_scientific_configs.py` валідує fail-closed experiment configs;
- `release_check.py` запускає релізний CPU gate.

Скрипти складання DOCX/PDF існують, але поточним джерелом змісту є Markdown. Старий PDF/DOCX не стає авторитетнішим лише через завершене форматування.

## 11. Як виправляти наукове твердження

Для кожного сумнівного речення пройти послідовність:

1. Визначити його тип за §4 (`LIT`, `PUB`, `DEF`, `DER`, `EMP`, `SYN`, `ILL`, `ADMIN`).
2. Якщо є число, знайти первинний JSON/PDF, а не сусідню таблицю в рукописі.
3. Для `EMP` перевірити candidate/comparator, знак ефекту, population, seed, sampler, codec і тип інтервалу.
4. Для `PUB` перевірити, чи число опубліковане, чи є новою локальною reproduction.
5. Для `DEF/DER` перерахувати одиниці та звірити з кодом.
6. Перевірити comparison licence. `WHOLE_CONFIGURATION`, `NEAR_MATCHED`, `PRIVILEGED_REFERENCE` і `EXTERNAL_BOUNDARY` не можна мовно підсилювати до `MATCHED`.
7. Перевірити, чи той самий висновок повторено в анотації, вступі, висновках і таблиці А.1.
8. Після правки запустити QA з §10.5 і переглянути Git diff вручну.

## 12. Критичні правила інтерпретації

### 12.1 Frontier

- 896: hierarchy confirmed, seeds 0–1.
- 1 152: hierarchy confirmed, seeds 0–2.
- 1 792: unresolved.
- 2 432: flat confirmed in primary frontier; some robustness variants unresolved family-wise.
- 3 072: flat confirmed, seeds 0–1.
- Не існує fitted continuous crossing, monotonic law або exact threshold.

### 12.2 B1

B1 змінює не лише stride/reach, а й target interface, target tokens, auxiliary loss і optimization load. Допустиме формулювання: «перевага повної конфігурації». Заборонене: «чистий причинний ефект stride».

### 12.3 E4b

Diffusion temporal-package effect стабільний у п’яти seeds. Regression effect і interaction змінюють знак. Не узагальнювати до незалежного від голови закону.

### 12.4 E5a

Ранній hierarchy effect відтворено; пізню запропоновану trust boundary не відтворено. Не повертати стару назву/заяву про стабільну 24-frame boundary.

### 12.5 E6

Hierarchy improves spatial support/continuity; flat improves digit identity. Абсолютні identity/coverage values лишаються низькими. Це external boundary, не natural-video generalization і не direct replication of LoopMom.

### 12.6 TempVerseFormer

Реверсивним є повторюваний doubled Transformer state, а не весь VAE-to-VAE pipeline. Constant activation-memory statement стосується виміряного recurrent core/contract і має враховувати anchors, RNG metadata, nonreversible boundaries та workspace.

### 12.7 V-BLT

Fixed patches, RoPE-ND, byte classification і entropy maps реалізовані. Dynamic entropy-based region allocation не реалізований і не перевірений. Інтегрований prototype використовує fixed regions і variance-like weighting, а не підтверджений entropy router.

### 12.8 Інтеграція

Не включати integration experiment до четвертої статті. Не приписувати йому 48-frame LoopMom result. Не заявляти multiplicative `10.67× × 2.58×` memory saving: перше є state-payload ratio, друге – same-model incremental peak activation/profile ratio.

## 13. Відомі обмеження та пункти, що потребують даних автора

Наукові центральні claims мають заморожені докази. Для інституційно фінальної версії все ще потрібні:

- точна УДК і затверджене формулювання титульного аркуша, включно з вибором між записом 124/галузь 12 та новішим F4 за документами конкретної освітньо-наукової програми й року вступу;
- офіційні назви, номери, строки й роль здобувача у двох держбюджетних темах;
- повний список шести conference publications;
- фінальний bibliographic status, venue, authorship і contribution statement статті [A4];
- підписані акти впровадження та точні літери додатків;
- page/figure/table/reference counts після інституційного верстання.

Наукові/відтворювальні обмеження, які треба лишити чесно видимими:

- exact historical raw manifests для частини V-BLT/TempVerseFormer запусків не були збережені; public repos дають reproduction path, але не гарантують bitwise historical identity;
- точні raw coordinates далеких TempVerseFormer memory-plot points до 1024 кроків відсутні;
- таблиця 2.6 є локальним supporting profile без окремого frozen T1–T4 immutable report;
- integration receipts локальні на D:-диску; їх потрібно передати або опублікувати для зовнішнього review;
- LoopMom і long Moving-MNIST не замінюють natural-video validation;
- інтегрований prototype має одну seed і fixed regions.

## 14. Рекомендований протокол сліпого аудиту з нуля

### Етап A – структура і сюжет

1. Прочитати вступ, висновки й заголовки розділів без таблиць.
2. Перевірити, чи видно одну проблему і три обмеження, а не три непов’язані дисертації.
3. Переконатися, що розділ 2 готує обмеження, розділ 3 формує центральну модель, а розділ 4 перевіряє її й лише потім показує integration feasibility.

### Етап B – claims

1. Для кожного підрозділу використати карту §6.
2. Кожне число звірити з §5.3/§10.3 або published paper.
3. Порівняти анотацію, вступ, §4 і загальні висновки.
4. Перевірити заборонені підсилення у `docs/CONTENT_FINALIZATION_CLAIM_MATRIX_2026-07-19.md`.

### Етап C – математика

1. Перевірити всі 114 позначок за §7.
2. Звірити dimensions, units і sign conventions.
3. Окремо перевірити `v`-prediction, anchor geometry, payload-vs-VRAM і clip-cluster bootstrap.

### Етап D – таблиці й рисунки

1. Перевірити всі 40 таблиць за §8.
2. Перевірити всі 77 номерів рисунків головного тексту та чотири рисунки додатків за §9 і provenance hashes.
3. Для result figures звірити plotted values з JSON.
4. Для published figures перевірити source/attribution; не оцінювати їх як нові dissertation experiments.
5. Для conceptual figures перевірити логіку стрілок, але не вимагати statistical source.

### Етап E – код і відтворення

1. Прочитати `docs/EXPERIMENT_AND_SCRIPT_INDEX.md`, `docs/REPRODUCIBILITY.md`, додатки Г і Ж.
2. Запустити read-only QA commands §10.5.
3. Перевірити hashes release manifest і D:-receipts.
4. Не запускати GPU studies без нового зареєстрованого reproduction protocol.

## 15. Швидкий машинний контроль повноти цього довідника

З кореня репозитарію:

```powershell
# Формули: очікується 14 + 42 + 53 + 5 = 114.
$files = '02_CHAPTER_1.md','03_CHAPTER_2.md','04_CHAPTER_3.md','05_CHAPTER_4.md'
foreach ($f in $files) {
  $raw = Get-Content "docs/dissertation/condensed_translated_humanized_ukrainian/$f" -Raw
  [regex]::Matches($raw, '\\tag\{([^}]+)\}').Count
}

# Нумеровані рисунки головного тексту: очікується 25, 19, 19, 14.
rg -n '^\*\*Рисунок ' docs/dissertation/condensed_translated_humanized_ukrainian

# Нумеровані таблиці: очікується 40 разом із додатками.
rg -n '^\*\*Таблиця ' docs/dissertation/condensed_translated_humanized_ukrainian

# Image targets, placeholders і manuscript inputs.
python scripts/check_manuscript_finalization.py
```

## 16. Фінальна коротка карта «твердження → доказ»

| Твердження | Мінімальний первинний доказ |
|---|---|
| hierarchy helps when flat cannot address relevant past | E2 pair-specific reach population in `confirmation_e2/results.json` |
| discrete frontier 896/1 152/unresolved 1 792/flat 2 432/3 072 | E2 simultaneous family |
| 2,67–3,43× fewer context tokens | formulas 3.3/4.5 + 16/64 token geometry |
| temporal full-resolution package can beat spatial hierarchy under diffusion | E4b five-seed artifact |
| late trust boundary failed | E5a artifact |
| robustness to sampler/NFE and one alternate codec is directional but qualified | E7c/E7a artifacts |
| raw explicit access is a lower-error reference | E8a artifact |
| hierarchy supports where; flat preserves what | E6 artifact plus absolute metrics |
| reversible temporal state reduces activation storage | TempVerseFormer paper/code + table 2.5; supporting local profile 2.6 |
| V-BLT supplies fixed hierarchical processing and entropy diagnostic | V-BLT paper/code, tables 2.8–2.10, figures 2.13–2.18 |
| integrated principles are technically compatible | D:-integration receipts, tests, tables 4.6–4.7, figure 4.12 |

Цей файл має оновлюватися разом із будь-якою зміною нумерації, артефактного шляху, формули або headline claim. Він навмисно містить і сильні результати, і межі: новому рецензенту потрібна карта не лише того, що можна захищати, а й того, що текст не має права стверджувати.

## 17. Супровідна карта четвертої статті

Авторитетний Markdown статті: `docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md`. Стаття використовує лише центральну LoopMom/E2–E8a/E6 програму. Інтегрований прототип із §3.7/4.8 дисертації до неї не входить.

### 17.1 Змістові блоки

| Розділ статті | Джерело |
|---|---|
| Abstract, Introduction, RQ/contributions | claim matrix і синтез E2/E4b/E5a/E7a/E7c/E8a/E6 |
| Controlled instrument | LoopMom generator, frozen codec, WindowDiT package definitions, B1 disclosure |
| Endpoints and inference | `hvp_rev` metrics, analytic observability, clip-cluster/max-statistic protocol |
| Results 4.1–4.9 | відповідні immutable artifacts з §5.3 |
| Reproducibility | release manifest, authorization/access logs, public `hvp` API |
| Boundaries and conclusion | дозволені/заборонені readings із claim matrix |
| Supplement S1/S2 | executed regimes, NFE sensitivity і complete observed arm registry |

### 17.2 Дев’ять display equations

У статті формули не нумеровані. За порядком появи:

1. package tuple – скорочена формула 3.1;
2. context cost і reach – формули 3.3 і 3.5;
3. conditional future-window distribution – формула 3.19;
4. coarse+fine `v`-prediction loss – формула 3.21;
5. pair-specific reach population – формула 3.38;
6. answerable re-entry fraction – частина означення 3.37;
7. median evidence age – частина означення 3.37;
8. paired record-weighted estimand – формула 3.41;
9. sign harmonization для E6 – спеціалізація правила 4.1.

Жодна з цих формул не створюється окремим article script; вони є компактними копіями реалізованих/визначених інтерфейсів розділу 3 дисертації.

### 17.3 Таблиці статті

| № | Зміст | Походження |
|---|---|---|
| 1 | protocol positioning relative to related work | literature synthesis; ручна таблиця з citations |
| 2 | exact central interface accounting | package configs/formulas 3.1–3.5; direct arithmetic |
| 3 | executed architecture/training/evaluation settings | frozen configs і experiment registry |
| 4 | context payload/materialized embeddings | direct arithmetic, як таблиця 4.3 дисертації |
| 5 | publication-facing study map | authorization/evidence registry |
| 6 | endpoint and assignment rules | metric implementations, формули 3.39–3.45 |
| S1a | common rollout regimes | experiment registry |
| S1b | per-candidate NFE sensitivity | frozen two-batch mechanism receipt; не equal-total-work comparison |
| S2 | complete observed package/interface registry | Appendix registry; absent cells terminal, not pending |

Фінальні evidence tables можуть бути регенеровані `scripts/make_rft_final_tables.py`; після цього необхідно перевірити, що ручна article assembly не втратила qualifiers.

### 17.4 Дев’ять рисунків статті

| № | Файл | Той самий об’єкт у дисертації | Генератор / artifact |
|---|---|---|---|
| 1 | `docs/figures/fig15_loopmom_observability_schematic.png` | 3.14 | `make_rft_core_schematics.py`; conceptual instrument |
| 2 | `docs/figures/fig16_context_packaging_architecture.png` | 3.9 | `make_rft_core_schematics.py`; model schematic |
| 3 | `docs/figures/article_confirmed_frontier.png` | 4.7 | `make_rft_result_figures.py`; E2 |
| 4 | `docs/figures/fig17_head_package_seed_heterogeneity.png` | 4.9 | той самий; E4b |
| 5 | `docs/figures/fig18_trust_negative_replication.png` | 4.12 | той самий; E5a |
| 6 | `docs/figures/fig19_sampler_nfe_robustness.png` | 4.10 | той самий; E7c |
| 7 | `docs/figures/fig20_alternate_codec_robustness.png` | 4.11 | той самий; E7a |
| 8 | `docs/figures/fig21_pose_indexed_raw_memory.png` | 4.13 | той самий; E8a |
| 9 | `docs/figures/fig22_phasec_v2_boundary.png` | 4.14 | той самий; E6 |

Стаття й дисертація повинні посилатися на одні й ті самі image files та immutable artifacts. Розходження чисел між ними є помилкою, а не допустимою редакційною варіацією.
