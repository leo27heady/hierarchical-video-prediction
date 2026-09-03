# Вступ

## Актуальність та обґрунтування теми дослідження

Відеопредиктор часто оцінюють покадрово, хоча багато суттєвих помилок стають помітними лише після накопичення похибок упродовж кількох кроків прогнозування. Рух камери може знову відкрити ділянку, яка вже була видимою, а модель заповнить її правдоподібною, проте не підтвердженою минулим текстурою. Це не просто зменшення різкості зображення. Такий випадок означає, що процес генерації не зберіг інформацію, яку задача вважає інваріантною.

На довгому горизонті ця проблема посилюється з двох причин. По-перше, відео є лише спостереженням за динамічним світом: частина стану може бути закрита, залишити поле огляду або зникнути з короткого контекстного вікна. По-друге, під час авторегресивного розгортання модель отримує власні попередні прогнози як нові входи. Помилка змінює не один кадр, а подальшу траєкторію обчислення. Тому близькість окремого прогнозу до еталона ще не доводить, що система збереже потрібний об’єкт, відновить його після повторної появи або не створить непідтверджений передній план.

Зміст консистентності залежить від задачі. Для спостереження за рухом вирішальною може бути неперервність траєкторії; для повторного розпізнавання – збереження ідентичності; для моделювання огляду сцени – правильна просторова підтримка після повернення камери. Один агрегований показник якості не розрізняє ці відмови. Отже, вимоги до пам’яті моделі треба пов’язувати з тим, яку властивість процесу необхідно зберегти й за якою спостережуваною помилкою це буде перевірено. Доступ до віддаленого стану є передумовою довгогоризонтної траєкторної та об’єктної консистентності, тоді як деталізація представлення визначає, які властивості цього стану збережуться.

Слово «пам’ять» тут позначає кілька різних витрат. Під час навчання зберігаються активації для зворотного проходу. Під час формування прогнозу модель утримує контекст або стан, з якого дістає минулу інформацію. Просторове представлення, своєю чергою, визначає, скільки обчислювальної ємності отримують ділянки різної складності. Зменшення однієї витрати не гарантує зменшення інших. Це не три незалежні теми, а три обмеження одного довгогоризонтного процесу, які потрібно вимірювати на відповідних інтерфейсах.

Попередні дослідження пропонують сильні компоненти, але рідко ізолюють механізм їхньої переваги за сумісних бюджетів. Збереження контрольних точок активацій і реверсивні мережі обмінюють пам’ять на повторні обчислення [30–32, 92, 108–113]. Ієрархічні моделі прихованого простору, рекурентний стан, стиснена пам’ять, пошук, розріджена увага та моделі простору станів розширюють часовий доступ різними способами [14–24, 33–37, 79–83]. Візуальні трансформери на основі патчів і моделі з байтовими представленнями у прихованому просторі змінюють саму одиницю обчислення [69–78]. Сам факт наявності кількох рівнів ще не доводить користі ієрархії. Під час порівняння нерідко одночасно змінюються архітектура, кількість токенів, цільова функція, інтерфейс цілі або оцінювана множина випадків. Тоді кращий показник характеризує всю конфігурацію, а не причину її переваги. Для відокремлення ефекту потрібні зіставлене за бюджетом втручання та точна геометрія доступної історії.

Звідси випливає методологія контрольованого системного дослідження. Консистентність у ній задається критеріями конкретної задачі, ресурси навчання та формування прогнозу обліковуються окремо, пакети контексту залишаються порівнянними, а оцінювані випадки добираються так, щоб досліджуваний механізм можна було ідентифікувати. Окремо враховується статистична залежність кадрів одного авторегресивного розгортання: багато кадрів із того самого кліпу не є багатьма незалежними спостереженнями.

The retained-context study holds the predictive mechanism fixed while changing exact packages of observed evidence. The later OpenSTL-native composition is a separately implemented end-to-end model with physical metadata and an internally reversible core. Their results remain distinct evidence blocks of one manuscript in preparation; neither inherits the other's claimed performance.

Memory is treated as four separately measured resources: retained context/state, intermediate activations, parameters with gradients/optimizer state, and temporary workspace/allocator overhead. Spatial compression can release retained capacity for a longer observed history; temporal allocation decides where that capacity is spent. Reversibility changes the internal training tape, not the age of available evidence. Routing at an unchanged quota changes information allocation rather than producing an additional memory multiplier.

The controlled TRB study asks whether distant observed evidence improves a four-frame prediction endpoint under specified context budgets. This is different from a long self-fed forecast, where prediction errors re-enter memory. Historical rollout evidence remains necessary for that second claim. OpenSTL evaluation asks a separate external question: whether a composed architecture is competitive under matched Moving MNIST and TaxiBJ protocols and provides a measured quality–memory advantage. None of these questions is settled by analytic token counts alone.

## Зв’язок із дослідницькими програмами, планами та темами

Дисертаційну роботу виконано у Вінницькому національному технічному університеті в межах наукового напряму «Системний аналіз та інтелектуальні інформаційні технології». **[ДАНІ ВІД АВТОРА: зазначити офіційну назву, номер, строки виконання та роль здобувача для держбюджетної теми 1.]** **[ДАНІ ВІД АВТОРА: додати відповідні перевірені відомості для держбюджетної теми 2.]** Остаточне формулювання має відповідати підписаним документам установи; назви тем і відповідальність виконавців не виводяться з історії програмного забезпечення.

## Мета та дослідницькі завдання

The aim is to improve task-relevant consistency and resource efficiency of hierarchical video prediction by analyzing temporal computation, spatial representation, retained context, and evaluation. The main architectural focus is retention of relevant distant evidence. Resource efficiency always names the resource saved and the quality/computation cost incurred.

The six dissertation-level tasks are preserved:

1. Analyze prediction as a partially observed dynamic system; distinguish uncertainty and consistency failures; define causal comparison and resource-accounting requirements.
2. Develop and delimit reversible temporal computation; verify state reconstruction and reverse differentiation; characterize the activation-memory/recomputation trade-off.
3. Develop hierarchical raster-byte representation, RoPE-ND, and entropy-based spatial-complexity diagnosis, distinguishing fixed-patch evidence from proposed adaptive allocation.
4. Specify a hierarchical predictor with explicit streams, ages, spatial identities, token cost, target interface, and causal update. The new budget, allocation, expansion, level, ownership, and space–time questions are nested under this task.
5. Establish exact observability, frame/object metrics, clip-aware paired inference, development selection, and protected confirmation. Distinguish distant-context access from autoregressive rollout stability.
6. Evaluate operating regions, robustness, negative cases, and external boundaries; test whether hierarchy, reversible core execution, and spatial routing can compose under a matched OpenSTL quality–memory protocol.

These tasks retain the contributions of Articles [A1]–[A3]; they are not replaced by a list of experiment codes. The new questions H1–H7 in Chapter 1 connect to the exact configurations in Chapter 3 and the evidence cells in Chapter 4. The intended result is a conditional design rule rather than an architecture assumed to win every comparison.

## Об’єкт і предмет дослідження

Об’єктом дослідження є процес нейронного прогнозування відео на довгих горизонтах із самоподачею за обмежених обчислювальних ресурсів і пам’яті. Він охоплює формування, збереження та повторне використання візуальної інформації під час навчання й авторегресивного застосування, коли згенеровані кадри надходять до наступних входів моделі, а релевантний вміст сцени може залишити поточне поле огляду або вікно контексту.

Предметом дослідження є ієрархічні контекстні представлення як основний об’єкт емпіричного зіставлення, а також системні моделі, реверсивні часові перетворення, просторові представлення, позиційні кодування, засоби оцінювання та статистичні процедури, призначені для збереження релевантної до задачі консистентності й визначення робочих областей досяжності та точності відтворення. Предмет охоплює зв’язки між втручанням, ресурсом, який воно безпосередньо змінює, та показниками консистентності, за якими його оцінюють. Загальні твердження про всі генеративні відеосистеми та ефективність застосування в неперевірених сферах до предмета дослідження не належать.

## Методи дослідження

Controlled procedural generation supplies interpretable motion and representation tests. TRB supplies exact viewport/world observability; historical long Moving-MNIST supplies a separate persistent-object rollout boundary. Published OpenSTL results retain their original protocols, while the new composition plans matched Moving MNIST and TaxiBJ evaluation. Hungarian object assignment, clip-aware inference, preregistration, and training-seed reporting remain distinct parts of the evaluation contract [1, 120–126, 168, 170].

The current integrated checks concern real packing, physical metadata, and same-forward internal reversible execution. Future matched GPU profiles compare ordinary autograd, checkpointing, and reversible storage. They do not inherit the old second-order temporal-prototype results. Configuration, data, code, checkpoint, and metric hashes define which empirical statement each artifact supports.

## Наукова новизна

The system contribution is a task-linked separation of retained evidence, internal training activations, spatial detail allocation, and the remaining parameter/workspace costs. This separates methods that are often grouped under the single label “memory efficient,” and makes each comparative claim testable at its own interface.

The published methodological contributions remain the implemented reversible temporal state/backward procedure in TempVerseFormer [A2], and the fixed local–global–local byte hierarchy, RoPE-ND, and predictive-complexity diagnostics in V-BLT [A1]. Article [A3] supplies the appearance/style, identity, geometry, and concept-consistency synthesis; the dissertation extends its reasoning to temporal observability without attributing video experiments to that review.

The retained-context contribution is an explicit package and observability contract together with controlled allocation experiments. The new campaign varies budget, distribution, coarse expansion, registered level constructions, ownership representations, and space–time allocation. CEX/LVL/OVR are compound interventions and are described accordingly. The complete matrix is development evidence; only evaluated selected roles can receive sealed confirmation.

The integrated architectural contribution is the specified composition of real packing, physical metadata, route-aware exposure, and a same-forward reversible core. The superseded fixed-region/second-order prototype is historical feasibility evidence, not the final contribution. New memory and competitive-quality claims remain conditional on their artifact cells.

> [RESULT PENDING: CLAIM_A4_SUMMARY; evidence=registered development estimate; source=artifacts/v2/article4_primary_30k/post_training/all_final_development_metrics.json; registry=refactor_v2/registries/placeholders.json. No empirical value inserted.]

> [RESULT PENDING: CLAIM_INT_SUMMARY; evidence=conditional integrated evidence cell; source=integrated accepted evidence bundle (not yet measured/locked); registry=refactor_v2/registries/placeholders.json. No empirical value inserted.]

## Практичне значення

The work provides controlled dataset generators, published implementations for TempVerseFormer and V-BLT, an explicit retained-context allocator and config-driven training campaign, and a separate OpenSTL-native integrated research model [171–173]. Their practical roles differ: reproducible data, reversible activation reconstruction, structured spatial diagnosis, controlled allocation, and a composition under evaluation.

The campaign controllers save optimizer, random-number, model/EMA, and progress state for resumption; local JSON/JSONL/checkpoints are authoritative and W&B mirrors the record. Exact branches, source hashes, commands, dataset manifests, and verification receipts are in Appendix Ж. CPU integration checks demonstrate bounded compatibility; practical GPU memory/quality advantages remain result-gated. Existing institutional implementation records are preserved verbatim in Appendix Б and the software-use subsection of Chapter 4; unverified administrative information is not inferred from repository activity.

## Достовірність результатів

Published results, historical protected experiments, new development estimates, selected confirmation, and CPU mechanism tests retain separate provenance. Every new numerical statement must resolve to an immutable artifact, metric implementation, population, configuration, checkpoint, and uncertainty unit. Both training seeds are displayed; clip-level intervals are conditional on a trained checkpoint. A broad training-seed confidence claim is not inferred from two runs.

No incomplete development result is reported as confirmatory. E5a's non-replicated late reversal and E6's coverage/identity conflict remain visible rather than being removed for an optimistic narrative. TRB, the historical long Moving-MNIST probe, and the planned standard OpenSTL protocols answer different questions. The current 86 training runs are neither restarted nor reinterpreted as long-rollout training by this editorial change.

## Особистий внесок здобувача

У межах програми здобувач сформулював системну декомпозицію, розробив програмне забезпечення та контрольовані генератори, реалізував віднесені до дисертації компоненти моделей і оцінювання, провів експерименти, упорядкував артефакти доказів та підготував науковий аналіз. Внесок у спільні публікації відокремлено від розширень, виконаних лише для дисертації.

Здобувач також виконав синтез на рівні дисертації: уніфікував термінологію окремих досліджень, зіставив межі реалізацій з опублікованими описами та звузив твердження там, де експерименти підтверджують лише фіксовану конфігурацію або умовну робочу область. Адміністративні формулювання ролей співавторів мають бути погоджені з відповідними авторами та установою.

- Для **[A1]** здобувач розробив і реалізував V-BLT, інтеграцію RoPE-ND, процедурні експерименти та аналіз; співавтор забезпечив наукове керівництво, системне формулювання та рецензування рукопису. У дисертації адаптивне формулювання звужено до фактично наведених доказів для фіксованих блоків.
- **[A2]** є одноосібною працею. Здобувач розробив TempVerseFormer і реверсивну процедуру зворотного проходу в часі, провів експерименти з контрольованим рухом та зовнішні порівняння, оцінив пам’ять і час. Перевірений контракт реверсивного стану та пізніший аналіз стохастичного відтворення викладено окремо як розширення статті.
- Для **[A3]** здобувач виконав технічний огляд і запропонував категорію концептуальної консистентності; співавтор здійснював наукове керівництво та редакційний перегляд. Дисертація розвиває обговорення статичних зображень до операційних критеріїв відео, не приписуючи статті часових доказів.
- Для **[A4]** здобувач розробив дослідження ієрархічного контексту, інструмент точної спостережуваності, набір показників, статистичний протокол, програму перевірки робастності, рисунки та відтворюваний ланцюг артефактів. **[ДАНІ ВІД АВТОРА: перевірити остаточний список авторів і формулювання внеску перед поданням.]**

The redesigned hierarchy study and OpenSTL integration form two evidence blocks of one combined manuscript in preparation [A4], whose bibliographic record remains provisional [136]. Historical `A4_*`/`A5_*` labels remain artifact identifiers, not a commitment to two additional papers. The author/contributor declarations above are retained; the final author list, institutional approval, and publication status still require author-supplied confirmation.

## Апробація та публікації

Основні результати опубліковано у трьох журнальних статтях [A1]–[A3], а рукопис про ієрархічний контекст [A4] підготовлено до подання. **[ДАНІ ВІД АВТОРА: додати повні записи шести матеріалів конференцій, назви заходів, місця проведення або онлайн-статус, дати, сторінки та остаточне зведення публікацій; підтвердити включення DSMSI-2023.]**

## Структура дисертації

The four chapter titles and semantic order are retained. Chapter 1 establishes the system problem, consistency taxonomy, resource terms, and falsifiable questions. Chapter 2 preserves published reversible-computation and V-BLT evidence while relocating standard derivations. Chapter 3 defines the controlled retained-context method, exact experiment programme, and separately implemented OpenSTL architecture. Chapter 4 organizes allocation results, selected confirmation, historical long-rollout boundaries, component profiles, and external quality–memory tests. General conclusions align with the six tasks. Existing appendices retain their Ukrainian letter sequence, administrative records, contribution declarations, and permissions while gaining the configuration, claim, formula, and migration registries.

This is an evidence-gated mixed-language draft: retained foundation text is Ukrainian and new complete blocks are English. Final language harmonization, institutional counts, numbering, and result-dependent abstract/conclusion sentences follow evidence lock rather than being guessed in advance.
