"""Deterministic editorial migration from a frozen manuscript, without experiment writes.

Substantive replacement prose lives in apply_patch-edited English source fragments.
This script performs the mechanical assembly, relocation, inventories and hash ledgers.
Only the dedicated dissertation branch is writable. No torch/CUDA/data loading.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "cf3f661"
MAN = ROOT / "docs/dissertation/condensed_translated_humanized_ukrainian"
EDIT = ROOT / "docs/dissertation/refactor_v2"
FIG = ROOT / "figures/dissertation_v2"
FILES = ["00_FRONT_MATTER.md", "01_INTRODUCTION.md", "02_CHAPTER_1.md", "03_CHAPTER_2.md",
         "04_CHAPTER_3.md", "05_CHAPTER_4.md", "06_CONCLUSIONS.md", "07_REFERENCES.md", "08_APPENDICES.md"]


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def jwrite(path: Path, value) -> None:
    write(path, json.dumps(value, ensure_ascii=False, indent=2))


def section(text: str, title: str) -> str:
    match = re.search(r"^(#{2,3}) " + re.escape(title) + r"[^\n]*\n", text, re.M)
    if not match:
        raise ValueError(f"missing section {title}")
    end = re.search(r"^#{1," + str(len(match[1])) + r"} ", text[match.end():], re.M)
    return text[match.start():match.end() + end.start() if end else len(text)]


def replace_section(text: str, title: str, replacement: str) -> str:
    old = section(text, title)
    return text.replace(old, replacement.strip() + "\n\n", 1)


def figures(text: str):
    pattern = r"!\[([^\]]*)\]\(([^\n]+?)\)\s*\n\s*\*\*Рисунок ([\dА-ЯІЇЄ]+\.\d+)\s*[–—-]([^\n]*)"
    return {m[3]: {"block": m[0], "alt": m[1], "path": m[2], "caption": m[4], "offset": m.start()}
            for m in re.finditer(pattern, text)}


def tables(text: str):
    found = {}
    for m in re.finditer(r"^\*\*Таблиця ([\dА-ЯІЇЄ]+\.\d+)[^\n]*", text, re.M):
        tail = text[m.end():]
        first = re.search(r"^\|.+\|\s*$", tail, re.M)
        if not first:
            continue
        # The first rectangular result/config table; second E6 panel preserved separately.
        end = first.start()
        for line in tail[first.start():].splitlines(keepends=True):
            if not line.strip().startswith("|"):
                break
            end += len(line)
        found[m[1]] = {"block": text[m.start():m.end() + end], "offset": m.start()}
    return found


def resolve_images(text: str) -> str:
    def fix(m):
        path = m[2]
        if path.startswith(("https:", "http:", "data:")):
            return m[0]
        target = (MAN / path).resolve()
        if not target.exists():
            matches = list(ROOT.glob("**/" + Path(path).name))
            matches = [p for p in matches if ".git" not in p.parts]
            if not matches:
                raise FileNotFoundError(path)
            same = [p for p in matches if p.suffix == target.suffix]
            target = (same or matches)[0]
        import os
        rel = os.path.relpath(target, MAN).replace("\\", "/")
        return f"![{m[1]}]({rel})"
    return re.sub(r"!\[([^\]]*)\]\(([^\n]+?)\)", fix, text)


def main():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "codex/dissertation-refactor-v2":
        raise SystemExit("Refusing editorial writes outside dedicated dissertation branch")
    original = {}
    raw_hashes = {}
    for name in FILES:
        raw = subprocess.check_output(["git", "show", f"{BASE}:{(MAN / name).relative_to(ROOT).as_posix()}"], cwd=ROOT)
        raw_hashes[name] = digest(raw)
        original[name] = raw.decode("utf-8-sig").replace("\r\n", "\n")
    current = dict(original)
    blocks = dict(re.findall(r"<!-- BLOCK (\w+) -->\n(.*?)\n<!-- END BLOCK -->", (EDIT / "english/SURGICAL_BLOCKS.md").read_text(encoding="utf-8"), re.S))
    all_figs = {key: {**value, "file": name} for name, text in original.items() for key, value in figures(text).items()}
    all_tables = {key: {**value, "file": name} for name, text in original.items() for key, value in tables(text).items()}
    archives = []
    migrations = []

    def change(name, title, block):
        old = section(current[name], title)
        archives.append((f"Source section {title} ({name})", old))
        migrations.append({"old_location": f"{name}:{title}", "new_location": f"{name}:{block}", "reason": "English refactor; full prior block archived", "source": BASE, "permission_administrative_constraint": "preserve existing attribution", "verification_status": "source block retained"})
        current[name] = replace_section(current[name], title, blocks[block])

    # Both abstracts have one claim-bearing source to prevent language drift.
    name = FILES[0]
    old_uk = section(current[name], "Українська анотація")
    institutional = old_uk.split("\n\n")[1]
    abstract = blocks["abstract"]
    archives.extend([("Source Ukrainian abstract", old_uk), ("Source English abstract", section(current[name], "Англійський реферат"))])
    current[name] = replace_section(current[name], "Українська анотація", "## Українська анотація\n\n" + institutional + "\n\nEditorial language note: the replacement scientific abstract is drafted in English pending final Ukrainian harmonization.\n\n" + abstract)
    current[name] = replace_section(current[name], "Англійський реферат", "## Англійський реферат\n\n" + abstract)
    # Preserve actual publication entries and institutional fields; add accurate status.
    current[name] += "\n\n" + blocks["publication_update"] + "\n\nFinal counts are generated in the refactor inventory; source-era counts are not submission counts.\n"
    glossary=section(original[FILES[0]],'Символи та скорочення')
    current[FILES[0]]=replace_section(current[FILES[0]],'Символи та скорочення','## Символи та скорочення\n\nThe preserved symbol and abbreviation registry is in Appendix В.v2. It is a method glossary, not a result table.')
    # replace_section reaches EOF for the original final heading; retain the appended status note.
    current[FILES[0]]+='\n\n'+blocks['publication_update']+'\n\nFinal counts are generated in the refactor inventory; source-era counts are not submission counts.\n'
    for title, block in [("Мета та дослідницькі завдання", "tasks"), ("Наукова новизна", "novelty"),
                         ("Практичне значення", "practical"), ("Надійність та валідність результатів", "validity"),
                         ("Структура та обсяг дисертації", "intro_structure")]:
        change(FILES[1], title, block)
    current[FILES[1]] = current[FILES[1]].replace("## Зв’язок", blocks["intro_relevance"] + "\n\n## Зв’язок", 1)
    # Locate old publication-boundary paragraph without changing author declaration bullets.
    current[FILES[1]] = re.sub(r"^Інтеграційне розширення виконано.*$", blocks["publication_update"], current[FILES[1]], flags=re.M)
    if blocks["intro_relevance"] not in current[FILES[1]]:
        p = current[FILES[1]].find("\n## ", 10)
        current[FILES[1]] = current[FILES[1]][:p] + "\n\n" + blocks["intro_relevance"] + current[FILES[1]][p:]
    # Chapters 1–2 retain the surrounding Ukrainian scientific prose and published results.
    for name, title, block in [(FILES[2], "1.1.1", "state_notation"), (FILES[2], "1.2.5", "static_bridge"), (FILES[2], "1.3.4", "system_map"),
                             (FILES[2], "1.5.2", "hypotheses"), (FILES[3], "2.1.1", "procedural"),
                             (FILES[3], "2.2.2", "reversible_block"), (FILES[3], "2.2.4", "replay"),
                             (FILES[3], "2.2.8", "stochastic"), (FILES[3], "2.3.4", "rope"),
                             (FILES[3], "2.3.7", "measured_router"), (FILES[3], "2.4", "synthesis")]:
        change(name, title, block)
    current[FILES[2]] = current[FILES[2]].replace("### 1.3.5", blocks["hierarchy"] + "\n\n### 1.3.5", 1)
    current[FILES[2]] = current[FILES[2]].replace("## 1.5", blocks["literature"] + "\n\n## 1.5", 1)
    # Remove source-era prototype assertions rather than merely appending a disclaimer.
    current[FILES[1]] = re.sub(r"^Центральною архітектурною задачею дисертації.*$", "The retained-context study holds the predictive mechanism fixed while changing exact packages of observed evidence. The later OpenSTL-native composition is a separately implemented end-to-end model with physical metadata and an internally reversible core. Their results remain distinct evidence blocks of one manuscript in preparation; neither inherits the other's claimed performance.", current[FILES[1]], flags=re.M)
    current[FILES[1]] = replace_section(current[FILES[1]], "Методи дослідження", "## Методи дослідження\n\n" + blocks["intro_method_update"])
    current[FILES[2]] = re.sub(r"^На рисунку 1.4 показано.*$", "The four-term accounting identifies the resource directly changed by each intervention. Retained evidence, internal activations, parameters/optimizer, and workspace must be measured separately. The current integrated architecture is a composition under evaluation; no total-memory advantage is inherited from earlier prototypes or multiplied component ratios.", current[FILES[2]], flags=re.M)
    current[FILES[2]] = re.sub(r"^У формулі \(1\.7\).*$", lambda _:blocks['objective_contract'], current[FILES[2]], flags=re.M)
    for name, source in [(FILES[4], "CHAPTER_3.md"), (FILES[5], "CHAPTER_4.md")]:
        # Full replaced methods/results are archival, not silently lost or final-method prose.
        for m in re.finditer(r"^## (3\.\d+|4\.\d+)[^\n]*", original[name], re.M):
            archives.append((f"Source section {m[1]} ({name})", section(original[name], m[1])))
        current[name] = (EDIT / "english" / source).read_text(encoding="utf-8")

    def expand(text):
        def fig(m):
            return all_figs[m[1]]["block"]
        def tab(m):
            return all_tables[m[1]]["block"]
        def sec(m):
            text = section(original[FILES[5]], m[1])
            return text[text.index("\n") + 1:].strip()
        text = re.sub(r"\{\{LEGACY_FIGURE:([^}]+)\}\}", fig, text)
        text = re.sub(r"\{\{LEGACY_TABLE:([^}]+)\}\}", tab, text)
        return re.sub(r"\{\{LEGACY_SECTION:([^}]+)\}\}", sec, text)
    current = {name: expand(text) for name, text in current.items()}
    # Replace the old resource picture with the four-term map; avoid duplicating it within Chapter 1.
    old_resource = all_figs["1.4"]["block"]
    resource_new = "![Four separate memory resources](../../../figures/dissertation_v2/resource_interventions.svg)\n\nFigure D1-resources — Retained state, internal activations, parameters/optimizer, and workspace. Conceptual accounting only; ratios are not multiplied."
    current[FILES[2]] = current[FILES[2]].replace(old_resource, resource_new)
    current[FILES[2]] = current[FILES[2]].replace("![Which memory resource is directly changed by each intervention?](../../../figures/dissertation_v2/resource_interventions.svg)", "See Figure D1-resources in Section 1.1.3 for the four-term accounting.")
    current[FILES[2]] = re.sub(r"^Figure D1-resources — Retained evidence.*\n", "", current[FILES[2]], flags=re.M)
    current[FILES[4]] = current[FILES[4]].replace("Світ LoopMom", "TRB world").replace("LoopMom", "TRB")
    current[FILES[4]] = current[FILES[4]].replace("`TRB` is retained only as a historical dataset/code identifier.", "`LoopMom` is retained only as a historical dataset/code identifier.")

    # Remove redundant main figures; preserve the source asset and original caption in appendix.
    remove_figs = {"1.2", "1.4", "1.6", "1.7", "1.8", "1.11", "1.12", "1.13", "1.14", "1.15", "1.16", "1.17", "1.18", "1.19", "1.20", "1.21", "1.25", "2.2", "2.5", "2.7", "2.8", "2.12", "2.14", "2.18", "2.19"}
    remove_tables = {"1.1", "1.2", "2.1", "2.2", "2.6", "2.7", "2.8", "3.1", "3.2", "3.3"}
    for name in FILES[1:6]:
        for key, value in figures(current[name]).items():
            if key in remove_figs:
                current[name] = current[name].replace(value["block"], f"See Appendix Е, source Figure {key}, for the supporting illustration.")
        for key, value in tables(current[name]).items():
            if key in remove_tables:
                current[name] = current[name].replace(value["block"], f"The method/configuration record is preserved in Appendix В (source Table {key}).")
    # Split published dataset tables without changing a single numerical value.
    published_table_rows = []
    for number in ["2.3", "2.4", "2.10"]:
        old = all_tables[number]["block"]
        archives.append((f"Source published Table {number}", old))
        lines = [re.sub(r"\*\*", "", line) for line in old.splitlines() if line.startswith("|")]
        header = [v.strip() for v in lines[0].strip("|").split("|")]
        records = []
        last_population = ""
        for line in lines[2:]:
            values = [v.strip() for v in line.strip("|").split("|")]
            if values[0]: last_population = values[0]
            else: values[0] = last_population
            records.append(values)
            published_table_rows.append({"source_table":number,"population":last_population,"header":header,"values":values,"source_sha256":digest(old)})
        if number == "2.3":
            replacement = old.split("\n")[0] + "\n\n" + "\n".join([lines[0],lines[1],*["| " + " | ".join(v) + " |" for v in records]])
        else:
            groups = list(dict.fromkeys(v[0] for v in records))
            parts = []
            for index, population in enumerate(groups):
                subheader = header[1:4] if number == "2.4" else header[1:]
                caption = f"Table {number}{chr(97+index)} — Published point estimates: {population}. Original metric scale; no retrospectively invented seed uncertainty. Source Table {number}."
                body = ["| " + " | ".join(subheader) + " |", "|" + "---|"*len(subheader)]
                body += ["| " + " | ".join(v[1:4] if number == "2.4" else v[1:]) + " |" for v in records if v[0] == population]
                parts.append(caption + "\n\n" + "\n".join(body))
            replacement = "\n\n".join(parts)
        current[FILES[3]] = current[FILES[3]].replace(old,replacement)
    jwrite(EDIT / "registries/published_table_transcription.json",published_table_rows)

    # Scientific-dependency inventory precedes formula demotion.
    guide_path = ROOT / "docs/dissertation/DISSERTATION_REVIEWER_PROVENANCE_AND_REPRODUCTION_GUIDE_2026-08-29.md"
    guide = guide_path.read_text(encoding="utf-8")
    guide_formula = guide[guide.index("### 7.1"):guide.index("## 8.")]
    formula_roles = {}
    for line in guide_formula.splitlines():
        cells = [v.strip() for v in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and re.fullmatch(r"[1-4]\.\d+", cells[0]):
            formula_roles[cells[0]] = cells[1:]
    formulas = []
    for name, text in original.items():
        for m in re.finditer(r"\$\$(.*?)\$\$", text, re.S):
            tag = re.search(r"\\tag\{([^}]+)\}", m[1])
            if not tag:
                continue
            number = tag[1]
            start = text.rfind("\n\n", 0, max(0, m.start() - 2))
            after = text.find("\n\n", m.end() + 2)
            context = text[max(0, start):after if after >= 0 else m.end()]
            refs = [f"{n}:{line_no}" for n, t in original.items() for line_no, line in enumerate(t.splitlines(), 1)
                    if f"({number})" in line]
            role = formula_roles[number]
            formulas.append({"current_number": number, "scientific_role": context.strip(), "role_summary_source": role[0], "relation_type": role[1], "implementation_location": role[2],
                             "provenance_guide": str(guide_path.relative_to(ROOT)), "provenance_guide_sha256": digest(guide),
                             "downstream_prose_references": refs,
                             "dependent_figures_tables": [f"{kind} {k}" for kind, items in [("figure",all_figs),("table",all_tables)] for k, v in items.items() if v["file"] == name and abs(v["offset"] - m.start()) < 1500],
                             "dependency_method": "source callout lines plus nearby original evidence objects; not a claim of causal dependence",
                             "result_estimand": role[0] if any(word in role[0].lower() for word in ["mse", "effect", "ефект", "entropy", "похиб", "варт", "пам’ят", "loss", "метрик"]) else "not a standalone result estimand: " + role[0],
                             "source_file": name, "source_sha256": digest(m[0]), "block": m[0]})
    # Retain only necessary main equations; standard/supporting relations remain accessible.
    demote = {f"1.{i}" for i in [6, 8, 9, 10, 11]} | {f"2.{i}" for i in [5, 6, 11, 14, 16, 17, 29, 30, 32, 33, 36, 37, 38, 39, 41, 42]}
    for name in FILES[2:4]:
        def demotion(m):
            tag = re.search(r"\\tag\{([^}]+)\}", m[0])
            if tag and tag[1] in demote:
                # Keep the compact relation inline where its surrounding Ukrainian definition
                # remains; move its numbering/derivation, not its needed antecedent.
                relation = re.sub(r"\\tag\{[^}]+\}", "", m[0][2:-2])
                relation = re.sub(r"\s+", " ", relation).strip()
                return "$" + relation + "$"
            return m[0]
        current[name] = re.sub(r"\$\$.*?\$\$", demotion, current[name], flags=re.S)

    # Main conclusions remain task-aligned and evidence gated rather than preserving obsolete totals.
    archives.append(("Source general conclusions", original[FILES[6]]))
    current[FILES[6]] = """# Загальні висновки

This editorial conclusion preserves the six-task order. Result-dependent comparative sentences are deferred to the locked claim ledger; the historical source conclusion is retained in Appendix Е.

1. **System and consistency analysis.** Retained state, internal activations, parameters/optimizer, and workspace require separate accounting. Appearance/style, identity, geometry, concept, trajectory, and cross-level consistency remain distinct task endpoints, retaining the conceptual contribution of [A3].
2. **Reversible temporal computation.** The published TempVerseFormer results and implemented state boundary in Chapter 2 support a specific activation-storage/recomputation trade-off, not universal quality superiority or enlargement of retained context [A2].
3. **Spatial representation and diagnosis.** V-BLT's fixed byte hierarchy, RoPE-ND, and entropy diagnostics remain demonstrated configuration-level contributions. Dynamic token routing is not retrospectively attributed to these experiments [A1].
4. **Hierarchical retained context.** Chapter 3 specifies exact packages, physical identities, causal targets, and the 43-configuration/two-seed 30k programme. Its new endpoint tests long observed context over four predicted frames, not long autoregressive stability. The final allocation result remains pending the corresponding artifact.
5. **Observability and evaluation.** Historical E2 uses a package-specific reach-difference population. The new confirmation uses a common 96/28-reference sprite-visibility frame mask and separately normalized four-frame errors. Clip-aware statistics preserve dependence, and training seeds remain separate observations. The complete development matrix and selected sealed confirmation are different evidence statuses.
6. **Operating region and composition.** Historical E5a does not reproduce a universal late reversal; E6 shows that coverage/trajectory can improve while identity worsens. The integrated OpenSTL architecture tests distinct hierarchy, reversibility, and routing hypotheses. Its correctness does not settle competitive quality or total-memory benefit; absent or negative component/external results narrow the conclusion.

{{CLAIM_A4_SUMMARY}}

{{CLAIM_INT_SUMMARY}}

The final design rule is conditional on the bottleneck and task: test access when evidence is missing, detail when identity is lost, activation reconstruction when training storage dominates, and dynamic allocation when unequal information density justifies its overhead. Do not multiply ratios from unrelated memory terms. The new hierarchy and integration blocks form one manuscript in preparation [A4]; institutional publication status, final counts, and language harmonization remain release gates.
"""
    # New primary sources; old bibliography retained because archived claims retain citations.
    current[FILES[7]] += """

175. PyTorch contributors. torch.utils.checkpoint: Activation checkpointing. Official documentation. https://docs.pytorch.org/docs/stable/checkpoint.html (accessed 3 September 2026).
176. Rao Y., Zhao W., Liu B., Lu J., Zhou J., Hsieh C.-J. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. NeurIPS, 2021. https://arxiv.org/abs/2106.02034.
177. Rougier N. P., Droettboom M., Bourne P. E. Ten Simple Rules for Better Figures. PLOS Computational Biology 10(9):e1003833, 2014. https://doi.org/10.1371/journal.pcbi.1003833.
178. OpenSTL contributors. OpenSTL official implementation and benchmark protocols. https://github.com/chengtan9907/OpenSTL (accessed 3 September 2026; local implementation revision recorded in Appendix Ж).
"""
    # Preserve every existing appendix line and add to its existing lettered sections.
    appendices = original[FILES[8]]
    def add(letter, value):
        nonlocal appendices
        marker = re.search(r"^## Додаток " + letter + r"[^\n]*\n", appendices, re.M)
        if not marker:
            raise ValueError(f"missing appendix {letter}")
        following = re.search(r"^## Додаток ", appendices[marker.end():], re.M)
        pos = marker.end() + following.start() if following else len(appendices)
        appendices = appendices[:pos].rstrip() + "\n\n" + value.strip() + "\n\n" + appendices[pos:]

    add("А", "### А.v2 Combined manuscript and task map\n\n" + blocks["publication_update"] + "\n\nThe generated content-transfer ledger preserves source paragraph, figure, table and claim identities for [A1] V-BLT, [A2] TempVerseFormer, and [A3] static-consistency synthesis. Task alignment is recorded in `refactor_v2/TASK_CLAIM_MAP.md`.")
    add("В", "### В.v2 Exact new configuration registry\n\nThe full machine-readable registry is `refactor_v2/registries/experiment_matrix.json`; its readable counterpart is `refactor_v2/EXPERIMENT_MATRIX.md`. It contains the 80 concrete run specifications and six explicitly unresolved dependency cells. Resolved identifiers/hashes are imported only after the campaign writes its dependency ledger. No unresolved hash is invented. Method tables relocated below retain their original source identifiers.\n\n" + "\n\n".join(v["block"] for k, v in all_tables.items() if k in remove_tables))
    add('В','#### Preserved front-matter symbol registry\n\n'+glossary.split('\n',1)[1])
    add("Г", "### Г.v2 Claim, content, and migration ledgers\n\nSentence-level pending claims and evidence objects are registered in `refactor_v2/registries/placeholders.json`. Their populations, seed units, allowed conclusions, absence rules, and expected artifacts are explicit. Frozen source hashes, paragraph-level content transfer, figure/table disposition, formula dependencies, and appendix migration are in the sibling registries. The complete development frontier is not promoted to confirmation. Existing contribution and provenance records above remain authoritative for their original source-era claims.")
    eq_archive = []
    for f in formulas:
        tag = f["current_number"]
        retained = any(f"\\tag{{{tag}}}" in current[n] for n in FILES[:7])
        f["destination"] = f"main source tag {tag}" if retained else f"Appendix Д eq-source-{tag.replace('.', '-')}" + ("; compact relation inline in original main subsection" if tag in demote and f['source_file'] in FILES[2:4] else "")
        f["replacement_callout"] = f"source equation {tag}" if not retained else tag
        eq_archive.append(f"<a id=\"eq-source-{tag.replace('.', '-')}\"></a>\n\n#### Source equation {tag}\n\nOriginal role/context and variable definitions:\n\n{f['scientific_role']}\n\nSource: `{f['source_file']}` at `{BASE}`. New location: {f['destination']}. This source-era relation does not override the revised method contract.")
    add("Д", "### Д.v2 Formula scientific-dependency archive\n\nOriginal identifiers below are source IDs, not final dissertation numbering. Each original equation is retained with its immediate scientific context; dependencies and implementation status are in `formula_dependencies.json`. Replaced formula systems in the main text do not silently redefine an old identifier.\n\n" + "\n\n".join(eq_archive))
    archived = []
    for i, (title, text) in enumerate(archives, 1):
        # Demote source headings so they remain inside Appendix Е.
        text = re.sub(r"^#{1,3} ", "##### ", text, flags=re.M)
        archived.append(f"#### Е.v2.{i} {title}\n\nHistorical/source-era material; superseded wording is preserved for provenance, not endorsed as the current model or current result status.\n\n{text}")
    add("Е", "### Е.v2 Migrated source material\n\nThe following source sections retain standard derivations, publication tutorial assets, historical E2–E8 detail, old external feasibility, and the superseded integration prototype. Read Chapter 3/4 for the current contract.\n\n" + "\n\n".join(archived) + "\n\n#### Е.v2 Supporting figure archive\n\n" + "\n\n".join(v["block"] for k, v in all_figs.items() if k in remove_figs))
    add("Ж", (EDIT / "english/REPRODUCTION_APPENDIX.md").read_text(encoding="utf-8"))
    add("И", "### И.v2 New asset provenance\n\nNew conceptual SVGs are author-created code-native schematics, not empirical observations. Calibration uses the archived hash-bound analysis/logs; simplified historical plots use an explicitly labelled mechanical transcription of preserved source tables. Every generator writes plotted JSON rows. Pending result figures have no synthetic values. Original asset licence/permission records above are preserved without inferring new permission. New generator/source/output hashes are recorded in `refactor_v2/registries/generated_assets.json`. Figure design follows a single-question contract [177].")
    # Archive all source table/figure IDs so references remain navigable even after relocation.
    current[FILES[8]] = appendices
    # Fix a descriptive-name typo defensively in the assembly, also fixed in source below.
    current[FILES[5]] = current[FILES[5]].replace("dissertationation_v2", "dissertation_v2")
    # Main-text callouts to moved original assets remain explicitly source-era.
    for name in FILES[:7]:
        for num in remove_figs:
            current[name] = re.sub(r"((?:[Рр]ис\w*|рис\.)\s+)" + re.escape(num) + r"(?!\d)", lambda m: m[0] + " (source; Appendix Е)", current[name])
        for num in remove_tables:
            current[name] = re.sub(r"((?:таблиц\w*|Таблиц\w*)\s+)" + re.escape(num) + r"(?!\d)", lambda m: m[0] + " (source; Appendix В)", current[name])
        # Equation references whose numbered object moved are explicit links to the source archive.
        for f in formulas:
            tag = f["current_number"]
            if f["destination"].startswith("Appendix"):
                current[name] = current[name].replace(f"({tag})", f"([source {tag}](08_APPENDICES.md#eq-source-{tag.replace('.', '-')}))")
    # Draft boundary appears once at the beginning, rather than silently presenting placeholders as final.
    current[FILES[0]] = current[FILES[0]].replace("# Вступні матеріали", "# Вступні матеріали\n\n> Editorial draft, 3 September 2026. English changes applied to the preserved Ukrainian foundation. Source figure/formula/table identifiers remain stable until result lock. Pending result and institutional fields prevent submission release.", 1)
    # Actual placeholder expansion is handled by the registry builder, before audit.
    for name, text in current.items():
        write(MAN / name, '\n'.join(line.rstrip() for line in resolve_images(text).splitlines()))
    jwrite(EDIT / "registries/source_manifest.json", {"baseline_commit": BASE, "source_sha256": raw_hashes, "branch": branch})
    jwrite(EDIT / "registries/formula_dependencies.json", formulas)
    # Paragraph-level transfer includes precise hashes and destination verification.
    transfer = []
    for name in FILES[2:4]:
        for index, para in enumerate(original[name].split("\n\n")):
            if not para.strip():
                continue
            explicit = re.findall(r"\[(A[123]|13[345])\]",para)
            article = {"133":"A3","134":"A1","135":"A2"}.get(explicit[0],explicit[0]) if explicit else "not certified by paragraph inventory; consult publication transfer map"
            destination = "main verbatim" if para in current[name] else "appendix verbatim" if para in current[FILES[8]] else "English restatement/source dependency record; baseline preserved"
            offset=original[name].find(para)
            headings=re.findall(r'^#{1,3} .+',original[name][:offset],re.M)
            source_heading=headings[-1] if headings else 'chapter opening'
            if para in current[name]: newloc=f"{name}:{current[name][:current[name].index(para)].count(chr(10))+1}"
            elif para in current[FILES[8]]: newloc=f"08_APPENDICES.md:{current[FILES[8]][:current[FILES[8]].index(para)].count(chr(10))+1}"
            else: newloc=f"{name} / retained semantic section {source_heading}; source equation/asset locator in dependency/disposition ledger"
            transfer.append({"source_file": name, "source_heading":source_heading,"paragraph_index": index, "sha256": digest(para), "publication_citation_in_source": article,"publication_origin_verification":"source citation only, not independently recertified article claim; detailed source anchors in PUBLICATION_TRANSFER_MAP.md", "type": "figure" if "![" in para else "table" if para.startswith("|") else "formula" if "\\tag" in para else "paragraph/claim", "source_manuscript_preservation": destination,"new_location":newloc, "source_excerpt": para[:300]})
    jwrite(EDIT / "registries/content_transfer.json", transfer)
    asset_rows = []
    for kind, items in [("figure", all_figs), ("table", all_tables)]:
        for key, value in items.items():
            main = any(value["block"] in current[n] for n in FILES[:7])
            asset_path=None
            if kind=='figure':
                resolved=resolve_images(value['block'])
                asset_path=(MAN/re.search(r'!\[[^\]]*\]\(([^)]+)\)',resolved)[1]).resolve()
            asset_rows.append({"kind": kind, "source_id": key, "source_file": value["file"], "source": value.get("path", "source manuscript table and its cited publication/artifact"), "destination": "main + source provenance" if main else "Appendix Е/В + preserved baseline", "reason": "single-purpose empirical/publication evidence retained" if main else "redundant tutorial, method registry, replaced architecture, or supporting historical evidence", "sha256": digest(value["block"]),"asset_file_path":str(asset_path) if asset_path else None,"asset_file_sha256":digest(asset_path.read_bytes()) if asset_path else None})
    jwrite(EDIT / "registries/asset_disposition.json", asset_rows)
    jwrite(EDIT/'registries/source_tables.json',[{'source_id':k,'source_file':v['file'],'source_commit':BASE,'sha256':digest(v['block']),'raw_markdown':v['block'],'rows':[[cell.strip() for cell in line.strip().strip('|').split('|')] for line in v['block'].splitlines() if line.startswith('|')]} for k,v in all_tables.items()])
    tree=subprocess.check_output(['git','ls-tree','-r',BASE,'--','docs/figures','figures'],cwd=ROOT).decode('utf-8')
    write(EDIT/'registries/source_figure_git_tree.txt',tree)
    jwrite(EDIT/'registries/source_figure_tree.json',{'baseline':BASE,'git_tree_listing_sha256':digest(tree),'referenced_asset_count':sum(r['kind']=='figure' for r in asset_rows),'hash_kind':'full baseline Git tree/object inventory plus SHA-256 of every manuscript-referenced file in asset_disposition.json; no source files removed'})
    # Each pre-existing appendix heading/asset/admin paragraph gets an unchanged destination receipt.
    appendix_rows = []
    for i, para in enumerate(original[FILES[8]].split("\n\n")):
        if para.strip():
            appendix_rows.append({"old_location": f"08_APPENDICES.md paragraph {i}", "asset_claim": para[:180], "new_location": "same Ukrainian-letter appendix; additions only", "reason": "preserve institutional, scientific and permissions record", "source": BASE, "sha256": digest(para), "permission_administrative_constraint": "no reinterpretation or inferred approval", "verification_status": "verbatim" if para in appendices else "FAIL"})
    jwrite(EDIT / "registries/appendix_migration.json", appendix_rows)
    jwrite(EDIT / "registries/section_migration.json", migrations)
    print(json.dumps({"files_assembled": len(current), "source_formulas": len(formulas), "source_assets": len(asset_rows), "paragraph_transfer_rows": len(transfer), "preserved_appendix_items": len(appendix_rows)}, indent=2))


if __name__ == "__main__":
    main()
