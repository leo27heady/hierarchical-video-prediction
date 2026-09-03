"""Draft and fail-closed release QA for the evidence-gated English refactor."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from refactor_dissertation_v2 import ROOT, EDIT, MAN, FILES, digest, write, jwrite


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--release",action="store_true"); args=parser.parse_args()
    texts={n:(MAN/n).read_text(encoding="utf-8") for n in FILES}
    errors=[]; gates=[]; inventory={}
    for name,text in texts.items():
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)",text):
            if not m[1].startswith(("https:","http:","data:")) and not (MAN/m[1]).resolve().is_file(): errors.append(f"missing image {name}:{m[1]}")
        if re.search(r"\{\{[A-Z0-9_]+\}\}",text): errors.append(f"unregistered raw placeholder in {name}")
        inventory[name]={"sha256":digest(text),"words":len(text.split()),"numbered_equations":len(re.findall(r"\\tag\{",text)),"figures":len(re.findall(r"!\[",text)),"tables":len(re.findall(r"^\|[^\n]+\n\|\s*[-:]",text,re.M)),"headings":re.findall(r"^#{1,3} .+",text,re.M)}
    main="\n".join(texts[n] for n in FILES[:7])
    tags=re.findall(r"\\tag\{([^}]+)\}",main)
    if len(tags)!=len(set(tags)): errors.append("duplicate main equation tags")
    formula=json.loads((EDIT/"registries/formula_dependencies.json").read_text(encoding="utf-8"))
    if len(formula)!=114: errors.append("original 114-formula inventory incomplete")
    current_formulas=json.loads((EDIT/'registries/current_formula_dependencies.json').read_text(encoding='utf-8'))
    if {r['tag'] for r in current_formulas}!=set(tags):errors.append('current equation dependency ledger incomplete')
    app=json.loads((EDIT/"registries/appendix_migration.json").read_text(encoding="utf-8"))
    if any(r["verification_status"]!="verbatim" for r in app): errors.append("appendix record not retained verbatim")
    matrix=json.loads((EDIT/"registries/experiment_matrix.json").read_text(encoding="utf-8"))
    if len(matrix["runs"])!=86 or matrix["unique_configurations"]!=43: errors.append("matrix count mismatch")
    slots=json.loads((EDIT/"registries/placeholders.json").read_text(encoding="utf-8"))
    pending=set(re.findall(r"\[RESULT PENDING: ([A-Z0-9_]+)",main))
    known={r["key"] for r in slots}
    if pending-known: errors.append("pending key absent from registry")
    for family,count in [('ALC',14),('LVL',6),('OVR',6)]:
        row=next(r for r in slots if r['key']==f'A4_{family}_PRIMARY_FIG')
        if len(row['exact_configuration_ids'])!=count:errors.append(f'{family} logical comparison incomplete')
    # Existing local links in the handoff must work independently of the app's cwd.
    for doc in [EDIT/'README.md',ROOT/'docs/reviews/DISSERTATION_REFACTOR_IMPLEMENTATION_REPORT_2026-09-03.md']:
        if not doc.exists():continue
        for link in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not link.startswith(('http:','https:','#')) and not (doc.parent/link.split('#')[0]).resolve().exists():errors.append(f'broken handoff link {doc.name}: {link}')
    # Citation round trip across main + migrated scientific appendix, not bibliography itself.
    cited=set()
    for text in [t for n,t in texts.items() if n!=FILES[7]]:
        for group in re.findall(r"\[([0-9][0-9, –—-]*)\]",text):
            if group.strip().startswith("0"):
                continue  # Array/seed notation, not a bibliography identifier.
            for part in group.split(","):
                nums=re.findall(r"\d+",part)
                if len(nums)==2: cited.update(range(int(nums[0]),int(nums[1])+1))
                elif len(nums)==1: cited.add(int(nums[0]))
    bib={int(n) for n in re.findall(r"^(\d+)\. ",texts[FILES[7]],re.M)}
    if cited-bib: errors.append(f"unresolved numeric citations {sorted(cited-bib)}")
    orphans=sorted(bib-cited)
    jwrite(EDIT/"registries/citation_audit.json",{"cited":sorted(cited),"bibliography_ids":sorted(bib),"unresolved":sorted(cited-bib),"uncited_entries":orphans})
    # Formal headings preserved at chapter-level; key claim boundaries must remain explicit.
    for needle in ["four-frame", "registered development", "selected", "same-forward", "checkpointing", "mixed", "E5", "E6"]:
        if needle.lower() not in main.lower(): errors.append(f"missing scientific boundary {needle}")
    if pending: gates.append(f"{len(pending)} unique pending result/summary slots")
    if "[ДАНІ ВІД АВТОРА" in main or "[ВВЕСТИ АВТОРОМ" in main: gates.append("author/institutional fields unresolved")
    gates += ["final language harmonization", "final numbering/pagination after evidence lock", "Word/PDF institutional release rendering and permission approval"]
    if orphans: gates.append(f"{len(orphans)} inherited bibliography entries require removal or an actual retained supporting citation")
    result={"draft_status":"PASS" if not errors else "FAIL","release_status":"BLOCKED" if errors or gates else "PASS","errors":errors,"release_gates":gates,"pending_keys":sorted(pending),"inventory":inventory}
    jwrite(EDIT/"registries/qa_report.json",result)
    rows=["# Refactor draft QA", "", f"Draft checks: **{result['draft_status']}**. Submission release: **{result['release_status']}**.", "", "| File | Words | Main/source-tagged equations | Embedded figures | Tables |", "|---|---:|---:|---:|---:|"]
    rows += [f"| {n} | {v['words']} | {v['numbered_equations']} | {v['figures']} | {v['tables']} |" for n,v in inventory.items()]
    rows += ["", "## Errors", "", *(["- "+e for e in errors] or ["None in the automated draft checks."]),"", "## Remaining release gates", "", *["- "+g for g in gates], "", "The appendix includes explicitly source-era equation/figure identifiers and historical statements. They are not final numbering and do not override the main method or evidence status. Automated checks do not replace scientific/visual review."]
    write(EDIT/"QA_REPORT.md","\n".join(rows))
    print(json.dumps({k:result[k] for k in ["draft_status","release_status","errors","release_gates"]},indent=2))
    raise SystemExit(1 if errors or (args.release and gates) else 0)


if __name__=="__main__": main()
