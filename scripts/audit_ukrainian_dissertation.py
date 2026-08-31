#!/usr/bin/env python3
"""Mechanical integrity audit for the authoritative Ukrainian dissertation source."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "dissertation" / "condensed_translated_humanized_ukrainian"
GUIDE = ROOT / "docs" / "dissertation" / "DISSERTATION_REVIEWER_PROVENANCE_AND_REPRODUCTION_GUIDE_2026-08-29.md"
FILES = [
    SOURCE / name
    for name in (
        "00_FRONT_MATTER.md",
        "01_INTRODUCTION.md",
        "02_CHAPTER_1.md",
        "03_CHAPTER_2.md",
        "04_CHAPTER_3.md",
        "05_CHAPTER_4.md",
        "06_CONCLUSIONS.md",
        "07_REFERENCES.md",
        "08_APPENDICES.md",
    )
]
CHAPTER_FILES = {chapter: FILES[chapter + 1] for chapter in range(1, 5)}
EXPECTED_EQUATIONS = {1: 14, 2: 42, 3: 53, 4: 5}
EXPECTED_FIGURES = {1: 25, 2: 19, 3: 19, 4: 14}
EXPECTED_TABLES = {1: 2, 2: 10, 3: 3, 4: 7}
EXPECTED_APPENDIX_TABLES = [
    "А.1", "Б.1", "В.1", "В.2", "В.3", "В.4", "В.5", "В.6", "В.7", "В.8",
    "Г.1", "Д.1", "Д.2", "Д.3", "Е.1", "Ж.1", "Ж.2", "Ж.3",
]
EXPECTED_TABLE_IDS = [
    *(f"{chapter}.{index}" for chapter, count in EXPECTED_TABLES.items() for index in range(1, count + 1)),
    *EXPECTED_APPENDIX_TABLES,
]
LEDGERS = [
    ROOT / "docs/figures/restructured/figure_provenance.json",
    ROOT / "docs/figures/restructured/ch2_ch4_figure_provenance.json",
    ROOT / "docs/figures/restructured/existing_figure_provenance.json",
    ROOT / "figures/restructured/integrated_figure_provenance.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_figure_id(value: str) -> str:
    return value.replace("В.", "B.").replace("Е.", "E.")


def caption_pairs(text: str) -> list[tuple[str, list[str]]]:
    pairs: list[tuple[str, list[str]]] = []
    pending: list[str] = []
    for line in text.splitlines():
        match = re.search(r"!\[[^]]*\]\(([^)]+)\)", line)
        if match:
            pending.append(match.group(1))
        caption = re.match(r"^\*\*Рисунок\s+([1-4ВBЕE]\.\d+)\s+[–-]", line)
        if caption:
            pairs.append((normalize_figure_id(caption.group(1)), list(pending)))
            pending.clear()
    return pairs


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    texts: dict[Path, str] = {}
    for path in FILES:
        if not path.is_file():
            errors.append(f"missing manuscript source: {path.relative_to(ROOT)}")
            continue
        texts[path] = path.read_text(encoding="utf-8")

    all_tags: set[str] = set()
    for chapter, path in CHAPTER_FILES.items():
        text = texts.get(path, "")
        tags = re.findall(rf"\\tag\{{({chapter}\.\d+)\}}", text)
        expected = [f"{chapter}.{index}" for index in range(1, EXPECTED_EQUATIONS[chapter] + 1)]
        if tags != expected:
            errors.append(f"{path.name}: equation sequence differs from {expected[0]}–{expected[-1]}")
        if re.search(rf"\\tag\{{{chapter}\.\d+[A-Za-z]", text):
            errors.append(f"{path.name}: suffixed equation tag remains")
        all_tags.update(tags)
        figures = re.findall(rf"(?m)^\*\*Рисунок\s+({chapter}\.\d+)\s+[–-]", text)
        expected_figures = [f"{chapter}.{index}" for index in range(1, EXPECTED_FIGURES[chapter] + 1)]
        if figures != expected_figures:
            errors.append(f"{path.name}: figure sequence/count differs from {expected_figures[0]}–{expected_figures[-1]}")
        if text.count("$$") % 2:
            errors.append(f"{path.name}: unbalanced display-math delimiters")
        tables = re.findall(rf"(?m)^\*\*Таблиця\s+({chapter}\.\d+)\s+[–-]", text)
        expected_tables = [f"{chapter}.{index}" for index in range(1, EXPECTED_TABLES[chapter] + 1)]
        if tables != expected_tables:
            errors.append(f"{path.name}: table sequence/count differs from {expected_tables[0]}–{expected_tables[-1]}")
        if r"\text{" in text:
            errors.append(f"{path.name}: plain text command remains inside display mathematics")

    scientific_text = "\n".join(
        texts.get(path, "") for path in FILES[1:7] + [FILES[8]]
    )
    for reference in re.findall(r"\(([1-4]\.\d+)\)", scientific_text):
        if reference not in all_tags:
            errors.append(f"equation reference ({reference}) has no matching tag")

    appendix_text = texts.get(FILES[8], "")
    appendix_tables = re.findall(r"(?m)^\*\*Таблиця\s+([А-ЯІЇЄ]\.\d+)\s+[–-]", appendix_text)
    if appendix_tables != EXPECTED_APPENDIX_TABLES:
        errors.append("08_APPENDICES.md: appendix table sequence/count differs from registry")
    appendix_figures = re.findall(r"(?m)^\*\*Рисунок\s+([ВBЕE]\.\d+)\s+[–-]", appendix_text)
    if [normalize_figure_id(item) for item in appendix_figures] != ["B.1", "B.2", "E.1", "E.2"]:
        errors.append("08_APPENDICES.md: appendix figure sequence/count differs from В.1, В.2, Е.1, Е.2")

    if not GUIDE.is_file():
        errors.append(f"missing reviewer guide: {GUIDE.relative_to(ROOT)}")
    else:
        guide = GUIDE.read_text(encoding="utf-8")
        formula_section = guide.split("## 7. Повний реєстр формул", 1)[-1].split("## 8. Повний реєстр таблиць", 1)[0]
        guide_equations = re.findall(r"(?m)^\| ([1-4]\.\d+) \|", formula_section)
        expected_equations = [
            f"{chapter}.{index}"
            for chapter, count in EXPECTED_EQUATIONS.items()
            for index in range(1, count + 1)
        ]
        if guide_equations != expected_equations:
            errors.append("reviewer guide: formula registry is not synchronized with manuscript")
        table_section = guide.split("## 8. Повний реєстр таблиць", 1)[-1].split("## 9. Повний реєстр рисунків", 1)[0]
        guide_tables = re.findall(r"(?m)^\| ([1-4А-ЯІЇЄ]\.\d+) \|", table_section)
        if guide_tables != EXPECTED_TABLE_IDS:
            errors.append("reviewer guide: table registry is not synchronized with manuscript")
        figure_section = guide.split("## 9. Повний реєстр рисунків", 1)[-1].split("## 10. Артефакти", 1)[0]
        guide_figures = re.findall(r"(?m)^\| ([1-4]\.\d+) \|", figure_section)
        expected_figures = [
            f"{chapter}.{index}"
            for chapter, count in EXPECTED_FIGURES.items()
            for index in range(1, count + 1)
        ]
        if guide_figures != expected_figures:
            errors.append("reviewer guide: main-figure registry is not synchronized with manuscript")

    provenance: dict[str, set[str]] = {}
    for ledger in LEDGERS:
        if not ledger.is_file():
            errors.append(f"missing provenance ledger: {ledger.relative_to(ROOT)}")
            continue
        for row in json.loads(ledger.read_text(encoding="utf-8")):
            file_name = row.get("file")
            figure_id = normalize_figure_id(str(row.get("figure_id", "")))
            if not file_name:
                continue
            provenance.setdefault(Path(file_name).as_posix(), set()).add(figure_id)
            asset = ROOT / file_name
            if not asset.is_file():
                errors.append(f"provenance asset missing: {file_name}")
                continue
            if row.get("bytes") is not None and asset.stat().st_size != row["bytes"]:
                errors.append(f"provenance byte-size mismatch: {file_name}")
            if row.get("sha256") and digest(asset) != row["sha256"]:
                errors.append(f"provenance hash mismatch: {file_name}")
            generator = row.get("generator")
            if generator and row.get("generator_sha256"):
                generator_path = ROOT / generator
                if not generator_path.is_file() or digest(generator_path) != row["generator_sha256"]:
                    errors.append(f"generator hash mismatch: {generator}")

    image_count = 0
    caption_count = 0
    for path, text in texts.items():
        for caption_id, links in caption_pairs(text):
            caption_count += 1
            if not links:
                errors.append(f"{path.name}: figure {caption_id} has no preceding image")
            for link in links:
                image_count += 1
                asset = (path.parent / link).resolve()
                if not asset.is_file():
                    errors.append(f"{path.name}: broken image link {link}")
                    continue
                try:
                    relative = asset.relative_to(ROOT).as_posix()
                except ValueError:
                    errors.append(f"{path.name}: image outside repository {link}")
                    continue
                ledger_ids = provenance.get(relative, set())
                if not any(item == caption_id or item.startswith(caption_id) for item in ledger_ids):
                    errors.append(
                        f"{path.name}: caption {caption_id} has no matching provenance id for {relative}"
                    )

    references = {
        int(number)
        for number in re.findall(r"(?m)^(\d+)\.\s", texts.get(FILES[7], ""))
    }
    citations: set[int] = set()
    for body in re.findall(r"\[([0-9][0-9,\s–-]*)\]", scientific_text):
        if any(int(number) == 0 for number in re.findall(r"\d+", body)):
            continue
        for item in re.split(r"\s*,\s*", body):
            bounds = re.split(r"\s*[–-]\s*", item)
            if len(bounds) == 2:
                citations.update(range(int(bounds[0]), int(bounds[1]) + 1))
            elif bounds[0]:
                citations.add(int(bounds[0]))
    if citations - references:
        errors.append(f"undefined bibliography references: {sorted(citations - references)}")
    if references - citations:
        errors.append(f"uncited bibliography references: {sorted(references - citations)}")

    marker_patterns = (r"\[\[", r"\[X\]", r"(?i)\bTODO\b", r"(?i)\bTBD\b", r"(?m)^\s*…\s*$")
    combined = "\n".join(texts.values())
    for pattern in marker_patterns:
        if re.search(pattern, combined):
            errors.append(f"unresolved editorial marker matched: {pattern}")
    author_inputs = re.findall(r"\[(?:ДАНІ ВІД АВТОРА|ВВЕСТИ АВТОРОМ|AUTHOR)[^]]*\]", combined, flags=re.IGNORECASE)
    if author_inputs:
        warnings.append(f"author/institution inputs still required: {len(author_inputs)}")

    print(
        f"equations={len(all_tags)} captions={caption_count} image_links={image_count} "
        f"references={len(references)} cited={len(citations)}"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in dict.fromkeys(errors)))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
