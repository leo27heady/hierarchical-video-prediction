#!/usr/bin/env python3
"""Mechanical quality gates for the page-bounded dissertation source."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "dissertation" / "condensed_english"
FILES = [SOURCE / f"0{i}_{name}.md" for i, name in enumerate((
    "FRONT_MATTER", "INTRODUCTION", "CHAPTER_1", "CHAPTER_2", "CHAPTER_3",
    "CHAPTER_4", "CONCLUSIONS", "REFERENCES", "APPENDICES"
))]
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
TARGETS = {
    "01_INTRODUCTION.md": (1800, 2400),
    "02_CHAPTER_1.md": (5000, 6800),
    "03_CHAPTER_2.md": (6500, 8300),
    "04_CHAPTER_3.md": (6500, 8300),
    "05_CHAPTER_4.md": (7500, 9200),
    "06_CONCLUSIONS.md": (1200, 1800),
}
EXPECTED_FIGURES = {1: 25, 2: 27, 3: 17, 4: 17}
EXPECTED_TABLES = {1: 4, 2: 10, 3: 6, 4: 10}
EXPECTED_EQUATIONS = {1: 14, 2: 41, 3: 40, 4: 7}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def prose_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block.startswith(("#", "|", "![", "**Figure", "**Table", "$$", "<!--", "```", "- ")):
            continue
        if re.match(r"^\d+\.\s", block):
            continue
        paragraphs.append(block)
    return paragraphs


def main() -> int:
    errors: list[str] = []
    total_scientific = 0
    total_figures = 0
    total_tables = 0
    all_text = ""
    scientific_text = ""
    for path in FILES:
        if not path.exists():
            errors.append(f"missing source: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        all_text += "\n" + text
        if path.name != "07_REFERENCES.md":
            scientific_text += "\n" + text
        words = word_count(text)
        figures = re.findall(r"(?m)^\*\*Figure\s+(\d+\.\d+)", text)
        tables = re.findall(r"(?m)^\*\*Table\s+(\d+\.\d+)", text)
        total_figures += len(figures)
        total_tables += len(tables)
        if path.name in TARGETS:
            lo, hi = TARGETS[path.name]
            if not lo <= words <= hi:
                errors.append(f"{path.name}: {words} words outside {lo}-{hi}")
            total_scientific += words
        for label, ids in (("Figure", figures), ("Table", tables)):
            if ids:
                chapter = int(ids[0].split(".")[0])
                expected = [f"{chapter}.{i}" for i in range(1, len(ids) + 1)]
                if ids != expected:
                    errors.append(f"{path.name}: {label} sequence {ids} expected {expected}")
        chapter_match = re.match(r"0([2-5])_CHAPTER_([1-4])\.md", path.name)
        if chapter_match:
            chapter = int(chapter_match.group(2))
            if len(figures) != EXPECTED_FIGURES[chapter]:
                errors.append(f"{path.name}: {len(figures)} figures, expected {EXPECTED_FIGURES[chapter]}")
            if len(tables) != EXPECTED_TABLES[chapter]:
                errors.append(f"{path.name}: {len(tables)} tables, expected {EXPECTED_TABLES[chapter]}")
            equation_tags = re.findall(rf"\\tag\{{{chapter}\.[^}}]+\}}", text)
            if len(equation_tags) != EXPECTED_EQUATIONS[chapter]:
                errors.append(f"{path.name}: {len(equation_tags)} equations, expected {EXPECTED_EQUATIONS[chapter]}")
        for link in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
            target = (path.parent / link).resolve()
            if not target.exists():
                errors.append(f"{path.name}: broken image {link}")
        paragraphs = prose_paragraphs(text)
        long_paragraphs = [word_count(p) for p in paragraphs if word_count(p) > 170]
        if long_paragraphs:
            errors.append(f"{path.name}: prose paragraph(s) over 170 words: {long_paragraphs}")
        starts = []
        for p in paragraphs:
            first = WORD_RE.findall(re.sub(r"^\*+", "", p))
            starts.append(first[0].lower() if first else "")
        for a, b, c in zip(starts, starts[1:], starts[2:]):
            if a and a == b == c:
                errors.append(f"{path.name}: three consecutive paragraphs start with '{a}'")
                break
    if not 29000 <= total_scientific <= 36000:
        errors.append(f"scientific text total {total_scientific} outside 29,000-36,000")
    if total_figures < 60:
        errors.append(f"main figure count {total_figures} is below the complete visual baseline of 60")
    if total_tables < 24:
        errors.append(f"main table count {total_tables} is below the decision-relevant baseline of 24")
    for marker in ("[REF:", "[UA-PASS:", "[AUTHOR:"):
        if marker in all_text:
            errors.append(f"legacy marker remains: {marker}")
    references = {
        int(number)
        for number in re.findall(r"(?m)^(\d+)\.\s", (SOURCE / "07_REFERENCES.md").read_text(encoding="utf-8"))
    }
    citations: set[int] = set()
    for body in re.findall(r"\[([0-9][0-9,\s–-]*)\]", scientific_text):
        for item in re.split(r"\s*,\s*", body):
            bounds = re.split(r"\s*[–-]\s*", item)
            if len(bounds) == 2:
                citations.update(number for number in range(int(bounds[0]), int(bounds[1]) + 1) if number > 0)
            elif bounds[0] and int(bounds[0]) > 0:
                citations.add(int(bounds[0]))
    if citations - references:
        errors.append(f"undefined references: {sorted(citations - references)}")
    if references - citations:
        errors.append(f"uncited references: {sorted(references - citations)}")
    required_content = {
        "complete rollout algorithm": "> **Input:** warm-up frames",
        "V-BLT abstract result": "Spatial representation is examined through Vision Byte Latent Transformer",
        "fixed versus adaptive boundary": "Dynamic region controller",
        "tested unresolved frontier": "1,792-token comparison is unresolved",
        "external support-identity boundary": "flat context better preserves digit identity",
    }
    for label, phrase in required_content.items():
        if phrase not in all_text:
            errors.append(f"required content missing: {label}")
    print(f"scientific_words={total_scientific} figures={total_figures} tables={total_tables}")
    if errors:
        print("FAIL")
        print("\n".join(f"- {e}" for e in errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
