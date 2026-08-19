#!/usr/bin/env python3
"""Audit structural parity and mandatory language rules for the Ukrainian dissertation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "docs" / "dissertation" / "condensed_english"
UA = ROOT / "docs" / "dissertation" / "condensed_translated_humanized_ukrainian"
PAIRS = [
    ("00_FRONT_MATTER.md", "00_FRONT_MATTER.md"),
    ("01_INTRODUCTION.md", "01_INTRODUCTION.md"),
    ("02_CHAPTER_1.md", "02_CHAPTER_1.md"),
    ("03_CHAPTER_2.md", "03_CHAPTER_2.md"),
    ("04_CHAPTER_3.md", "04_CHAPTER_3.md"),
    ("05_CHAPTER_4.md", "05_CHAPTER_4.md"),
    ("06_CONCLUSIONS.md", "06_CONCLUSIONS.md"),
    ("07_REFERENCES.md", "07_REFERENCES.md"),
    ("08_APPENDICES.md", "08_APPENDICES.md"),
]

# The Ukrainian Chapter 1 contains one dissertation-native video-prediction
# panel added during the author's manual review. It supplements Figure 1.6 and
# therefore has no counterpart in the earlier condensed English source.
UA_ONLY_IMAGES = {
    "02_CHAPTER_1.md": ["../../figures/restructured/ch1_fig1_6b_video_prediction_example.png"],
}

HEADING = re.compile(r"^(#{1,6})\s+", re.M)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TAG = re.compile(r"\\tag\{([^}]+)\}")
CITATION = re.compile(r"\[(\d+(?:\s*[–-]\s*\d+)?(?:,\s+\d+(?:\s*[–-]\s*\d+)?)*)\]")
REF_LINE = re.compile(r"^(\d+)\.\s", re.M)
PROSE_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def count_table_rows(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("|") and not re.match(r"^\|[-:| ]+\|$", line))


def prose_paragraph_words(text: str) -> list[int]:
    values: list[int] = []
    for block in re.split(r"\n\s*\n", text):
        s = block.strip()
        if not s or s.startswith(("#", "|", "!", "$$", "```", ">")):
            continue
        if re.fullmatch(r"(?:\d+\.|[-*])\s+.*", s, flags=re.S):
            continue
        s = re.sub(r"`[^`]+`|\$[^$]+\$|\[[^\]]+\]\([^)]+\)", " ", s)
        words = re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9]+", s)
        if words:
            values.append(len(words))
    return values


def sentence_opening_runs(text: str) -> list[str]:
    plain = re.sub(r"```.*?```|\$\$.*?\$\$|\|.*", " ", text, flags=re.S)
    openings: list[str] = []
    for sentence in PROSE_SENTENCE.split(plain):
        words = re.findall(r"[А-Яа-яІіЇїЄєҐґA-Za-z]+", sentence.strip())
        if words:
            openings.append(words[0].casefold())
    return [openings[i] for i in range(len(openings) - 2) if len(set(openings[i:i+3])) == 1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []
    total_images = total_tags = total_tables = 0

    for en_name, ua_name in PAIRS:
        en_path, ua_path = EN / en_name, UA / ua_name
        if not en_path.exists() or not ua_path.exists():
            errors.append(f"missing pair: {en_name} / {ua_name}")
            continue
        en = en_path.read_text(encoding="utf-8")
        ua = ua_path.read_text(encoding="utf-8")

        en_heads = Counter(len(x) for x in HEADING.findall(en))
        ua_heads = Counter(len(x) for x in HEADING.findall(ua))
        if en_heads != ua_heads:
            errors.append(f"{ua_name}: heading-level counts {dict(ua_heads)} != EN {dict(en_heads)}")

        en_images, ua_images = IMAGE.findall(en), IMAGE.findall(ua)
        parity_ua_images = ua_images.copy()
        for allowed in UA_ONLY_IMAGES.get(ua_name, []):
            if allowed in parity_ua_images:
                parity_ua_images.remove(allowed)
            else:
                errors.append(f"{ua_name}: missing approved Ukrainian-only image {allowed}")
        if en_images != parity_ua_images:
            errors.append(f"{ua_name}: image path/order mismatch ({len(ua_images)} vs {len(en_images)})")
        for rel in ua_images:
            if not (ua_path.parent / rel).resolve().exists():
                errors.append(f"{ua_name}: broken image {rel}")

        en_tags, ua_tags = TAG.findall(en), TAG.findall(ua)
        if en_tags != ua_tags:
            errors.append(f"{ua_name}: equation tags mismatch {ua_tags} != {en_tags}")

        en_rows, ua_rows = count_table_rows(en), count_table_rows(ua)
        if en_rows != ua_rows:
            errors.append(f"{ua_name}: Markdown table-row mismatch {ua_rows} != {en_rows}")

        if ua.count("$$") % 2:
            errors.append(f"{ua_name}: unbalanced display-math delimiters")
        no_display = re.sub(r"\$\$.*?\$\$", "", ua, flags=re.S)
        if len(re.findall(r"(?<!\\)\$", no_display)) % 2:
            errors.append(f"{ua_name}: unbalanced inline-math delimiters")
        if "—" in ua:
            errors.append(f"{ua_name}: contains forbidden em dash")
        if re.search(r"узгоджен(?:ість|ий|а|е|і|ого|ої|их|ими)", ua, re.I):
            errors.append(f"{ua_name}: contains forbidden consistency translation")
        if "�" in ua or "Ã" in ua or "Ð" in ua:
            errors.append(f"{ua_name}: possible mojibake")
        if re.search(r"\n\s*[.,;:]\s*\n", ua):
            errors.append(f"{ua_name}: isolated punctuation line")

        en_cites = CITATION.findall(en)
        ua_cites = CITATION.findall(ua)
        if en_cites != ua_cites:
            warnings.append(f"{ua_name}: citation occurrence/order differs ({len(ua_cites)} vs {len(en_cites)})")

        sizes = prose_paragraph_words(ua)
        if sizes:
            over = sum(v > 210 for v in sizes)
            print(f"{ua_name}: paragraphs={len(sizes)}, median={statistics.median(sizes):.0f}, max={max(sizes)}, >210={over}")
            if over:
                warnings.append(f"{ua_name}: {over} prose paragraphs exceed 210 words")
        runs = sentence_opening_runs(ua)
        if runs:
            warnings.append(f"{ua_name}: repeated three-sentence openings: {Counter(runs)}")

        total_images += len(ua_images)
        total_tags += len(ua_tags)
        total_tables += ua_rows

    refs = (UA / "07_REFERENCES.md").read_text(encoding="utf-8")
    numbers = [int(x) for x in REF_LINE.findall(refs)]
    if numbers != list(range(1, 175)):
        errors.append("07_REFERENCES.md: bibliography numbering is not exactly 1..174")

    print(f"TOTAL: images={total_images}, equation_tags={total_tags}, table_rows={total_tables}, references={len(numbers)}")
    if warnings:
        print("WARNINGS:")
        for item in warnings:
            print(f"- {item}")
    if errors:
        print("ERRORS:")
        for item in errors:
            print(f"- {item}")
        return 1
    print("PASS: structural parity and mandatory mechanical checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
