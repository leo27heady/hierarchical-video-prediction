#!/usr/bin/env python3
"""Create a compact dissertation source set without altering the long-form archive.

The script keeps headings, equations, tables, citations and selected visual evidence, then
selects the most informative prose paragraphs within each subsection under an explicit word
budget. Selection is deterministic and conservative: opening/closing paragraphs, numeric
receipts, citations and claim-bearing language receive priority; roadmap, restatement and
repeated-scope paragraphs are penalised. The result is a working source for a manual flow pass,
not a replacement for scientific review.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'’\-]*")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FIGURE_ID_RE = re.compile(r"Figure\s+([1-4]\.\d+)", re.I)
CITATION_RE = re.compile(r"\[(?:\d+(?:\s*[–-]\s*\d+)?)(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*\]")

TARGETS = {
    "00_FRONT_MATTER.md": 1650,
    "01_INTRODUCTION.md": 2300,
    "02_CHAPTER_1.md": 6500,
    "03_CHAPTER_2.md": 8200,
    "04_CHAPTER_3.md": 8200,
    "05_CHAPTER_4.md": 9000,
    "06_CONCLUSIONS.md": 1750,
    "08_APPENDICES.md": 1500,
}

# The final compact edition retains the complete scientific visual and table inventory. Page
# control comes from concise interpretation and removal of repeated prose, not by deleting
# evidence-bearing artifacts.
DROP_FIGURES: dict[str, set[str]] = {}
DROP_TABLES: dict[str, set[str]] = {}

LOW_VALUE = (
    "this section",
    "the next section",
    "the remainder",
    "the chapter therefore",
    "the chapter closes",
    "it is important to note",
    "in other words",
    "for completeness",
    "the purpose of this paragraph",
    "should be read as",
    "does not mean that",
    "must not be interpreted",
    "is not presented as",
    "is deliberately",
    "the distinction matters",
    "the point is not",
)

HIGH_VALUE = (
    "we define",
    "is defined",
    "the result",
    "confirmed",
    "unresolved",
    "favour",
    "confidence",
    "bootstrap",
    "registered",
    "memory",
    "token",
    "estimand",
    "reversible",
    "hierarchical",
    "flat context",
    "limitation",
    "novelty",
    "practical",
)


@dataclass
class Block:
    text: str
    kind: str
    section: int
    position: int
    total_in_section: int = 0
    tokens: set[str] | None = None

    @property
    def words(self) -> int:
        return len(WORD_RE.findall(self.text))


def split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if b.strip()]


def kind_of(block: str) -> str:
    if HEADING_RE.match(block):
        return "heading"
    if block.startswith("<!--"):
        return "comment"
    if block.startswith("!["):
        return "image"
    if block.startswith("**Figure ") or re.match(r"^Figure\s+[1-4]\.\d+", block):
        return "caption"
    if block.startswith("**Table ") or re.match(r"^Table\s+[1-4]\.\d+", block):
        return "caption"
    if block.startswith("$$") or "\\tag{" in block or block.startswith("\\["):
        return "math"
    lines = block.splitlines()
    if len(lines) >= 2 and all(line.lstrip().startswith("|") for line in lines[:2]):
        return "table"
    if block.startswith("```"):
        return "code"
    if block.startswith("> **Input:**"):
        return "code"
    if block.startswith("- ") or re.match(r"^\d+\.\s", block):
        return "list"
    return "prose"


def tokenise(text: str) -> set[str]:
    stop = {"the", "and", "that", "this", "with", "from", "into", "for", "are", "was", "were", "not", "but", "its", "their", "than", "which", "can", "may", "has", "have", "had"}
    return {w.lower() for w in WORD_RE.findall(text) if len(w) > 3 and w.lower() not in stop}


def paragraph_score(block: Block) -> float:
    text = block.text
    low = text.lower()
    words = max(block.words, 1)
    score = 0.0
    score += 1.1 if block.position == 0 else 0.0
    score += 0.8 if block.position == block.total_in_section - 1 else 0.0
    score += min(1.4, 0.22 * len(CITATION_RE.findall(text)))
    score += 0.75 if re.search(r"\b\d+(?:[.,]\d+)?(?:%|\s*(?:GB|MB|tokens?|frames?|seeds?))\b", text, re.I) else 0.0
    score += 0.40 * sum(phrase in low for phrase in HIGH_VALUE)
    score -= 0.45 * sum(phrase in low for phrase in LOW_VALUE)
    score -= max(0.0, (words - 180) / 130.0)
    score += 0.25 if 45 <= words <= 125 else 0.0
    return score


def remove_figure_blocks(blocks: list[str], filename: str) -> list[str]:
    drops = DROP_FIGURES.get(filename, set())
    if not drops:
        return blocks
    remove: set[int] = set()
    for index, block in enumerate(blocks):
        match = FIGURE_ID_RE.search(block)
        if not match or match.group(1) not in drops or kind_of(block) != "caption":
            continue
        figure_id = match.group(1)
        remove.add(index)
        cursor = index - 1
        while cursor >= 0 and kind_of(blocks[cursor]) == "image":
            remove.add(cursor)
            cursor -= 1
        cursor = index + 1
        if cursor < len(blocks) and kind_of(blocks[cursor]) == "prose" and blocks[cursor].lower().startswith(f"figure {figure_id}".lower()):
            remove.add(cursor)
    return [block for index, block in enumerate(blocks) if index not in remove]


def remove_table_blocks(blocks: list[str], filename: str) -> list[str]:
    drops = DROP_TABLES.get(filename, set())
    if not drops:
        return blocks
    remove: set[int] = set()
    for index, block in enumerate(blocks):
        match = re.search(r"Table\s+([1-4]\.\d+)", block, re.I)
        if not match or match.group(1) not in drops:
            continue
        table_id = match.group(1)
        if kind_of(block) in {"caption", "prose"} or block.startswith("**Table "):
            remove.add(index)
            cursor = index + 1
            while cursor < len(blocks) and kind_of(blocks[cursor]) == "table":
                remove.add(cursor)
                cursor += 1
            if cursor < len(blocks) and kind_of(blocks[cursor]) == "prose" and blocks[cursor].lower().startswith(f"table {table_id}".lower()):
                remove.add(cursor)
    return [block for index, block in enumerate(blocks) if index not in remove]


def compact_caption(block: str) -> str:
    if not block.startswith("**Figure "):
        return block
    source = ""
    source_match = re.search(r"\*Source:\s*(.+?)\*\s*$", block, re.S)
    if source_match:
        source = f" *Source: {source_match.group(1).strip()}*"
    main = re.sub(r"\s*\*Source:.*$", "", block, flags=re.S).strip()
    match = re.match(r"(\*\*Figure\s+[1-4]\.\d+\s+[—-]\s+.+?\.\*\*)(?:\s+.*)?$", main, re.S)
    if match:
        return match.group(1).replace("\n", " ") + source
    # Preserve at most the first interpretive sentence if formatting is irregular.
    sentence = re.split(r"(?<=[.!?])\s+", main, maxsplit=1)[0]
    return sentence.replace("\n", " ") + source


def sentence_compact(text: str, cap: int) -> str:
    words = WORD_RE.findall(text)
    if len(words) <= cap:
        return text
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$*\[])", text)
    if len(sentences) <= 1:
        return " ".join(text.split()[:cap]).rstrip(",;:") + "."
    scored: list[tuple[float, int, str]] = []
    for i, sentence in enumerate(sentences):
        low = sentence.lower()
        score = (1.2 if i == 0 else 0.0) + (0.35 if i == len(sentences) - 1 else 0.0)
        score += 0.8 * len(CITATION_RE.findall(sentence))
        score += 0.55 if re.search(r"\b\d+(?:[.,]\d+)?", sentence) else 0.0
        score += 0.22 * sum(p in low for p in HIGH_VALUE)
        score -= 0.30 * sum(p in low for p in LOW_VALUE)
        scored.append((score, i, sentence))
    chosen: set[int] = {0}
    used = len(WORD_RE.findall(sentences[0]))
    for _, i, sentence in sorted(scored[1:], reverse=True):
        count = len(WORD_RE.findall(sentence))
        if used + count <= cap * 1.08:
            chosen.add(i)
            used += count
    return " ".join(sentences[i] for i in sorted(chosen)).strip()


def select_prose(blocks: list[str], target: int) -> list[str]:
    typed: list[Block] = []
    section = -1
    position_by_section: Counter[int] = Counter()
    for raw in blocks:
        kind = kind_of(raw)
        if kind == "heading":
            section += 1
        pos = position_by_section[section]
        position_by_section[section] += 1
        typed.append(Block(raw, kind, section, pos))
    totals = Counter(b.section for b in typed if b.kind == "prose")
    for b in typed:
        b.total_in_section = totals[b.section]
        if b.kind == "prose":
            b.tokens = tokenise(b.text)

    fixed_words = sum(b.words for b in typed if b.kind != "prose")
    prose_budget = max(500, target - fixed_words)
    prose = [b for b in typed if b.kind == "prose"]
    if sum(b.words for b in prose) <= prose_budget:
        return [b.text for b in typed]

    by_section: dict[int, list[Block]] = {}
    for b in prose:
        by_section.setdefault(b.section, []).append(b)
    current_words = {s: sum(b.words for b in ps) for s, ps in by_section.items()}
    denominator = sum(math.sqrt(v) for v in current_words.values()) or 1
    section_budget = {
        s: max(45, int(prose_budget * math.sqrt(v) / denominator)) for s, v in current_words.items()
    }

    selected_ids: set[int] = set()
    for section, paras in by_section.items():
        budget = section_budget[section]
        chosen: list[Block] = []
        # Preserve the opening claim; preserve the closing bridge only when short or substantive.
        if paras:
            chosen.append(paras[0])
        candidates = [p for p in paras[1:] if p is not paras[-1]]
        if len(paras) > 1 and paragraph_score(paras[-1]) >= 0.25:
            candidates.append(paras[-1])
        while candidates:
            used = sum(p.words for p in chosen)
            if used >= budget:
                break
            best = None
            best_value = -999.0
            chosen_tokens = set().union(*(p.tokens or set() for p in chosen))
            for p in candidates:
                novelty = 1.0
                if p.tokens and chosen_tokens:
                    novelty = 1.0 - len(p.tokens & chosen_tokens) / max(1, len(p.tokens | chosen_tokens))
                value = paragraph_score(p) + 0.9 * novelty
                if used + p.words > budget * 1.16:
                    value -= 1.2
                if value > best_value:
                    best_value, best = value, p
            if best is None or (sum(p.words for p in chosen) >= 70 and best_value < -0.25):
                break
            chosen.append(best)
            candidates.remove(best)
        for p in chosen:
            selected_ids.add(id(p))

    output = [b.text for b in typed if b.kind != "prose" or id(b) in selected_ids]
    # If rounding left unused space, restore highest-value omitted paragraphs globally.
    words = sum(len(WORD_RE.findall(b)) for b in output)
    omitted = sorted((b for b in prose if id(b) not in selected_ids), key=paragraph_score, reverse=True)
    for b in omitted:
        if words + b.words > target * 1.02:
            continue
        output_index = next((i for i, item in enumerate(output) if item == b.text), None)
        # The paragraph is not yet in output; rebuild by source order at the end instead.
        selected_ids.add(id(b))
        words += b.words
    compacted: list[str] = []
    prose_index = 0
    caps = (72, 108, 88, 126, 64, 96)
    for b in typed:
        if b.kind != "prose":
            compacted.append(compact_caption(b.text) if b.kind == "caption" else b.text)
        elif id(b) in selected_ids:
            compacted.append(sentence_compact(b.text, caps[prose_index % len(caps)]))
            prose_index += 1
    return compacted


def renumber(text: str, label: str, chapter: int) -> str:
    pattern = re.compile(rf"{label}\s+({chapter}\.\d+)", re.I)
    seen: list[str] = []
    for match in pattern.finditer(text):
        old = match.group(1)
        if old not in seen:
            seen.append(old)
    mapping = {old: f"{chapter}.{i}" for i, old in enumerate(seen, 1)}
    placeholders: dict[str, str] = {}
    for i, (old, new) in enumerate(mapping.items()):
        token = f"@@{label.upper()}_{chapter}_{i}@@"
        text = re.sub(rf"(?i){label}\s+{re.escape(old)}\b", token, text)
        placeholders[token] = f"{label} {new}"
    for token, replacement in placeholders.items():
        text = text.replace(token, replacement)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    for source_path in sorted(args.source.glob("*.md")):
        if not re.match(r"^0[0-8]_", source_path.name):
            continue
        destination = args.destination / source_path.name
        if source_path.name == "07_REFERENCES.md":
            shutil.copy2(source_path, destination)
            continue
        if source_path.name not in TARGETS:
            shutil.copy2(source_path, destination)
            continue
        blocks = remove_figure_blocks(split_blocks(source_path.read_text(encoding="utf-8")), source_path.name)
        blocks = remove_table_blocks(blocks, source_path.name)
        compact = select_prose(blocks, TARGETS[source_path.name])
        text = "\n\n".join(compact).rstrip() + "\n"
        chapter_match = re.match(r"0([2-5])_", source_path.name)
        if chapter_match:
            chapter = int(chapter_match.group(1)) - 1
            text = renumber(text, "Figure", chapter)
            text = renumber(text, "Table", chapter)
        destination.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
