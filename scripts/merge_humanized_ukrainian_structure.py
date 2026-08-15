#!/usr/bin/env python3
"""Restore Markdown structure around humanized Ukrainian dissertation prose.

The structured pre-humanization translation supplies headings, equations, images, tables,
captions, citations, and author-input markers. Humanized prose is retained when a monotonic
same-language alignment is reliable. Paragraphs containing inline mathematics stay on the
structured side because the humanizer removed subscripts and braces from several formulas.

This script writes a review candidate only. It does not replace the working dissertation files.
"""

from __future__ import annotations

import argparse
import difflib
import re
from dataclasses import dataclass
from pathlib import Path


WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+", re.UNICODE)
CITATION_RE = re.compile(
    r"\[(?:\d+(?:\s*[–-]\s*\d+)?)(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*\]"
)
AUTHOR_RE = re.compile(r"\*\*\[[^\]]*(?:АВТОР|AUTHOR)[^\]]*\]\*\*", re.I)


@dataclass
class Block:
    text: str
    kind: str


def blocks(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def kind(text: str) -> str:
    lines = text.splitlines()
    first = lines[0].strip()
    if first.startswith("#"):
        return "heading"
    if first.startswith("!["):
        return "image"
    if first.startswith("$$") or ("\\begin{" in text and "$" in text):
        return "equation"
    if all(line.lstrip().startswith("|") for line in lines):
        return "table"
    if re.match(r"^\*\*(?:Рисунок|Таблиця|Алгоритм|Figure|Table|Algorithm)\b", first, re.I):
        return "caption"
    if re.match(r"^(?:Рисунок|Таблиця|Алгоритм)\s+\d", first, re.I):
        return "caption_plain"
    if re.match(r"^\d+(?:\.\d+){0,2}\s+\S", first) and len(lines) == 1:
        return "heading_plain"
    if first in {"Вступ", "Загальні висновки", "Список літератури", "Додатки"}:
        return "heading_plain"
    if first.startswith("```" ) or first.startswith(">"):
        return "fixed"
    if re.match(r"^(?:[-*+]\s+|\d+\.\s+)", first):
        return "list"
    return "prose"


def normalized(text: str) -> str:
    text = CITATION_RE.sub(" ", text)
    text = AUTHOR_RE.sub(" ", text)
    text = re.sub(r"[*_`#|$\\{}\[\]()]", " ", text)
    return " ".join(token.lower() for token in WORD_RE.findall(text))


def similarity(left: str, right: str) -> float:
    a = normalized(left)
    b = normalized(right)
    if not a or not b:
        return 0.0
    seq = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    sa = set(a.split())
    sb = set(b.split())
    jac = len(sa & sb) / max(1, len(sa | sb))
    ratio = max(len(a), len(b)) / max(1, min(len(a), len(b)))
    length_penalty = 0.0 if ratio <= 1.65 else min(0.30, (ratio - 1.65) * 0.12)
    return 0.62 * seq + 0.38 * jac - length_penalty


def human_prose_candidates_from_blocks(raw: list[str]) -> list[str]:
    retained: list[str] = []
    for index, block in enumerate(raw):
        k = kind(block)
        if k not in {"prose", "list"}:
            continue
        lines = block.splitlines()
        if any(line.lstrip().startswith("|") for line in lines):
            continue
        if "$$" in block or "\\tag{" in block:
            continue
        if re.search(r"(?m)^(?:Рисунок|Таблиця|Алгоритм)\s+\d", block):
            continue
        # The humanizer converted image alt text to a short line immediately before a caption.
        if index + 1 < len(raw) and kind(raw[index + 1]) == "caption_plain" and len(block.split()) < 18:
            continue
        if block.startswith("$") or block.endswith("$") or "\\tag{" in block:
            continue
        retained.append(block)
    return retained


def heading_key(text: str) -> str | None:
    plain = re.sub(r"^#+\s*", "", text).strip()
    match = re.match(r"^(\d+(?:\.\d+)*)\b", plain)
    if match:
        return match.group(1)
    lowered = normalized(plain)
    for key in ("вступ", "загальні висновки", "додатки", "список літератури"):
        if lowered.startswith(key):
            return key
    return None


def locate_human_headings(source_blocks: list[Block], human_blocks: list[str]) -> dict[int, int]:
    """Locate the humanized counterpart of every structured heading monotonically."""
    locations: dict[int, int] = {}
    cursor = 0
    for source_index, block in enumerate(source_blocks):
        if block.kind != "heading":
            continue
        source_key = heading_key(block.text)
        best: tuple[float, int] | None = None
        for human_index in range(cursor, len(human_blocks)):
            candidate = human_blocks[human_index]
            candidate_key = heading_key(candidate)
            if source_key and candidate_key == source_key:
                best = (1.0, human_index)
                break
            score = similarity(block.text, candidate)
            if best is None or score > best[0]:
                best = (score, human_index)
            if human_index - cursor > 18 and best and best[0] >= 0.35:
                break
        if best is None or best[0] < 0.18:
            continue
        locations[source_index] = best[1]
        cursor = best[1] + 1
    return locations


def append_structured_metadata(human: str, structured: str) -> str:
    extras: list[str] = []
    for match in CITATION_RE.findall(structured):
        if match not in human and match not in extras:
            extras.append(match)
    for match in AUTHOR_RE.findall(structured):
        if match not in human and match not in extras:
            extras.append(match)
    if extras:
        human = human.rstrip()
        if human and human[-1] not in ".!?]":
            human += "."
        human += " " + " ".join(extras)
    return human


def align_prose(structured: list[str], human: list[str]) -> tuple[dict[int, tuple[int, int, float]], list[str]]:
    """Greedy monotonic 1-to-(1..3) alignment with conservative acceptance."""
    mapping: dict[int, tuple[int, int, float]] = {}
    warnings: list[str] = []
    cursor = 0
    for i, source in enumerate(structured):
        best: tuple[float, int, int] | None = None
        upper = min(len(human), cursor + 8)
        for start in range(cursor, upper):
            for width in (1, 2, 3):
                end = start + width
                if end > len(human):
                    continue
                candidate = "\n\n".join(human[start:end])
                score = similarity(source, candidate)
                # Prefer a nearby, narrower mapping when scores are close.
                score -= 0.012 * (start - cursor) + 0.008 * (width - 1)
                if best is None or score > best[0]:
                    best = (score, start, end)
        if best is None:
            warnings.append(f"prose {i}: no candidate")
            continue
        score, start, end = best
        if score < 0.23:
            warnings.append(f"prose {i}: weak score {score:.3f}")
            continue
        mapping[i] = (start, end, score)
        cursor = end
    if cursor < len(human):
        warnings.append(f"{len(human) - cursor} trailing human prose block(s) unconsumed")
    return mapping, warnings


def merge(structured_text: str, human_text: str) -> tuple[str, list[str], list[float]]:
    source_blocks = [Block(text=b, kind=kind(b)) for b in blocks(structured_text)]
    human_blocks = blocks(human_text)
    heading_locations = locate_human_headings(source_blocks, human_blocks)
    heading_source_indices = [i for i, b in enumerate(source_blocks) if b.kind == "heading"]
    warnings: list[str] = []
    if len(heading_locations) != len(heading_source_indices):
        warnings.append(
            f"located {len(heading_locations)}/{len(heading_source_indices)} humanized headings"
        )
    replacements: dict[int, tuple[str, float]] = {}
    boundaries = heading_source_indices + [len(source_blocks)]
    for position, source_start in enumerate(heading_source_indices):
        source_end = boundaries[position + 1]
        human_heading = heading_locations.get(source_start)
        if human_heading is None:
            continue
        next_human = len(human_blocks)
        for later_source in heading_source_indices[position + 1 :]:
            if later_source in heading_locations:
                next_human = heading_locations[later_source]
                break
        source_indices = [
            i for i in range(source_start + 1, source_end) if source_blocks[i].kind == "prose"
        ]
        human_prose = human_prose_candidates_from_blocks(
            human_blocks[human_heading + 1 : next_human]
        )
        source_prose = [source_blocks[i].text for i in source_indices]
        mapping, segment_warnings = align_prose(source_prose, human_prose)
        warnings.extend(f"section {source_blocks[source_start].text}: {w}" for w in segment_warnings)
        for local_index, source_index in enumerate(source_indices):
            aligned = mapping.get(local_index)
            if aligned is None:
                continue
            start, end, score = aligned
            replacements[source_index] = ("\n\n".join(human_prose[start:end]), score)
    output: list[str] = []
    scores: list[float] = []
    for source_index, block in enumerate(source_blocks):
        if block.kind != "prose":
            output.append(block.text)
            continue
        aligned = replacements.get(source_index)
        # Inline mathematics and fragile author fields stay structurally exact.
        fragile = "$" in block.text or "**[" in block.text
        if aligned is None or fragile:
            output.append(block.text)
        else:
            replacement, score = aligned
            output.append(append_structured_metadata(replacement, block.text))
            scores.append(score)
    return "\n\n".join(output).rstrip() + "\n", warnings, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structured", type=Path)
    parser.add_argument("humanized", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    merged, warnings, scores = merge(
        args.structured.read_text(encoding="utf-8"),
        args.humanized.read_text(encoding="utf-8"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(merged, encoding="utf-8")
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"aligned={len(scores)} mean_score={mean:.3f} warnings={len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
