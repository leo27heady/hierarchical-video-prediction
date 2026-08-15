"""Assemble the page-bounded English dissertation Markdown source."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "docs" / "dissertation" / "condensed_english"
OUT = ROOT / "docs" / "dissertation" / "final" / "DISSERTATION_CONDENSED_ENGLISH.md"
SECTIONS = [WORK / f"0{i}_{name}.md" for i, name in enumerate((
    "FRONT_MATTER", "INTRODUCTION", "CHAPTER_1", "CHAPTER_2", "CHAPTER_3",
    "CHAPTER_4", "CONCLUSIONS", "REFERENCES", "APPENDICES"
))]


def main() -> None:
    chunks = []
    for path in SECTIONS:
        text = path.read_text(encoding="utf-8").strip()
        if text.count("$$") % 2:
            raise ValueError(f"unbalanced display math: {path}")
        chunks.append(text)
    final = "\n\n---\n\n".join(chunks).rstrip() + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(final, encoding="utf-8", newline="\n")
    for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", final):
        resolved = (OUT.parent / target).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"broken image: {target} -> {resolved}")
    print(OUT)


if __name__ == "__main__":
    main()
