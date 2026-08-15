"""Assemble and validate the restructured English dissertation Markdown source."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "docs" / "dissertation" / "restructured_english"
OUT = ROOT / "docs" / "dissertation" / "final" / "DISSERTATION_RESTRUCTURED_ENGLISH.md"

SECTIONS = [
    WORK / "00_FRONT_MATTER.md",
    WORK / "01_INTRODUCTION.md",
    WORK / "02_CHAPTER_1.md",
    WORK / "03_CHAPTER_2.md",
    WORK / "04_CHAPTER_3.md",
    WORK / "05_CHAPTER_4.md",
    WORK / "06_CONCLUSIONS.md",
    WORK / "07_REFERENCES.md",
    WORK / "08_APPENDICES.md",
]

REGISTRY = (
    ROOT
    / "docs"
    / "dissertation"
    / "rewrite"
    / "appendices"
    / "APPENDIX_CH4_CONFIGURATION_AND_EVIDENCE_REGISTRY.md"
)


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def word_count(text: str) -> int:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    return len(re.findall(r"\b[\w'’\-]+\b", text, flags=re.UNICODE))


def check_source(path: Path, text: str) -> None:
    prohibited = {
        "unbound reference marker": r"\[REF:",
        "unresolved figure instruction": r"NEW-FIGURE",
        "draft placeholder": r"\b(?:TODO|TBD)\b",
    }
    for label, pattern in prohibited.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise ValueError(f"{path}: {label} remains")

    if text.count("$$") % 2:
        raise ValueError(f"{path}: unbalanced display-math delimiters")


def check_images(final_text: str) -> None:
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", final_text):
        if re.match(r"^[a-z]+://", target, flags=re.IGNORECASE):
            continue
        resolved = (OUT.parent / target).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Broken image link: {target} -> {resolved}")


def check_numeric_citations(final_text: str) -> None:
    reference_numbers = {
        int(number)
        for number in re.findall(r"(?m)^(\d+)\.\s", final_text)
    }
    cited: set[int] = set()
    body = final_text.split("# References", maxsplit=1)[0]
    for group in re.findall(r"\[([0-9,;\s\-–]+)\]", body):
        for first, last in re.findall(r"(\d+)(?:\s*[\-–]\s*(\d+))?", group):
            a = int(first)
            b = int(last) if last else a
            cited.update(range(min(a, b), max(a, b) + 1))
    cited.discard(0)
    undefined = sorted(cited - reference_numbers)
    if undefined:
        raise ValueError(f"Undefined numeric citations: {undefined}")


def main() -> None:
    chunks: list[str] = []
    for path in SECTIONS:
        text = read(path)
        check_source(path, text)
        chunks.append(text)

    registry = read(REGISTRY)
    if "[REF:" in registry:
        raise ValueError(f"{REGISTRY}: unbound reference marker remains")
    chunks.append(registry)

    final_text = "\n\n---\n\n".join(chunks).rstrip() + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(final_text, encoding="utf-8", newline="\n")

    check_images(final_text)
    check_numeric_citations(final_text)

    main_text = "\n\n".join(read(path) for path in SECTIONS[1:7])
    print(f"assembled={OUT}")
    print(f"main_scientific_words={word_count(main_text)}")
    print(f"total_review_words={word_count(final_text)}")
    print(f"figures={len(re.findall(r'!\[', final_text))}")
    print(f"author_inputs={len(re.findall(r'\[AUTHOR INPUT:', final_text))}")


if __name__ == "__main__":
    main()
