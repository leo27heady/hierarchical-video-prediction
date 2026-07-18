"""Assemble the final English article and dissertation Markdown artifacts.

Scientific chapter sources remain modular. Reference binding is performed first, after which this
script concatenates the dissertation in institutional order, copies the binding fourth article
to its release location, validates local figure links, and writes a mechanical assembly report.
"""
from __future__ import annotations

import re
from pathlib import Path

from bind_dissertation_references import main as bind_references


ROOT = Path(__file__).resolve().parents[1]
REWRITE = ROOT / "docs" / "dissertation" / "rewrite"
DISS_FINAL = ROOT / "docs" / "dissertation" / "final"
ARTICLE_SOURCE = ROOT / "docs" / "draft" / "ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md"
ARTICLE_FINAL = ROOT / "docs" / "article" / "final"

DISSERTATION_PARTS = [
    REWRITE / "00_FRONT_MATTER.md",
    DISS_FINAL / "01_VSTUP.md",
    DISS_FINAL / "02_ROZDIL1.md",
    DISS_FINAL / "03_ROZDIL2.md",
    DISS_FINAL / "04_ROZDIL3.md",
    DISS_FINAL / "05_ROZDIL4.md",
    DISS_FINAL / "06_VYSNOVKY.md",
    DISS_FINAL / "07_REFERENCES_BOUND.md",
]


def _validate_images(path: Path, text: str) -> list[Path]:
    missing: list[Path] = []
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if re.match(r"^[a-z]+://", target, flags=re.IGNORECASE):
            continue
        target = target.split(" ", 1)[0].strip("<>")
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            missing.append(resolved)
    return missing


def _caption_count(text: str, label: str) -> int:
    return len(re.findall(rf"\*\*{label}\s+[A-Za-z0-9]", text, flags=re.IGNORECASE))


def main() -> None:
    bind_references()

    separator = "\n\n<div style=\"page-break-after: always;\"></div>\n\n"
    dissertation = separator.join(
        path.read_text(encoding="utf-8").strip() for path in DISSERTATION_PARTS
    ) + "\n"
    dissertation_path = DISS_FINAL / "DISSERTATION_MANUSCRIPT.md"
    dissertation_path.write_text(dissertation, encoding="utf-8", newline="\n")

    ARTICLE_FINAL.mkdir(parents=True, exist_ok=True)
    article = ARTICLE_SOURCE.read_text(encoding="utf-8")
    # The source lives one directory below docs; the final copy lives two directories below.
    article = article.replace("](../figures/", "](../../figures/")
    article_path = ARTICLE_FINAL / "ARTICLE_REACH_FIDELITY_TRUST.md"
    article_path.write_text(article, encoding="utf-8", newline="\n")

    missing = {
        dissertation_path: _validate_images(dissertation_path, dissertation),
        article_path: _validate_images(article_path, article),
    }
    missing = {path: items for path, items in missing.items() if items}
    if missing:
        detail = "\n".join(f"{path}: {items}" for path, items in missing.items())
        raise FileNotFoundError(f"Missing manuscript image targets:\n{detail}")

    forbidden = ("[REF:", "[REF]", "[UA-PASS:", "[EXPECTED", "[PENDING")
    for path, text in ((dissertation_path, dissertation), (article_path, article)):
        present = [marker for marker in forbidden if marker in text]
        if present:
            raise RuntimeError(f"{path} contains unresolved markers: {present}")

    references = len(re.findall(r"^\d+\. ", (DISS_FINAL / "07_REFERENCES_BOUND.md").read_text(
        encoding="utf-8"
    ), flags=re.MULTILINE))
    report = "\n".join(
        [
            "# Final Manuscript Assembly Report",
            "",
            f"- Dissertation words: {len(dissertation.split()):,}",
            f"- Article words: {len(article.split()):,}",
            f"- Dissertation figure captions: {_caption_count(dissertation, 'Figure')}",
            f"- Dissertation table captions: {_caption_count(dissertation, 'Table')}",
            f"- Bound external references: {references}",
            f"- Remaining author-input markers: {dissertation.count('[AUTHOR:')}",
            "- Local figure-link validation: PASS",
            "- Reference-marker validation: PASS",
            "- Projected/pending-result marker validation: PASS",
            "",
            "Remaining author-input markers are administrative facts listed in "
            "`docs/dissertation/AUTHOR_INPUT_CHECKLIST.md`; they do not represent missing "
            "scientific results.",
            "",
        ]
    )
    (DISS_FINAL / "MANUSCRIPT_ASSEMBLY_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    print("FINAL_MANUSCRIPTS_ASSEMBLED")
    print(report)


if __name__ == "__main__":
    main()
