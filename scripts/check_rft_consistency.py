"""Strict consistency gate for canonical packaging texts and R/F/T discussion resources."""
from __future__ import annotations

import re
from pathlib import Path


ARTICLE_CHAPTERS = [
    Path("docs/draft/CH1_INTRO.md"),
    Path("docs/draft/CH2_TESTBED.md"),
    Path("docs/draft/CH3_MECHANISM.md"),
    Path("docs/draft/CH4_RESULTS.md"),
    Path("docs/draft/CH5_GENERALIZATION.md"),
    Path("docs/draft/CH6_REVERSIBILITY.md"),
    Path("docs/draft/CH7_RELATED.md"),
    Path("docs/draft/CH8_CONCLUSION.md"),
]

BINDING_ARTICLE = Path("docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md")

AUTHORITATIVE = [
    Path("README.md"),
    Path("docs/CLAIM_LEDGER.md"),
    BINDING_ARTICLE,
    Path("docs/UNTOUCHED_CONFIRMATION_PROTOCOL_2026-07-11.md"),
    Path("docs/PHASE_C_V2_PROTOCOL_2026-07-11.md"),
]

DISCUSSION_RESOURCES = [
    Path("docs/dissertation/REACH_FIDELITY_TRUST_SYNTHESIS_2026-07-11.md"),
]

PROHIBITED = {
    r"exact arithmetic edge": "tested configurations do not establish an exact edge",
    r"unique continuous crossing is established": "crossing uniqueness is not established",
    r"trustworthy-memory law": "trust is a measured boundary pending replication",
    r"B8.?60k uses twice the data": "c2 changes the optimizer regime at matched examples",
    r"regression (tie|equivalence)": "opposite seed signs are heterogeneity",
    r"\[EXPECTED-PhaseC": "legacy Phase C is blocked rather than projected",
}

REQUIRED_ARTICLE_RECORD = [
    "context packaging",
    "reach",
    "fidelity",
    "trust",
    "untouched",
    "configuration effect",
    "coverage",
]


def main() -> None:
    failures = []
    checked = AUTHORITATIVE + ARTICLE_CHAPTERS + DISCUSSION_RESOURCES
    for path in checked:
        if not path.is_file():
            failures.append(f"missing canonical text: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, reason in PROHIBITED.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                line = text[:match.start()].count("\n") + 1
                context = text[max(0, match.start() - 300):match.start()].lower()
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line_text = text[line_start:line_end if line_end >= 0 else len(text)].lower()
                if (any(word in context for word in ("retired", "avoid", "not establish", "does not establish"))
                        or any(word in line_text for word in ("retired", "do not", "does not", "not a", "avoid"))):
                    continue
                failures.append(f"{path}:{line}: {match.group(0)!r}: {reason}")
    article = BINDING_ARTICLE.read_text(encoding="utf-8") + "\n" + "\n".join(
        path.read_text(encoding="utf-8") for path in ARTICLE_CHAPTERS
    )
    for phrase in REQUIRED_ARTICLE_RECORD:
        if phrase.lower() not in article.lower():
            failures.append(f"canonical CH1-CH8 record lacks required concept {phrase!r}")
    for path in ARTICLE_CHAPTERS:
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:8]).lower()
        if "detailed" not in header or "not the binding" not in header:
            failures.append(f"article chapter lacks the detailed/non-binding authority marker: {path}")
    binding_text = " ".join(
        BINDING_ARTICLE.read_text(encoding="utf-8").lower().split()
    )
    # Authority follows from the canonical final-source path plus a stable scientific identity,
    # not from editorial metadata that would leak into the submission manuscript.
    for marker in (
        "# when does hierarchical video memory pay?",
        "## abstract",
        "hierarchy buys reach",
        "flat context buys fidelity",
    ):
        if marker not in binding_text:
            failures.append(
                f"canonical final article lacks stable scientific identity marker {marker!r}"
            )
    introduction = Path("docs/dissertation/rewrite/01_VSTUP.md").read_text(encoding="utf-8")
    for publication in ("[A1]", "[A2]", "[A3]", "[A4]"):
        if publication not in introduction:
            failures.append(f"dissertation introduction lost publication strand {publication}")
    if "seven" not in introduction.lower() and "7" not in introduction:
        failures.append("dissertation introduction no longer exposes the fixed seven-task structure")
    if failures:
        print("RFT CONSISTENCY FAIL")
        for failure in failures:
            print("  ", failure)
        raise SystemExit(1)
    print(
        f"PACKAGING/RFT CONSISTENCY PASS: {len(AUTHORITATIVE)} authoritative texts, "
        f"{len(ARTICLE_CHAPTERS)} detailed source chapters and {len(DISCUSSION_RESOURCES)} "
        "discussion resources; fixed publication/task strands present"
    )


if __name__ == "__main__":
    main()
