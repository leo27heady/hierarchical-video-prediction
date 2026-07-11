"""Strict consistency gate for canonical reach–fidelity–trust texts."""
from __future__ import annotations

import re
from pathlib import Path


CANONICAL = [
    Path("README.md"),
    Path("docs/CLAIM_LEDGER.md"),
    Path("docs/draft/ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md"),
    Path("docs/dissertation/REACH_FIDELITY_TRUST_SYNTHESIS_2026-07-11.md"),
    Path("docs/UNTOUCHED_CONFIRMATION_PROTOCOL_2026-07-11.md"),
    Path("docs/PHASE_C_V2_PROTOCOL_2026-07-11.md"),
]

PROHIBITED = {
    r"exact arithmetic edge": "tested configurations do not establish an exact edge",
    r"unique continuous crossing is established": "crossing uniqueness is not established",
    r"trustworthy-memory law": "trust is a measured boundary pending replication",
    r"B8.?60k uses twice the data": "c2 changes the optimizer regime at matched examples",
    r"regression (tie|equivalence)": "opposite seed signs are heterogeneity",
    r"\[EXPECTED-PhaseC": "legacy Phase C is blocked rather than projected",
}

REQUIRED_ARTICLE = [
    "context packaging",
    "Reach",
    "Fidelity",
    "Trust",
    "development evidence",
    "untouched",
    "configuration effect",
    "coverage",
]


def main() -> None:
    failures = []
    for path in CANONICAL:
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
                        or any(word in line_text for word in ("retired", "do not", "does not", "avoid"))):
                    continue
                failures.append(f"{path}:{line}: {match.group(0)!r}: {reason}")
    article = CANONICAL[2].read_text(encoding="utf-8")
    for phrase in REQUIRED_ARTICLE:
        if phrase.lower() not in article.lower():
            failures.append(f"canonical article lacks required concept {phrase!r}")
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
    print(f"RFT CONSISTENCY PASS: {len(CANONICAL)} canonical texts; fixed publication/task strands present")


if __name__ == "__main__":
    main()
