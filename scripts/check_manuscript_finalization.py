"""Fail-closed checks for the final article/dissertation content contract.

This gate complements numeric evidence checks.  It protects the scientific wording corrected
during the August 2026 red-team pass, verifies that the cross-strand synthesis and curated
appendix remain assembled, and guarantees that figures reproduced from published articles have
not been edited.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REWRITE = ROOT / "docs" / "dissertation" / "rewrite"
ARTICLE = ROOT / "docs" / "article" / "final" / "ARTICLE_REACH_FIDELITY_TRUST.md"
ASSEMBLED = ROOT / "docs" / "dissertation" / "final" / "DISSERTATION_MANUSCRIPT.md"
FREEZE = ROOT / "docs" / "reviews" / "PUBLISHED_FIGURE_FREEZE_2026-08-03.sha256"

SOURCE_TEXTS = sorted(REWRITE.glob("*.md")) + [ARTICLE]
FIGURE_SOURCES = [
    ROOT / "scripts" / "make_dissertation_foundation_figures.py",
    ROOT / "scripts" / "make_rft_result_figures.py",
]

PROHIBITED = {
    r"buffer contents at every rollout step are distributionally identical": (
        "rollout layout/indexing may match training, but self-generated values do not"
    ),
    r"clean-context training (?:is )?sufficient": "exposure bias remains during self-fed rollout",
    r"additive structure is inherently non-contractive": (
        "Lip(I+F) has an upper bound; this wording asserts an invalid lower-bound conclusion"
    ),
    r"CLIP(?:-based)? similarity (?:demonstrates|proves|establishes) causal": (
        "CLIP is a semantic-alignment proxy, not evidence of causal understanding"
    ),
    r"hierarchical consistency (?:is|remains) guaranteed": (
        "conditioning supplies an inductive bias, not a logical guarantee"
    ),
    r"memory does not scale with trajectory length": (
        "only the local activation tape is approximately horizon-independent; replay inputs remain"
    ),
    r"measured here:\s*exact gradients": (
        "reverse replay is validated numerically against BPTT within reported tolerance"
    ),
    r"Source:\s*NEW\s*\(draw\)": "editorial production residue must not appear in the manuscript",
    r"Pending \(no npz yet\)": "terminal evidence tables cannot contain pending configurations",
    r"rewrite,? phase\s*1": "revision history belongs outside the scientific narrative",
    r"matched[- ]NFE": (
        "the mechanism study matches regression/diffusion capacity and declares unequal total work"
    ),
    r"decoder ceiling": "the autoencoder round trip is an empirical reference, not a ceiling",
    r"approximate Pareto equivalence": (
        "Pareto equivalence requires matched runtime as well as memory evidence"
    ),
    r"unidirectional-control principle|unidirectional context transfer": (
        "the executed predictor uses bidirectional joint target attention"
    ),
    r"rejecting a stable trust boundary": (
        "a failed replication fails to establish the boundary; it does not reject every boundary"
    ),
    r"events separated by more than .* are informationally inaccessible": (
        "a flat package limits directly represented history, not all possible causal influence"
    ),
}

REQUIRED_BY_FILE = {
    REWRITE / "01_VSTUP.md": [
        "Contribution and Evidence Crosswalk",
        "Problem-level integration",
        "Evidentiary separation",
        "not one experimentally validated end-to-end model",
    ],
    REWRITE / "03_ROZDIL2.md": [
        "activation-tape",
        "checkpointing",
        "numerical",
    ],
    REWRITE / "04_ROZDIL3.md": [
        "self-generated",
        "exposure bias",
        "foreground column set",
        "answerable",
        "AE round-trip sharpness reference",
    ],
    REWRITE / "05_ROZDIL4.md": [
        "comparison license",
        "configuration",
        "reach–fidelity",
        "Total regression / diffusion evaluations per record",
        "V-BLT entropy allocation [A1]",
        "Consistency criteria [A3]",
    ],
    REWRITE / "06_VYSNOVKY.md": [
        "Retained-state memory and activation-tape memory are distinct resource axes",
        "not one executable pipeline",
    ],
    ARTICLE: [
        "context tokens",
        "untouched",
        "unresolved",
        "2.67",
        "3.43",
        "lower-is-better errors",
        "higher-is-better accuracy or coverage",
    ],
}

REQUIRED_FIGURES = [
    ROOT / "docs" / "figures" / "fig61_dissertation_systems_map.png",
    ROOT / "docs" / "figures" / "fig62_two_resource_synthesis.png",
    ROOT / "docs" / "figures" / "fig63_executed_configuration_matrix.png",
]

REQUIRED_LINKS_BY_FILE = {
    REWRITE / "04_ROZDIL3.md": [
        "../../figures/fig16_context_packaging_architecture.png",
        "../../figures/fig22_token_sequence_layout.png",
        "../../figures/fig23_rollout_ring_buffer.png",
        "../../figures/fig15_loopmom_observability_schematic.png",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_prose(text: str) -> str:
    """Normalize Markdown emphasis and wrapping for resilient phrase checks."""

    return re.sub(r"\s+", " ", text.replace("*", " ")).strip().casefold()


def _check_frozen_figures(failures: list[str]) -> None:
    if not FREEZE.is_file():
        failures.append(f"missing published-figure freeze manifest: {FREEZE.relative_to(ROOT)}")
        return
    for line_number, raw in enumerate(FREEZE.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s{2}(.+)", line)
        if not match:
            failures.append(f"invalid freeze-manifest row {line_number}: {raw!r}")
            continue
        expected, relative = match.groups()
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"frozen published figure is missing: {relative}")
        elif _sha256(path) != expected:
            failures.append(f"published figure changed despite freeze: {relative}")


def main() -> None:
    failures: list[str] = []
    searchable = [path for path in SOURCE_TEXTS + FIGURE_SOURCES if path.is_file()]
    for path in searchable:
        text = path.read_text(encoding="utf-8")
        for pattern, reason in PROHIBITED.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text[: match.start()].count("\n") + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: prohibited wording {match.group(0)!r}: {reason}"
                )

    for path, phrases in REQUIRED_BY_FILE.items():
        if not path.is_file():
            failures.append(f"missing authoritative manuscript source: {path.relative_to(ROOT)}")
            continue
        text = _normalized_prose(path.read_text(encoding="utf-8"))
        for phrase in phrases:
            if _normalized_prose(phrase) not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} lacks required finalization concept {phrase!r}"
                )

    for path, targets in REQUIRED_LINKS_BY_FILE.items():
        if not path.is_file():
            failures.append(f"missing authoritative manuscript source: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for target in targets:
            if f"]({target})" not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} does not bind required figure {target!r}"
                )

    # Every numbered dissertation figure caption must be paired with a Markdown image. This
    # catches the prior failure mode in which prose captions survived assembly but four Chapter 3
    # schematics were never embedded.
    for path in sorted(REWRITE.glob("0[1-6]_*.md")):
        text = path.read_text(encoding="utf-8")
        image_count = len(re.findall(r"^!\[[^\]]*\]\([^)]+\)\s*$", text, flags=re.MULTILINE))
        caption_count = len(
            re.findall(r"^\*\*Figure\s+(?:I\.|[1-9])", text, flags=re.MULTILINE)
        )
        if image_count != caption_count:
            failures.append(
                f"{path.relative_to(ROOT)} has {image_count} image links but "
                f"{caption_count} numbered figure captions"
            )

    # Duplicate local equation tags create silent cross-reference ambiguity in the rendered
    # manuscript even when every individual equation parses successfully.
    for path in sorted(REWRITE.glob("0[1-6]_*.md")):
        tags = re.findall(r"\\tag\{([^}]+)\}", path.read_text(encoding="utf-8"))
        duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
        if duplicates:
            failures.append(
                f"{path.relative_to(ROOT)} has duplicate equation tags: {duplicates}"
            )

    appendices = sorted((REWRITE / "appendices").glob("APPENDIX_*.md"))
    if not appendices:
        failures.append("no curated APPENDIX_*.md file is bound for submission")

    for path in REQUIRED_FIGURES:
        if not path.is_file():
            failures.append(f"missing final synthesis figure: {path.relative_to(ROOT)}")

    if not ASSEMBLED.is_file():
        failures.append("assembled dissertation Markdown is missing")
    else:
        assembled = ASSEMBLED.read_text(encoding="utf-8")
        for appendix in appendices:
            heading = next(
                (line for line in appendix.read_text(encoding="utf-8").splitlines() if line.startswith("#")),
                "",
            )
            if heading and heading not in assembled:
                failures.append(f"curated appendix is not assembled: {appendix.relative_to(ROOT)}")

    _check_frozen_figures(failures)

    if failures:
        print("MANUSCRIPT FINALIZATION CHECK FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(
        "MANUSCRIPT FINALIZATION CHECK PASS: scientific wording, integration spine, curated "
        f"appendix, {len(REQUIRED_FIGURES)} synthesis figures, and published-figure freeze verified"
    )


if __name__ == "__main__":
    main()
