"""Run structural and scope checks for the restructured dissertation sources."""

from __future__ import annotations

import json
import hashlib
import re
import sys
from pathlib import Path

import fitz
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "docs" / "dissertation" / "restructured_english"
FILES = {
    "front": WORK / "00_FRONT_MATTER.md",
    "intro": WORK / "01_INTRODUCTION.md",
    "ch1": WORK / "02_CHAPTER_1.md",
    "ch2": WORK / "03_CHAPTER_2.md",
    "ch3": WORK / "04_CHAPTER_3.md",
    "ch4": WORK / "05_CHAPTER_4.md",
    "conclusions": WORK / "06_CONCLUSIONS.md",
    "references": WORK / "07_REFERENCES.md",
    "appendices": WORK / "08_APPENDICES.md",
}

WORD_TARGETS = {
    "intro": (2800, 3600),
    "ch1": (9700, 12750),
    "ch2": (13200, 16850),
    "ch3": (11950, 15500),
    "ch4": (10650, 13650),
    "conclusions": (1500, 2200),
}


def read(key: str) -> str:
    return FILES[key].read_text(encoding="utf-8")


def visible(text: str) -> str:
    return re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'’\-]+\b", visible(text), flags=re.UNICODE))


def image_targets(text: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)


def caption_image_pairs(text: str, chapter_number: int) -> list[tuple[str, list[str], str]]:
    """Return caption id, preceding image links, and caption paragraph."""
    pairs: list[tuple[str, list[str], str]] = []
    pending: list[str] = []
    lines = text.splitlines()
    index = 0
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    caption_pattern = re.compile(rf"^\*\*Figure\s+({chapter_number}\.\d+)\b")
    while index < len(lines):
        image_match = image_pattern.search(lines[index])
        if image_match:
            pending.append(image_match.group(1))
        caption_match = caption_pattern.match(lines[index])
        if caption_match:
            paragraph = [lines[index]]
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].strip():
                paragraph.append(lines[cursor])
                cursor += 1
            pairs.append((caption_match.group(1), list(pending), "\n".join(paragraph)))
            pending.clear()
            index = cursor
            continue
        index += 1
    return pairs


def expand_citation_group(group: str) -> set[int]:
    numbers: set[int] = set()
    for start, end in re.findall(r"(\d+)(?:\s*[\-–]\s*(\d+))?", group):
        lo = int(start)
        hi = int(end) if end else lo
        numbers.update(range(min(lo, hi), max(lo, hi) + 1))
    return numbers


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    texts = {key: read(key) for key in FILES}

    annotation_match = re.search(
        r"## English annotation\s+(.*?)\s+## Applicant publications",
        texts["front"],
        flags=re.DOTALL,
    )
    if not annotation_match:
        errors.append("English annotation boundaries were not found")
        annotation = ""
    else:
        annotation = annotation_match.group(1)
        count = words(annotation)
        if not 1000 <= count <= 1400:
            errors.append(f"English annotation word count {count} is outside 1000-1400")

    for key, (minimum, maximum) in WORD_TARGETS.items():
        count = words(texts[key])
        if not minimum <= count <= maximum:
            errors.append(f"{key} word count {count} is outside {minimum}-{maximum}")

    for key, text in {"annotation": annotation, "intro": texts["intro"]}.items():
        if image_targets(text):
            errors.append(f"{key} contains an image")
        if re.search(r"(?m)^\s*\|.+\|\s*$", text):
            errors.append(f"{key} contains a Markdown table")
        if "$$" in text:
            errors.append(f"{key} contains displayed mathematics")

    main_chapters = "\n".join(texts[key] for key in ("ch1", "ch2", "ch3", "ch4"))
    figure_numbers: dict[str, list[int]] = {}
    for chapter_number, key in enumerate(("ch1", "ch2", "ch3", "ch4"), start=1):
        numbers = [
            int(number)
            for number in re.findall(
                rf"(?m)^\*\*Figure\s+{chapter_number}\.(\d+)\b", texts[key]
            )
        ]
        figure_numbers[key] = numbers
        if numbers != list(range(1, len(numbers) + 1)):
            errors.append(f"{key} figure captions are not sequential: {numbers}")
    total_figures = sum(len(numbers) for numbers in figure_numbers.values())
    if total_figures < 73:
        errors.append(f"main chapter image count {total_figures} is below the 73-figure minimum")

    for key in ("ch1", "ch2", "ch3", "ch4"):
        if texts[key].count("$$") % 2:
            errors.append(f"{key} has unbalanced display-math delimiters")

    all_review_text = "\n".join(texts.values())
    for marker in ("[REF:", "NEW-FIGURE", "TODO", "TBD"):
        if marker.lower() in all_review_text.lower():
            errors.append(f"unresolved marker remains: {marker}")

    body = "\n".join(texts[key] for key in ("intro", "ch1", "ch2", "ch3", "ch4", "conclusions"))
    for phrase in ("the first article", "the second article", "the third article", "the fourth article"):
        if phrase in body.lower():
            errors.append(f"publication chronology phrase remains in scientific body: {phrase}")

    required_phrases = {
        "unresolved frontier": "1,792",
        "lower hierarchy budget": "1,152",
        "upper flat budget": "2,432",
        "fixed patch boundary": "fixed",
        "raster byte wording": "raster pixel-channel bytes",
        "dynamic allocation boundary": "not implemented",
    }
    for label, phrase in required_phrases.items():
        if phrase.lower() not in body.lower():
            errors.append(f"required scope phrase absent ({label}): {phrase}")

    explicit_exclusions = (
        "final predictor does not contain tempverseformer",
        "final predictor does not use tempverseformer",
        "reversible temporal computation is not installed inside the model described here",
        "hierarchical predictor contains neither tempverseformer",
    )
    if not any(phrase in body.lower() for phrase in explicit_exclusions):
        warnings.append("exact final-predictor/TempVerseFormer exclusion phrase was not found")

    for key in ("ch1", "ch2", "ch3", "ch4"):
        for target in image_targets(texts[key]):
            if re.match(r"^[a-z]+://", target, flags=re.IGNORECASE):
                continue
            resolved = (FILES[key].parent / target).resolve()
            if not resolved.is_file():
                errors.append(f"broken image link in {key}: {target}")

    ref_numbers = {
        int(n) for n in re.findall(r"(?m)^(\d+)\.\s", texts["references"])
    }
    if not 170 <= len(ref_numbers) <= 230:
        errors.append(f"reference count {len(ref_numbers)} is outside 170-230")
    cited: set[int] = set()
    for group in re.findall(r"\[([0-9,;\s\-–]+)\]", body):
        cited.update(expand_citation_group(group))
    # Bracketed numeric domains such as [0, 1] occur in mathematical prose and are
    # not bibliography citations; the reference list is one-indexed.
    cited.discard(0)
    undefined = sorted(cited - ref_numbers)
    uncited = sorted(ref_numbers - cited)
    if undefined:
        errors.append(f"undefined citations: {undefined}")
    if uncited:
        warnings.append(f"bibliography entries not cited in body: {uncited}")

    forbidden_mojibake = ("\u00e2\u20ac", "\u00c3", "\u00c2", "\ufffd")
    for key, text in texts.items():
        if any(token in text for token in forbidden_mojibake):
            errors.append(f"possible mojibake remains in {key}")

    provenance_paths = (
        ROOT / "docs" / "figures" / "restructured" / "figure_provenance.json",
        ROOT / "docs" / "figures" / "restructured" / "ch2_ch4_figure_provenance.json",
        ROOT / "docs" / "figures" / "restructured" / "existing_figure_provenance.json",
    )
    provenance_targets: dict[Path, tuple[dict, str]] = {}
    permission_open: list[str] = []
    required_common = {
        "schema_version",
        "figure_id",
        "file",
        "asset_class",
        "sha256",
        "bytes",
        "modification_status",
        "rights_status",
        "permission_status",
        "role",
        "caption_attribution",
    }
    for provenance_path in provenance_paths:
        if not provenance_path.is_file():
            errors.append(f"figure provenance ledger is missing: {provenance_path.name}")
            continue
        try:
            entries = json.loads(provenance_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(
                f"figure provenance ledger is invalid JSON ({provenance_path.name}): {exc}"
            )
            continue
        if not isinstance(entries, list):
            errors.append(f"figure provenance ledger is not a list: {provenance_path.name}")
            continue
        for entry in entries:
            missing = sorted(key for key in required_common if entry.get(key) in (None, ""))
            # Native figures legitimately have no source page; it is not a common required field.
            if missing:
                errors.append(
                    f"incomplete provenance entry in {provenance_path.name} "
                    f"({entry.get('file', '<unknown>')}): missing {missing}"
                )
                continue
            target = entry["file"]
            expected_hash = entry["sha256"]
            resolved = (ROOT / target).resolve()
            if resolved in provenance_targets:
                prior = provenance_targets[resolved][1]
                errors.append(
                    f"duplicate provenance target in {prior} and {provenance_path.name}: {target}"
                )
                continue
            provenance_targets[resolved] = (entry, provenance_path.name)
            if not resolved.is_file():
                errors.append(f"provenance target is missing: {target}")
                continue
            if resolved.stat().st_size != entry["bytes"]:
                errors.append(f"provenance byte-size mismatch: {target}")
            observed_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if observed_hash != expected_hash:
                errors.append(f"provenance hash mismatch: {target}")

            suffix = resolved.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
                expected_dimensions = entry.get("pixel_dimensions")
                if not (
                    isinstance(expected_dimensions, list)
                    and len(expected_dimensions) == 2
                    and all(isinstance(value, int) and value > 0 for value in expected_dimensions)
                ):
                    errors.append(f"missing/invalid pixel dimensions: {target}")
                else:
                    with Image.open(resolved) as image:
                        observed_dimensions = [int(image.width), int(image.height)]
                        if observed_dimensions != expected_dimensions:
                            errors.append(f"pixel-dimension mismatch: {target}")
                        if entry["asset_class"].startswith("author-generated"):
                            rgb = image.convert("RGB")
                            corners = [
                                rgb.getpixel((0, 0)),
                                rgb.getpixel((rgb.width - 1, 0)),
                                rgb.getpixel((0, rgb.height - 1)),
                                rgb.getpixel((rgb.width - 1, rgb.height - 1)),
                            ]
                            if any(min(pixel) < 245 for pixel in corners):
                                errors.append(f"author-generated raster lacks white corners: {target}")
            elif suffix == ".pdf":
                expected_dimensions = entry.get("page_dimensions_points")
                if not (
                    isinstance(expected_dimensions, list)
                    and len(expected_dimensions) == 2
                    and all(isinstance(value, (int, float)) and value > 0 for value in expected_dimensions)
                ):
                    errors.append(f"missing/invalid PDF page dimensions: {target}")
                else:
                    reader = PdfReader(resolved)
                    observed_dimensions = [
                        round(float(reader.pages[0].mediabox.width), 3),
                        round(float(reader.pages[0].mediabox.height), 3),
                    ]
                    if any(abs(a - b) > 0.01 for a, b in zip(observed_dimensions, expected_dimensions)):
                        errors.append(f"PDF page-dimension mismatch: {target}")
            else:
                errors.append(f"unsupported provenance media type: {target}")

            asset_class = entry["asset_class"]
            if "open:" in entry["permission_status"].lower():
                permission_open.append(target)
            if asset_class == "published-reused-unchanged":
                published_required = ("source_paper", "original_figure", "source_page", "extraction_method")
                absent = [key for key in published_required if entry.get(key) in (None, "")]
                if absent:
                    errors.append(f"published provenance lacks {absent}: {target}")
                status = entry["modification_status"].lower()
                if "unchanged" not in status and "no crop" not in status and "without pixel" not in status:
                    errors.append(f"published provenance does not assert unchanged pixels: {target}")
                # Fail closed against the supplied publication/source asset, not only the ledger hash.
                source_reference = entry.get("source_asset") or entry.get("data_source")
                if source_reference:
                    source_text = str(source_reference)
                    source_file_text = source_text.split("#page=", 1)[0]
                    source_file = Path(source_file_text)
                    if source_file.is_file():
                        if "#page=" in source_text and ";xref=" in source_text:
                            xref = int(source_text.rsplit(";xref=", 1)[1])
                            document = fitz.open(source_file)
                            source_bytes = document.extract_image(xref)["image"]
                            document.close()
                            if source_bytes != resolved.read_bytes():
                                errors.append(f"published PDF extraction is no longer byte-identical: {target}")
                        elif entry.get("source_image"):
                            page_index = int(entry["source_page"]) - 1
                            matches = [
                                image for image in PdfReader(source_file).pages[page_index].images
                                if image.name == entry["source_image"]
                            ]
                            if len(matches) != 1 or matches[0].data != resolved.read_bytes():
                                errors.append(f"published pypdf extraction is no longer byte-identical: {target}")
                        elif source_file.suffix.lower() == resolved.suffix.lower():
                            if source_file.read_bytes() != resolved.read_bytes():
                                errors.append(f"published source copy is no longer byte-identical: {target}")
                    else:
                        warnings.append(f"published source asset unavailable for byte-parity check: {source_file}")
            elif asset_class.startswith("author-generated"):
                native_required = ("generator", "generator_sha256", "data_or_concept_source")
                absent = [key for key in native_required if entry.get(key) in (None, "")]
                if absent:
                    errors.append(f"author-generated provenance lacks {absent}: {target}")
                else:
                    generator = (ROOT / entry["generator"]).resolve()
                    if not generator.is_file():
                        errors.append(f"figure generator is missing: {entry['generator']}")
                    elif hashlib.sha256(generator.read_bytes()).hexdigest() != entry["generator_sha256"]:
                        errors.append(f"figure generator hash mismatch: {entry['generator']}")
            else:
                errors.append(f"unknown provenance asset class {asset_class!r}: {target}")

    for key in ("ch1", "ch2", "ch3", "ch4"):
        for target in image_targets(texts[key]):
            if re.match(r"^[a-z]+://", target, flags=re.IGNORECASE):
                continue
            resolved = (FILES[key].parent / target).resolve()
            if resolved not in provenance_targets:
                errors.append(f"image lacks a provenance entry in {key}: {target}")

    # Bind each live image to the numbered caption immediately following it and
    # require the caption to disclose whether the asset is published, derived,
    # implementation-informed, or conceptual.
    linked_resolved: set[Path] = set()
    for chapter_number, key in enumerate(("ch1", "ch2", "ch3", "ch4"), start=1):
        for caption_id, targets, caption_block in caption_image_pairs(texts[key], chapter_number):
            if not targets:
                errors.append(f"caption {caption_id} has no preceding image")
            if "source" not in caption_block.lower():
                errors.append(f"caption {caption_id} lacks an explicit source/status note")
            for target in targets:
                resolved = (FILES[key].parent / target).resolve()
                linked_resolved.add(resolved)
                provenance = provenance_targets.get(resolved)
                if provenance is None:
                    continue
                provenance_id = str(provenance[0]["figure_id"])
                base_match = re.match(r"\d+\.\d+", provenance_id)
                if not base_match or base_match.group(0) != caption_id:
                    errors.append(
                        f"caption/provenance figure-id mismatch: caption {caption_id}, "
                        f"ledger {provenance_id}, file {target}"
                    )

    live_images = {
        (FILES[key].parent / target).resolve()
        for key in ("ch1", "ch2", "ch3", "ch4")
        for target in image_targets(texts[key])
        if not re.match(r"^[a-z]+://", target, flags=re.IGNORECASE)
    }
    if linked_resolved != live_images:
        missing_pairs = sorted(str(path) for path in live_images - linked_resolved)
        errors.append(f"live images not bound to a numbered caption: {missing_pairs}")

    if permission_open:
        warnings.append(
            f"published/source permission remains open for {len(permission_open)} assets: "
            + ", ".join(permission_open)
        )

    filler_patterns = {
        "does not": r"\bdoes not\b",
        "not a": r"\bnot a\b",
        "rather than": r"\brather than\b",
        "it is important to note": r"\bit is important to note\b",
        "the next section": r"\bthe next section\b",
    }
    filler_counts = {
        label: len(re.findall(pattern, body, flags=re.IGNORECASE))
        for label, pattern in filler_patterns.items()
    }

    author_inputs = len(re.findall(r"\[AUTHOR INPUT:", all_review_text))
    print(f"main_chapter_images={total_figures}")
    print(
        "figures_by_chapter="
        + ", ".join(f"{key}:{len(numbers)}" for key, numbers in figure_numbers.items())
    )
    print(f"references={len(ref_numbers)} cited={len(cited)} uncited={len(uncited)}")
    print(f"author_inputs={author_inputs}")
    print("prose_repetition=" + ", ".join(f"{key}:{value}" for key, value in filler_counts.items()))
    for key in ("front", "intro", "ch1", "ch2", "ch3", "ch4", "conclusions"):
        print(f"words[{key}]={words(texts[key])}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
