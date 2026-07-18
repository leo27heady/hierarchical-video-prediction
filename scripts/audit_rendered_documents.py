"""Audit rendered manuscript pages and build original-resolution visual-review strips."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pdfplumber
from PIL import Image, ImageChops


def numeric_key(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def audit(pdf_path: Path, image_dir: Path, output_dir: Path, pages_per_strip: int) -> dict:
    pages = sorted(image_dir.glob("page-*.png"), key=numeric_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"pdf": str(pdf_path), "pages": len(pages), "page_audits": [], "strips": []}
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != len(pages):
            raise ValueError(f"PDF/image page mismatch: {len(pdf.pages)} != {len(pages)}")
        for index, (pdf_page, page_path) in enumerate(zip(pdf.pages, pages), start=1):
            with Image.open(page_path).convert("RGB") as image:
                background = Image.new("RGB", image.size, "white")
                diff = ImageChops.difference(image, background).convert("L")
                bbox = diff.point(lambda value: 255 if value > 10 else 0).getbbox()
                text = pdf_page.extract_text() or ""
                margin = None
                if bbox:
                    margin = {
                        "left": bbox[0],
                        "top": bbox[1],
                        "right": image.width - bbox[2],
                        "bottom": image.height - bbox[3],
                    }
                report["page_audits"].append({
                    "page": index,
                    "image": str(page_path),
                    "size": image.size,
                    "bbox": bbox,
                    "margin_px": margin,
                    "text_chars": len(text),
                    "blank": bbox is None,
                    "edge_touch": bool(margin and min(margin.values()) < 8),
                })

    for start in range(0, len(pages), pages_per_strip):
        batch = pages[start : start + pages_per_strip]
        images = [Image.open(path).convert("RGB") for path in batch]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        strip = Image.new("RGB", (width, height), "white")
        y = 0
        for image in images:
            strip.paste(image, (0, y))
            y += image.height
            image.close()
        strip_path = output_dir / f"pages-{start + 1:03d}-{start + len(batch):03d}.png"
        strip.save(strip_path, optimize=True)
        report["strips"].append(str(strip_path))

    report["blank_pages"] = [
        page["page"] for page in report["page_audits"] if page["blank"]
    ]
    report["edge_touch_pages"] = [
        page["page"] for page in report["page_audits"] if page["edge_touch"]
    ]
    report["low_text_pages"] = [
        page["page"] for page in report["page_audits"] if page["text_chars"] < 40
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pages-per-strip", type=int, default=6)
    args = parser.parse_args()
    report = audit(args.pdf, args.image_dir, args.output_dir, args.pages_per_strip)
    report_path = args.output_dir / "audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pages": report["pages"],
        "blank_pages": report["blank_pages"],
        "edge_touch_pages": report["edge_touch_pages"],
        "low_text_pages": report["low_text_pages"],
        "strips": len(report["strips"]),
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
