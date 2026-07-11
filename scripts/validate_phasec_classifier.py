"""Validate a seeded classifier checkpoint on prepared NPZ domain fixtures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hvp_rev.classifier_validation import ValidationDomain, validate_classifier
from hvp_rev.digit_classifier import DigitCNN
from hvp_rev.provenance import atomic_write_json, sha256_file


DEFAULT_MINIMUMS = {
    "clean_test": 0.98,
    "translated": 0.95,
    "rendered_component": 0.94,
    "ae_reconstruction": 0.90,
    "generated_degraded": 0.75,
    "overlap": None,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domains", required=True,
                        help="NPZ with <domain>_images and <domain>_labels arrays")
    parser.add_argument("--out", required=True)
    parser.add_argument("--write-marker", action="store_true")
    args = parser.parse_args()
    model = DigitCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    with np.load(args.domains) as data:
        domains = {}
        for name, threshold in DEFAULT_MINIMUMS.items():
            ik, lk = f"{name}_images", f"{name}_labels"
            if ik not in data.files or lk not in data.files:
                raise KeyError(f"classifier validation fixture lacks {ik}/{lk}")
            domains[name] = ValidationDomain(name, data[ik], data[lk], threshold)
        report = validate_classifier(model, domains, device="cpu")
    report.update({
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "domains_sha256": sha256_file(args.domains),
    })
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("classifier validation failed")
    if args.write_marker:
        marker = Path("gates/phase_c_classifier_validated.txt")
        marker.write_text(
            "CLASSIFIER_VALIDATED\n"
            f"checkpoint_sha256={report['checkpoint_sha256']}\n"
            f"report_sha256={sha256_file(args.out)}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
