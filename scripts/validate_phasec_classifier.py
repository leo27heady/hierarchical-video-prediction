"""Validate a seeded classifier checkpoint on prepared NPZ domain fixtures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hvp_rev import wb
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
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="")
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
        "schema_version": 1,
        "status": "CLASSIFIER_VALIDATION_PASS" if report["passed"] else "CLASSIFIER_VALIDATION_FAIL",
        "confidence_threshold": 0.0,
        "abstention_policy": "disabled; every matched detection receives a class prediction",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "domains_sha256": sha256_file(args.domains),
    })
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("classifier validation failed")
    if args.wandb_project or args.wandb_group:
        if not args.wandb_project or not args.wandb_group or not args.source_commit:
            raise ValueError("W&B validation requires project, group and source commit")
        run_id = f"phasec-v2-classifier-validation-{report['checkpoint_sha256'][:12]}"
        wb.init(
            run_id,
            config={"report": report, "source_commit": args.source_commit},
            project=args.wandb_project,
            run_id=run_id,
            group=args.wandb_group,
            tags=["moving-mnist", "phase-c-v2", "classifier-validation"],
            job_type="classifier-validation",
            required=True,
            resume="never",
        )
        for name, values in report["domains"].items():
            wb.summary({
                f"domains/{name}/accuracy": values["accuracy"],
                f"domains/{name}/ece": values["ece_10bin"],
                f"domains/{name}/n": values["n"],
            })
        wb.artifact(
            f"phasec-v2-classifier-validation-{report['checkpoint_sha256'][:12]}",
            "classifier-validation",
            [args.checkpoint, args.domains, args.out],
            aliases=["latest"],
        )
        wb.finish()
    if args.write_marker:
        if not args.source_commit:
            raise ValueError("--write-marker requires --source-commit")
        marker = Path("gates/phase_c_classifier_validated.json")
        atomic_write_json(marker, {
            "schema_version": 1,
            "status": "CLASSIFIER_VALIDATION_PASS",
            "subject": Path(args.out).as_posix(),
            "subject_sha256": sha256_file(args.out),
            "source_commit": args.source_commit,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "domains_sha256": report["domains_sha256"],
        })


if __name__ == "__main__":
    main()
