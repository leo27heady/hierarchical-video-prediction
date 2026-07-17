"""Development-only reliability ceiling for the E6 persistent-object evaluator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hvp_rev import hier_convae, wb
from hvp_rev.digit_classifier import DigitCNN
from hvp_rev.phasec_v2 import evaluate_prediction_frame, metadata_objects
from hvp_rev.provenance import atomic_write_json, sha256_file


def _means(rows: list[dict]) -> dict:
    return {
        "records": len(rows),
        "coverage": float(np.mean([row["objects"]["coverage"] for row in rows])),
        "identity_accuracy": float(np.nanmean([row["objects"]["identity_accuracy"] for row in rows])),
        "trajectory_error_px": float(np.mean([row["objects"]["trajectory_error"] for row in rows])),
        "missed_objects": float(np.mean([row["objects"]["missed"] for row in rows])),
        "extra_detections": float(np.mean([row["objects"]["extra"] for row in rows])),
        "foreground_mse": float(np.nanmean([
            row["frame"]["mse_gt_fg_nonempty"] for row in rows
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", required=True)
    args = parser.parse_args()

    out, marker = Path(args.out), Path(args.marker)
    if out.exists() or marker.exists():
        raise FileExistsError("evaluator reliability report/marker is immutable")
    import json
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    evaluator = config["evaluation"]["object_evaluator"]
    floors = config["validation"]["evaluator_reliability_floors"]
    clips = np.load(args.development, mmap_mode="r")
    with np.load(args.metadata) as source:
        metadata = {key: source[key] for key in source.files}
    classifier = DigitCNN().cuda().eval()
    classifier.load_state_dict(torch.load(args.classifier, map_location="cuda", weights_only=True))
    ae = hier_convae.load(args.ae)

    rng = np.random.RandomState(config["validation"]["reliability_seed"])
    candidates = np.argwhere(np.ones(clips.shape[:2], dtype=bool))
    rng.shuffle(candidates)
    candidates = candidates[:config["validation"]["reliability_records"]]
    raw_rows, ae_rows = [], []
    for start in range(0, len(candidates), 128):
        batch_index = candidates[start:start + 128]
        true = np.asarray([clips[c, f] for c, f in batch_index], np.float32) / 255.0
        tensor = torch.from_numpy(true)[:, None].cuda()
        with torch.no_grad():
            _, fine = ae.encode(tensor * 2 - 1)
            reconstructed = ((ae.decode_fine(fine).clamp(-1, 1) + 1) * 0.5)[:, 0].cpu().numpy()
        for local, (clip, frame) in enumerate(batch_index):
            gt = metadata_objects(
                object_id=metadata["object_id"], class_id=metadata["class_id"],
                bbox_yxyx=metadata["bbox_yxyx"], clip=int(clip), frame=int(frame),
            )
            kwargs = {
                "ground_truth": gt,
                "classifier": classifier,
                "device": "cuda",
                "detection_threshold": evaluator["detection_threshold"],
                "min_component_pixels": evaluator["min_component_pixels"],
                "confidence_threshold": evaluator["confidence_threshold"],
                "distance_weight": evaluator["distance_weight"],
                "iou_weight": evaluator["iou_weight"],
                "max_assignment_cost": evaluator["max_assignment_cost"],
            }
            raw_rows.append(evaluate_prediction_frame(true[local], true[local], **kwargs))
            ae_rows.append(evaluate_prediction_frame(reconstructed[local], true[local], **kwargs))
    nonoverlap = [
        not bool(metadata["overlap"][clip, frame].any()) for clip, frame in candidates
    ]
    raw_all = _means(raw_rows)
    raw_nonoverlap = _means([row for row, keep in zip(raw_rows, nonoverlap) if keep])
    ae_all = _means(ae_rows)
    ae_nonoverlap = _means([row for row, keep in zip(ae_rows, nonoverlap) if keep])
    checks = {
        "raw_nonoverlap_coverage": raw_nonoverlap["coverage"] >= floors["raw_nonoverlap_coverage"],
        "raw_nonoverlap_identity": (
            raw_nonoverlap["identity_accuracy"] >= floors["raw_nonoverlap_identity_accuracy"]
        ),
        "raw_nonoverlap_trajectory": (
            raw_nonoverlap["trajectory_error_px"] <= floors["raw_nonoverlap_trajectory_px"]
        ),
        "ae_nonoverlap_coverage": ae_nonoverlap["coverage"] >= floors["ae_nonoverlap_coverage"],
        "ae_nonoverlap_identity": (
            ae_nonoverlap["identity_accuracy"] >= floors["ae_nonoverlap_identity_accuracy"]
        ),
    }
    report = {
        "schema_version": 1,
        "status": "EVALUATOR_RELIABILITY_PASS" if all(checks.values()) else "EVALUATOR_RELIABILITY_FAIL",
        "checks": checks,
        "floors": floors,
        "raw_all": raw_all,
        "raw_nonoverlap": raw_nonoverlap,
        "ae_all": ae_all,
        "ae_nonoverlap": ae_nonoverlap,
        "overlap_fraction": float(1 - np.mean(nonoverlap)),
        "sources": {
            "development_sha256": sha256_file(args.development),
            "metadata_sha256": sha256_file(args.metadata),
            "ae_sha256": sha256_file(args.ae),
            "classifier_sha256": sha256_file(args.classifier),
            "config_sha256": sha256_file(args.config),
        },
    }
    atomic_write_json(out, report)
    reliability_id = (
        "phasec-v2-evaluator-reliability-"
        f"{report['sources']['ae_sha256'][:8]}-"
        f"{report['sources']['classifier_sha256'][:8]}"
    )
    wb.init(
        reliability_id,
        config={"report": report, "source_commit": args.source_commit},
        project=args.wandb_project,
        run_id=reliability_id,
        group=args.wandb_group,
        tags=["moving-mnist", "phase-c-v2", "evaluator-reliability"],
        job_type="evaluator-validation",
        required=True,
        resume="never",
    )
    for population, values in (
        ("raw_all", raw_all), ("raw_nonoverlap", raw_nonoverlap),
        ("ae_all", ae_all), ("ae_nonoverlap", ae_nonoverlap),
    ):
        wb.summary({f"{population}/{key}": value for key, value in values.items()})
    wb.artifact(
        reliability_id,
        "evaluator-validation",
        [args.out, args.config],
        aliases=["latest"],
    )
    wb.finish()
    if not all(checks.values()):
        raise SystemExit("Phase C v2 evaluator reliability gate failed")
    atomic_write_json(marker, {
        "schema_version": 1,
        "status": "EVALUATOR_RELIABILITY_PASS",
        "subject": out.as_posix(),
        "subject_sha256": sha256_file(out),
        "source_commit": args.source_commit,
    })
    print(f"PHASEC_V2_EVALUATOR_RELIABILITY_PASS {out}", flush=True)


if __name__ == "__main__":
    main()
