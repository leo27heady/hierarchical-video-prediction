"""Fail-closed design, data, checkpoint, CRN-parity, and budget validator for E7c."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file
from hvp_rev.rhvd import ARMS, arm_cfg, rollout_crn_seed


EXPECTED_ARMS = {
    "hier-min", "flat-min", "hier-long", "flat-budget", "strided-flat", "hier-c2", "flat-c2"
}
EXPECTED_VARIANTS = {(8, 0), (16, 0), (32, 0), (16, 1), (16, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if config.get("status") != "FROZEN" or config.get("study") != "robustness-e7c":
        failures.append("E7c config is not a frozen robustness-e7c study")

    arms = config.get("arms", [])
    arm_names = {row["arm"] for row in arms}
    if arm_names != EXPECTED_ARMS or len(arms) != len(EXPECTED_ARMS):
        failures.append(f"wrong E7c arm set: {sorted(arm_names)}")
    variants = {(int(v["nfe"]), int(v["sampler_seed"])) for v in config.get("variants", [])}
    if variants != EXPECTED_VARIANTS or len(config.get("variants", [])) != len(EXPECTED_VARIANTS):
        failures.append(f"wrong E7c variants: {sorted(variants)}")
    if len(config.get("contrasts", [])) != 4:
        failures.append("E7c requires exactly four registered contrasts")
    for pair in config.get("contrasts", []):
        if pair["candidate"] not in arm_names or pair["baseline"] not in arm_names:
            failures.append(f"contrast {pair['id']} references an absent arm")
        if pair.get("expected_direction") != "positive":
            failures.append(f"contrast {pair['id']} lacks the registered positive direction")

    for row in arms:
        arm = row["arm"]
        if arm not in ARMS or arm_cfg(arm)[2] != "diff":
            failures.append(f"E7c arm is not a registered diffusion arm: {arm}")
            continue
        checkpoint = Path(f"logs_rhcp/rhvd_{arm}_s{row['seed']}_t{row['steps']}.pt")
        if not checkpoint.is_file() or sha256_file(checkpoint) != row["checkpoint_sha256"]:
            failures.append(f"checkpoint missing/hash mismatch: {checkpoint}")

    for artifact in config.get("frozen_artifacts", {}).values():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            failures.append(f"frozen artifact missing/hash mismatch: {path}")

    expected_hours = sum(
        float(row["reference_hours_nfe16"]) * sum(v["nfe"] / 16 for v in config["variants"])
        for row in arms
    )
    budget = config["budget"]
    if abs(expected_hours - float(budget["expected_gpu_hours"])) > 0.02:
        failures.append(f"runtime projection mismatch: {expected_hours:.4f}")
    if float(budget["authorized_max_gpu_hours"]) != 8.0:
        failures.append("E7c hard maximum must remain 8.0 GPU-hours")
    for chunk in range(7):
        for step in range(38):
            if rollout_crn_seed(0, chunk, step) != 100000 + 1000 * chunk + step:
                failures.append("sampler seed zero does not preserve legacy CRN semantics")
                break

    manifest_hash = None
    dataset_id = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(manifest_path)
        dataset_id = manifest.get("dataset_id")
        if manifest.get("purpose") != "robustness_holdout":
            failures.append("dataset purpose is not robustness_holdout")
        for key in ("n", "frames", "world", "digits", "speed", "p_cont", "p_hold"):
            expected_key = "clips" if key == "n" else key
            if manifest["configuration"].get(key) != config["dataset"].get(expected_key):
                failures.append(f"dataset {key} differs from frozen config")
        if manifest["generator"].get("seed") != config["dataset"]["seed"]:
            failures.append("dataset seed differs from frozen config")
        for artifact in manifest.get("artifacts", {}).values():
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                failures.append(f"dataset artifact missing/hash mismatch: {path}")
        e2_manifest = Path("artifacts/confirmation_e2/dataset_manifest_2026-07-12.json")
        if e2_manifest.is_file():
            e2 = json.loads(e2_manifest.read_text(encoding="utf-8"))
            if e2["artifacts"]["clips"]["sha256"] == manifest["artifacts"]["clips"]["sha256"]:
                failures.append("E7c dataset collides with the one-time E2 holdout")

    existing = []
    if dataset_id:
        for spec_path in Path("artifacts/runs/robustness-e7c").rglob("run_spec.json"):
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if spec.get("arm_config", {}).get("dataset_id") == dataset_id:
                existing.append(spec_path.as_posix())
    if existing and not args.allow_existing:
        failures.append(f"pre-existing E7c RunSpecs require explicit resume audit: {existing}")

    report = {
        "schema_version": 1,
        "status": "DESIGN_VALIDATION_PASS" if not failures else "FAIL",
        "config": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest_hash,
        "dataset_id": dataset_id,
        "arms": len(arms),
        "variants": len(variants),
        "evaluations": len(arms) * len(variants),
        "expected_gpu_hours": expected_hours,
        "authorized_gpu_hours": float(budget["authorized_max_gpu_hours"]),
        "failures": failures,
    }
    if args.out:
        atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
