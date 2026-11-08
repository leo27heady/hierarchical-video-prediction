"""Fail-closed E7a protocol, artifact, data-isolation and budget validator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import arm_registry, rhvd
from hvp_rev.provenance import atomic_write_json, sha256_file


EXPECTED_ARMS = {
    "hier-min", "flat-min", "hier-long", "flat-budget", "strided-flat", "hier-c2", "flat-c2"
}


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
    if config.get("status") != "FROZEN" or config.get("study") != "robustness-e7a":
        failures.append("E7a config is not a frozen robustness-e7a study")
    runs = config.get("predictor_runs", [])
    arms = {row.get("arm") for row in runs}
    if arms != EXPECTED_ARMS or len(runs) != len(EXPECTED_ARMS):
        failures.append(f"wrong E7a predictor matrix: {sorted(arms)}")
    registry = arm_registry.from_legacy_mapping(rhvd.ARMS)
    for row in runs:
        arm = row["arm"]
        if arm not in registry or registry[arm].head != "diff":
            failures.append(f"E7a arm is not a registered diffusion arm: {arm}")
        if (row.get("seed"), row.get("steps"), row.get("batch"), row.get("lr"),
                row.get("warmup"), row.get("precision")) != (0, 30000, 16, 0.0003, 2000, "bf16"):
            failures.append(f"{arm} changes the frozen predictor regime")
    contrasts = config.get("contrasts", [])
    if len(contrasts) != 4:
        failures.append("E7a requires exactly four registered contrasts")
    for contrast in contrasts:
        if contrast.get("candidate") not in arms or contrast.get("baseline") not in arms:
            failures.append(f"contrast references missing arm: {contrast}")
        if contrast.get("expected_direction") != "positive":
            failures.append(f"contrast lacks registered positive direction: {contrast.get('id')}")

    codecs = config.get("codecs", {})
    alt = codecs.get("alternate", {})
    if (alt.get("seed"), alt.get("steps"), alt.get("batch"), alt.get("lr")) != (1, 7000, 64, 0.0003):
        failures.append("alternate codec changes more than the registered seed")
    original = codecs.get("original", {})
    original_path = Path(original.get("path", ""))
    if not original_path.is_file() or sha256_file(original_path) != original.get("sha256"):
        failures.append("original codec missing or hash-mismatched")
    original_ckpts = config.get("original_predictor_checkpoints", [])
    if {row.get("arm") for row in original_ckpts} != EXPECTED_ARMS or len(original_ckpts) != 7:
        failures.append("original predictor checkpoint inventory differs from seven arms")
    for row in original_ckpts:
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            failures.append(f"original predictor checkpoint missing/hash mismatch: {path}")
    for artifact in config.get("frozen_artifacts", {}).values():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            failures.append(f"frozen artifact missing/hash mismatch: {path}")

    budget = config.get("budget", {})
    predictor_allocated = sum(float(row.get("max_gpu_hours", 0)) for row in runs)
    components = (float(budget.get("codec_training_max_gpu_hours", 0)) + predictor_allocated +
                  float(budget.get("evaluation_allocated_gpu_hours", 0)))
    maximum = float(budget.get("authorized_max_gpu_hours", 0))
    if abs(predictor_allocated - float(budget.get("predictor_training_allocated_gpu_hours", -1))) > 1e-9:
        failures.append("predictor allocation sum differs from frozen budget")
    if components > maximum or maximum != 36.0:
        failures.append(f"invalid E7a budget components: {components:.3f} / {maximum:.3f}")
    if int(config["evaluation"].get("nfe", -1)) != 16 or int(config["evaluation"].get("sampler_seed", -1)) != 0:
        failures.append("E7a evaluation sampler differs from registered NFE16/seed0")

    required = [
        "hvp_rev/hier_convae.py", "hvp_rev/hier_convae_v2.py", "hvp_rev/rhvd_train_v2.py",
        "hvp_rev/rhvd.py", "scripts/evaluate_e7a_codec_floor.py",
        "scripts/validate_loopmom_confirmation.py",
        "scripts/analyze_e7a_codec_robustness.py", "scripts/validate_e7a_completion.py",
        "scripts/run_e7a_codec_robustness.ps1",
        "docs/E7A_ALTERNATE_CODEC_ROBUSTNESS_PROTOCOL_2026-07-16.md",
        "docs/confirmation/E7A_TABLE_SHELL.md",
    ]
    failures.extend(f"missing E7a implementation file: {path}" for path in required if not Path(path).is_file())

    manifest_hash = None
    dataset_id = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(manifest_path)
        dataset_id = manifest.get("dataset_id")
        if manifest.get("purpose") != "codec_robustness_holdout":
            failures.append("E7a dataset purpose is not codec_robustness_holdout")
        expected = config["dataset"]
        for key in ("frames", "world", "digits", "speed", "p_cont", "p_hold"):
            if manifest["configuration"].get(key) != expected.get(key):
                failures.append(f"E7a dataset {key} differs from frozen design")
        if manifest["configuration"].get("n") != expected.get("clips"):
            failures.append("E7a dataset clip count differs from frozen design")
        if manifest["generator"].get("seed") != expected.get("seed"):
            failures.append("E7a dataset seed differs from frozen design")
        for artifact in manifest.get("artifacts", {}).values():
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                failures.append(f"E7a dataset artifact missing/hash mismatch: {path}")
        clip_hash = manifest.get("artifacts", {}).get("clips", {}).get("sha256")
        comparison_manifests = [
            Path("data/loopmom_test_500x200.manifest.json"),
            Path("artifacts/confirmation_e2/dataset_manifest_2026-07-12.json"),
            Path("experiments/manifests/loopmom_e7c_robustness_2026-07-15.json"),
        ]
        comparison_hashes = {config["frozen_artifacts"]["development_test"]["sha256"]}
        for other_path in comparison_manifests:
            if other_path.is_file():
                other = json.loads(other_path.read_text(encoding="utf-8"))
                if "artifacts" in other and "clips" in other["artifacts"]:
                    comparison_hashes.add(other["artifacts"]["clips"]["sha256"])
        if clip_hash in comparison_hashes:
            failures.append("E7a dataset collides with development, E2, or E7c")

    existing = []
    for base in (Path("artifacts/autoencoders/e7a-alternate-codec"),
                 Path("artifacts/training/e7a-alternate-codec-predictors"),
                 Path("artifacts/runs/robustness-e7a")):
        if base.exists():
            existing.extend(path.as_posix() for path in base.rglob("COMPLETE"))
    if existing and not args.allow_existing:
        failures.append(f"pre-existing E7a artifacts require explicit resume audit: {existing}")

    report = {
        "schema_version": 1,
        "status": "DESIGN_VALIDATION_PASS" if not failures else "FAIL",
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest_hash,
        "dataset_id": dataset_id,
        "predictor_runs": len(runs),
        "evaluations": 14,
        "budget_components_gpu_hours": components,
        "authorized_gpu_hours": maximum,
        "failures": failures,
    }
    if args.out:
        atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
