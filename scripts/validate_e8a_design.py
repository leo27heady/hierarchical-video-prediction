"""Fail-closed design, checkpoint, data-isolation and budget validator for E8a."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file
from hvp_rev.rhvd import ARMS, arm_cfg


EXPECTED_ARMS = {
    "hier-min", "flat-min", "hier-long", "flat-budget", "strided-flat", "hier-c2", "flat-c2"
}
EXPECTED_VARIANTS = {
    ("pose-mosaic-k6", 6),
    ("pose-mosaic-k14", 14),
    ("pose-mosaic-k38", 38),
    ("pose-mosaic-k48", 48),
}
EXPECTED_EFFECTS = {
    ("raw-reach-14-to-48", "pose-mosaic-k14", "pose-mosaic-k48"),
    ("raw-reach-38-to-48", "pose-mosaic-k38", "pose-mosaic-k48"),
    ("hier-min-to-raw48", "hier-min", "pose-mosaic-k48"),
    ("hier-long-to-raw48", "hier-long", "pose-mosaic-k48"),
    ("strided-flat-to-raw48", "strided-flat", "pose-mosaic-k48"),
    ("flat-c2-to-raw48", "flat-c2", "pose-mosaic-k48"),
}
EXPECTED_NEURAL_DECISION_EFFECTS = {
    "hier-min-to-raw48", "hier-long-to-raw48", "strided-flat-to-raw48",
    "flat-c2-to-raw48",
}
EXPECTED_RAW_SPAN_DIAGNOSTICS = {
    "raw-reach-14-to-48", "raw-reach-38-to-48",
}
EXPECTED_SECONDARY_POPULATIONS = {
    "answerable-k48": (6, 48),
    "reach-14-to-48": (14, 48),
    "reach-38-to-48": (38, 48),
}


def _other_locked_manifests(exclude: Path) -> list[Path]:
    candidates = set(Path("data").rglob("*.manifest.json"))
    candidates.update(Path("experiments/manifests").glob("*.json"))
    return sorted(path for path in candidates if path.resolve() != exclude.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    protocol_path = Path(args.protocol)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if config.get("status") != "FROZEN" or config.get("study") != "reference-e8a":
        failures.append("E8a config is not a frozen reference-e8a study")
    protocol = protocol_path.read_text(encoding="utf-8")
    for required in (
        "privileged", "not a token-matched", "retrieval_reference_holdout",
        "RAW_ACCESS_CEILING_CI_CLEAN", "No adaptive",
    ):
        if required.lower() not in protocol.lower():
            failures.append(f"E8a protocol lacks required scope marker: {required}")

    variants = {
        (row["id"], int(row["warmup_frames_stored"]))
        for row in config.get("memory", {}).get("variants", [])
    }
    if variants != EXPECTED_VARIANTS or len(variants) != 4:
        failures.append(f"wrong E8a mosaic variants: {sorted(variants)}")
    if config.get("memory", {}).get("update_after_warmup") is not False:
        failures.append("E8a mosaic must remain frozen after warm-up")
    if config.get("memory", {}).get("uses_future_pixels") is not False:
        failures.append("E8a config does not prohibit future-pixel use")
    if config.get("memory", {}).get("budget_matching") is None:
        failures.append("E8a must disclose lack of token-budget matching")
    prior = config["memory"]["unknown_fill"]
    prior_path = Path(prior["training_array"])
    if not prior_path.is_file() or sha256_file(prior_path) != prior["training_array_sha256"]:
        failures.append(f"training-only prior source missing/hash mismatch: {prior_path}")

    neural = config.get("neural_reference_runs", [])
    arms = {row["arm"] for row in neural}
    if arms != EXPECTED_ARMS or len(neural) != len(EXPECTED_ARMS):
        failures.append(f"wrong E8a neural arm set: {sorted(arms)}")
    for row in neural:
        arm = row["arm"]
        if arm not in ARMS or arm_cfg(arm)[2] != "diff":
            failures.append(f"E8a neural reference is not a registered diffusion arm: {arm}")
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != row["checkpoint_sha256"]:
            failures.append(f"checkpoint missing/hash mismatch: {checkpoint}")
    codec = Path(config["codec"]["path"])
    if not codec.is_file() or sha256_file(codec) != config["codec"]["sha256"]:
        failures.append(f"codec missing/hash mismatch: {codec}")

    effects = {
        (row["id"], row["baseline"], row["candidate"])
        for row in config.get("registered_effects", [])
    }
    if effects != EXPECTED_EFFECTS or len(effects) != 6:
        failures.append(f"wrong E8a registered family: {sorted(effects)}")
    decision = config.get("decision", {})
    if set(decision.get("neural_reference_effect_ids", [])) != EXPECTED_NEURAL_DECISION_EFFECTS:
        failures.append("wrong E8a neural-reference decision subset")
    if set(decision.get("raw_span_diagnostic_effect_ids", [])) != EXPECTED_RAW_SPAN_DIAGNOSTICS:
        failures.append("wrong E8a raw-span diagnostic subset")
    evaluation = config["evaluation"]
    expected_geometry = {
        "warmup_frames": 48,
        "rollout_frames": 152,
        "evaluation_clips": 100,
        "nfe": 16,
        "sampler_seed": 0,
        "metric_threshold": 0.5,
    }
    for key, expected in expected_geometry.items():
        if evaluation.get(key) != expected:
            failures.append(f"E8a evaluation {key} differs from frozen value {expected}")
    secondary = evaluation.get("secondary_populations", {})
    if set(secondary) != set(EXPECTED_SECONDARY_POPULATIONS):
        failures.append(f"wrong E8a secondary populations: {sorted(secondary)}")
    else:
        for population_id, (lower, upper) in EXPECTED_SECONDARY_POPULATIONS.items():
            rule = secondary[population_id]
            if (
                rule.get("dt_lower_exclusive"),
                rule.get("dt_upper_inclusive"),
                rule.get("hard_lower_exclusive"),
                rule.get("seen_lower_inclusive"),
            ) != (lower, upper, 0.1, 0.5):
                failures.append(f"wrong E8a population rule: {population_id}")
    expected_hours = sum(float(row["reference_eval_hours_nfe16"]) for row in neural)
    if abs(expected_hours - float(config["budget"]["expected_gpu_hours"])) > 0.02:
        failures.append(f"E8a runtime projection mismatch: {expected_hours:.4f}")
    if float(config["budget"]["authorized_max_gpu_hours"]) != 4.0:
        failures.append("E8a hard maximum must remain 4.0 GPU-hours")

    required_files = (
        "scripts/evaluate_e8a_pose_mosaic.py",
        "scripts/analyze_e8a_pose_mosaic.py",
        "scripts/validate_e8a_design.py",
        "scripts/run_e8a_label_parity.py",
        "scripts/generate_e8a_reference_set.ps1",
        "scripts/run_e8a_reference.ps1",
        "scripts/validate_e8a_completion.py",
        "tests/test_e8a_pose_mosaic.py",
        "docs/confirmation/E8A_TABLE_SHELL.md",
    )
    for file in required_files:
        if not Path(file).is_file():
            failures.append(f"missing E8a implementation file: {file}")
    generator_text = Path("scripts/generate_loopmom_confirmation.py").read_text(encoding="utf-8")
    if "retrieval_reference_holdout" not in generator_text:
        failures.append("LoopMom generator does not support the E8a purpose")
    rhvd_text = Path("hvp_rev/rhvd.py").read_text(encoding="utf-8")
    if "--retrieval_manifest" not in rhvd_text:
        failures.append("neural evaluator lacks a purpose-matched E8a switch")

    manifest_hash = None
    dataset_id = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(manifest_path)
        dataset_id = manifest.get("dataset_id")
        if manifest.get("purpose") != "retrieval_reference_holdout":
            failures.append("E8a dataset purpose is not retrieval_reference_holdout")
        for key in ("n", "frames", "world", "digits", "speed", "p_cont", "p_hold"):
            expected_key = "clips" if key == "n" else key
            if manifest["configuration"].get(key) != config["dataset"].get(expected_key):
                failures.append(f"E8a dataset {key} differs from frozen config")
        if manifest["generator"].get("seed") != config["dataset"]["seed"]:
            failures.append("E8a dataset seed differs from frozen config")
        for artifact in manifest.get("artifacts", {}).values():
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                failures.append(f"E8a dataset artifact missing/hash mismatch: {path}")
        current_hashes = {
            key: value["sha256"] for key, value in manifest.get("artifacts", {}).items()
        }
        for other_path in _other_locked_manifests(manifest_path):
            try:
                other = json.loads(other_path.read_text(encoding="utf-8"))
                other_hashes = {
                    key: value["sha256"]
                    for key, value in other.get("artifacts", {}).items()
                    if isinstance(value, dict) and "sha256" in value
                }
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            for key, digest in current_hashes.items():
                if digest and digest == other_hashes.get(key):
                    failures.append(
                        f"E8a {key} artifact collides with earlier manifest {other_path}"
                    )

    existing = []
    if dataset_id:
        for spec_path in Path("artifacts/runs/reference-e8a").rglob("run_spec.json"):
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if spec.get("arm_config", {}).get("dataset_id") == dataset_id:
                existing.append(spec_path.as_posix())
    if existing and not args.allow_existing:
        failures.append(f"pre-existing E8a RunSpecs require explicit resume audit: {existing}")

    report = {
        "schema_version": 1,
        "status": "DESIGN_VALIDATION_PASS" if not failures else "FAIL",
        "config": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "protocol": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": manifest_hash,
        "dataset_id": dataset_id,
        "mosaic_variants": len(variants),
        "neural_evaluations": len(neural),
        "registered_effects": len(effects),
        "expected_gpu_hours": expected_hours,
        "authorized_gpu_hours": float(config["budget"]["authorized_max_gpu_hours"]),
        "failures": failures,
    }
    if args.out:
        atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
