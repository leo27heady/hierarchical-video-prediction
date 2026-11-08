"""Fail-closed prelaunch design/data/parity validator for E4b."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import arm_registry, rhvd
from hvp_rev.provenance import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/regression_factorial_seeds3_4.json")
    parser.add_argument("--out", default="gates/e4b_design_validation.json")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    failures = []
    if config.get("status") != "FROZEN":
        failures.append("config is not frozen")
    runs = config.get("new_training_runs", [])
    arms = {"hier-long", "strided-flat", "reg-long", "strided-flat-reg"}
    expected = {(arm, seed) for seed in (3, 4) for arm in arms}
    actual = {(run.get("arm"), int(run.get("seed", -1))) for run in runs}
    if len(runs) != 8 or actual != expected:
        failures.append(f"wrong E4b cells: {sorted(actual)}")
    allocated = sum(float(run.get("max_gpu_hours", 0)) for run in runs)
    maximum = float(config.get("budget", {}).get("authorized_max_gpu_hours", 0))
    if allocated != 36.0 or maximum != 36.0:
        failures.append(f"E4b must allocate exactly 36 replacement hours: {allocated}/{maximum}")
    if config.get("budget", {}).get("protected_reserve_allowed") is not False:
        failures.append("protected reserve is not explicitly prohibited")
    if "E8c" not in config.get("budget", {}).get("replaces", ""):
        failures.append("E4b does not explicitly replace E8c")

    registry = arm_registry.from_legacy_mapping(rhvd.ARMS)
    resolved = {}
    for run in runs:
        arm = run["arm"]
        spec = registry[arm]
        resolved[arm] = spec.resolved()
        expected_head = "reg" if run["head"] == "regression" else "diff"
        if spec.head != expected_head:
            failures.append(f"{arm} head mismatch")
        if spec.context_reach != 48:
            failures.append(f"{arm} reach changed: {spec.context_reach}")
        if run.get("steps") != 30000 or run.get("batch") != 16:
            failures.append(f"{arm}/s{run['seed']} is not legacy B16x30k")
        if run.get("lr") != 0.0003 or run.get("warmup") != 2000 or run.get("ema") != 0.999:
            failures.append(f"{arm}/s{run['seed']} optimizer/EMA regime changed")
        if run.get("precision") != "bf16" or run.get("max_gpu_hours") != 4.5:
            failures.append(f"{arm}/s{run['seed']} precision/cell budget changed")
    token_expectations = {
        "hier-long": 1152, "strided-flat": 1088,
        "reg-long": 1152, "strided-flat-reg": 1088,
    }
    for arm, tokens in token_expectations.items():
        if resolved.get(arm, {}).get("context_tokens") != tokens:
            failures.append(f"{arm} context tokens changed")

    e4a = json.loads(Path("artifacts/e4_head_package/results.json").read_text(encoding="utf-8"))
    e4a_validation = json.loads(
        Path("artifacts/e4_head_package/completion_validation.json").read_text(encoding="utf-8")
    )
    if not e4a.get("e4b_trigger", {}).get("fired"):
        failures.append("E4a did not fire the E4b trigger")
    if e4a_validation.get("status") != "PASS":
        failures.append("E4a completion validator is not PASS")

    h3b = json.loads(Path("gates/h3b_parity_report.json").read_text(encoding="utf-8"))
    if h3b.get("status") != "H3B_PARITY_PASS":
        failures.append("H3b parity report is not PASS")
    parity_hashes = h3b.get("source_hashes", {})
    for path, expected_hash in parity_hashes.items():
        if not Path(path).is_file() or sha256_file(path) != expected_hash:
            failures.append(f"H3b-parity source changed: {path}")

    required = [
        "data/loopmom_train_3000x100.npy",
        "data/loopmom_train_3000x100_actions.npy",
        "data/loopmom_test_500x200.npy",
        "data/loopmom_test_500x200_actions.npy",
        "logs_rhcp/hier_ae_mom.pt",
        "scripts/analyze_e4b_factorial.py",
        "scripts/run_e4b_factorial.ps1",
        "scripts/validate_e4b_completion.py",
        "docs/confirmation/E4B_TABLE_SHELL.md",
    ]
    for path in required:
        if not Path(path).is_file():
            failures.append(f"missing required artifact: {path}")

    existing = []
    run_root = Path("artifacts/training") / config["study"]
    if run_root.exists():
        existing = [path.as_posix() for path in run_root.rglob("run_spec.json")]
    if existing and not args.allow_existing:
        failures.append(f"pre-existing E4b RunSpecs require explicit resume audit: {existing}")

    report = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "config_sha256": sha256_file(args.config),
        "runs": len(runs),
        "allocated_gpu_hours": allocated,
        "authorized_gpu_hours": maximum,
        "replacement": config.get("budget", {}).get("replaces"),
        "e4a_results_sha256": sha256_file("artifacts/e4_head_package/results.json"),
        "e4a_completion_sha256": sha256_file("artifacts/e4_head_package/completion_validation.json"),
        "h3b_parity_sha256": sha256_file("gates/h3b_parity_report.json"),
        "resolved_arms": resolved,
        "required_hashes": {path: sha256_file(path) for path in required if Path(path).is_file()},
        "existing_runs": existing,
        "failures": failures,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
