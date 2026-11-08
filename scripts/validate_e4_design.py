"""Fail-closed design validator for E4a before GPU authorization."""
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
    parser.add_argument("--config", default="experiments/configs/regression_factorial_seed2.json")
    parser.add_argument("--out", default="gates/e4_design_validation.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    failures = []
    if config.get("status") != "FROZEN":
        failures.append("config is not frozen")
    runs = config.get("new_training_runs", [])
    expected = {("reg-long", 2), ("strided-flat-reg", 2)}
    actual = {(run.get("arm"), int(run.get("seed", -1))) for run in runs}
    if actual != expected or len(runs) != 2:
        failures.append(f"wrong new training cells: {sorted(actual)}")
    allocated = sum(float(run.get("max_gpu_hours", 0)) for run in runs)
    maximum = float(config.get("budget", {}).get("authorized_max_gpu_hours", 0))
    if allocated > maximum or maximum > 11.0:
        failures.append(f"invalid budget allocation: {allocated} / {maximum}")
    registry = arm_registry.from_legacy_mapping(rhvd.ARMS)
    resolved = {}
    for run in runs:
        arm = run["arm"]
        spec = registry[arm]
        resolved[arm] = spec.resolved()
        if spec.head != "reg":
            failures.append(f"{arm} is not a regression arm")
        if spec.context_reach != 48:
            failures.append(f"{arm} reach changed: {spec.context_reach}")
        if run.get("steps") != 30000 or run.get("batch") != 16:
            failures.append(f"{arm} is not legacy B16x30k")
        if run.get("lr") != 0.0003 or run.get("warmup") != 2000 or run.get("ema") != 0.999:
            failures.append(f"{arm} optimizer/EMA regime changed")
        legacy_checkpoint = Path(f"logs_rhcp/rhvd_{arm}_s2_t30000.pt")
        if legacy_checkpoint.exists():
            failures.append(f"pre-existing untracked-regime seed-2 checkpoint: {legacy_checkpoint}")
    if resolved.get("reg-long", {}).get("context_tokens") != 1152:
        failures.append("spatial package context-token count changed")
    if resolved.get("strided-flat-reg", {}).get("context_tokens") != 1088:
        failures.append("temporal package context-token count changed")
    required = [
        "data/loopmom_train_3000x100.npy",
        "data/loopmom_train_3000x100_actions.npy",
        "data/loopmom_test_500x200.npy",
        "data/loopmom_test_500x200_actions.npy",
        "logs_rhcp/hier_ae_mom.pt",
        "hvp_rev/rhvd_train_v2.py",
        "scripts/analyze_e4_factorial.py",
    ]
    missing = [path for path in required if not Path(path).is_file()]
    failures.extend(f"missing required artifact: {path}" for path in missing)
    report = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "config_sha256": sha256_file(args.config),
        "runs": len(runs),
        "allocated_gpu_hours": allocated,
        "authorized_gpu_hours": maximum,
        "resolved_arms": resolved,
        "required_hashes": {path: sha256_file(path) for path in required if Path(path).is_file()},
        "failures": failures,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
