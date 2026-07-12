"""Fail-closed CPU validator for the frozen E2 run matrix and dataset manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.evaluation.reach import ContextGeometry
from hvp_rev.provenance import atomic_write_json, sha256_file
from hvp_rev.rhvd import ARMS, arm_cfg, ctx_reach


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    runs = config.get("runs", [])
    keys = [(run["arm"], int(run["seed"]), int(run["steps"])) for run in runs]
    if len(keys) != len(set(keys)):
        failures.append("duplicate arm/seed/steps run")
    allocated = sum(float(run["max_gpu_hours"]) for run in runs)
    maximum = float(config["budget"]["authorized_max_gpu_hours"])
    if allocated > maximum:
        failures.append(f"per-run allocation {allocated:.3f} exceeds tranche maximum {maximum:.3f}")

    available = set()
    for run in runs:
        arm, seed, steps = run["arm"], int(run["seed"]), int(run["steps"])
        if arm not in ARMS:
            failures.append(f"unknown arm {arm}")
            continue
        path = Path(f"logs_rhcp/rhvd_{arm}_s{seed}_t{steps}.pt")
        if not path.is_file():
            failures.append(f"missing checkpoint {path}")
        elif sha256_file(path) != run["checkpoint_sha256"]:
            failures.append(f"checkpoint hash mismatch {path}")
        available.add((arm, seed, steps))

    for pair in config["primary_pairs"]:
        for seed in pair["seeds"]:
            for arm in (pair["candidate"], pair["baseline"]):
                if not any(key[0] == arm and key[1] == seed for key in available):
                    failures.append(f"primary pair {pair['id']} lacks {arm} seed {seed}")
        candidate_cfg = arm_cfg(pair["candidate"])
        baseline_cfg = arm_cfg(pair["baseline"])
        candidate_reach = ctx_reach(
            candidate_cfg[0], candidate_cfg[1], candidate_cfg[3], candidate_cfg[4],
            candidate_cfg[5], candidate_cfg[6],
        )
        baseline_reach = ctx_reach(
            baseline_cfg[0], baseline_cfg[1], baseline_cfg[3], baseline_cfg[4],
            baseline_cfg[5], baseline_cfg[6],
        )
        if candidate_reach != pair["candidate_reach"] or baseline_reach != pair["baseline_reach"]:
            failures.append(
                f"primary reach mismatch {pair['id']}: {candidate_reach}/{baseline_reach}"
            )
        # Constructing the geometry also checks the intended band representation.
        ContextGeometry(pair["candidate"], coarse_slots=candidate_cfg[0],
                        coarse_stride=candidate_cfg[3], fine_recent=candidate_cfg[1],
                        fine_history_slots=candidate_cfg[4])

    for pair in config["frontier_pairs"]:
        steps = int(pair.get("steps", 30000 if pair["regime"] == "B16x30k" else 60000))
        for arm in (pair["candidate"], pair["baseline"]):
            if (arm, int(pair["seed"]), steps) not in available:
                failures.append(f"frontier pair {pair['id']} lacks {arm} at {steps} steps")

    manifest_hash = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(manifest_path)
        if manifest.get("purpose") != "confirmation_holdout":
            failures.append("dataset purpose is not confirmation_holdout")
        for key in ("seed", "n", "frames", "world", "digits", "speed", "p_cont", "p_hold"):
            expected_key = "clips" if key == "n" else key
            expected = config["dataset"][expected_key]
            actual = manifest["generator"]["seed"] if key == "seed" else manifest["configuration"][key]
            if actual != expected:
                failures.append(f"dataset {key} mismatch: {actual!r} != {expected!r}")
        for artifact in manifest["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                failures.append(f"dataset artifact missing/hash mismatch: {path}")

    report = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "config": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest_hash,
        "runs": len(runs),
        "allocated_gpu_hours": allocated,
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
