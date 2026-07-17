"""Fail-closed semantic and test gate for the E6 implementation before any protected access."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from hvp_rev import rhvd
from hvp_rev.provenance import atomic_write_json, sha256_file


IMPLEMENTATION = (
    "hvp_rev/data/moving_mnist.py",
    "hvp_rev/digit_classifier.py",
    "hvp_rev/classifier_validation.py",
    "hvp_rev/evaluation/frame_metrics.py",
    "hvp_rev/evaluation/object_metrics.py",
    "hvp_rev/phasec_v2.py",
    "hvp_rev/phasec_eval_v2.py",
    "hvp_rev/rhvd_train_v2.py",
    "scripts/gen_moving_v2.py",
    "scripts/train_phasec_classifier.py",
    "scripts/build_phasec_classifier_domains.py",
    "scripts/validate_phasec_classifier.py",
    "scripts/validate_phasec_evaluator.py",
    "scripts/validate_phasec_datasets.py",
    "scripts/analyze_phasec_v2.py",
    "scripts/validate_phasec_completion.py",
    "scripts/run_phasec_v2.ps1",
)


def _tokens(arm: str) -> tuple[int, int]:
    coarse, fine, _, stride, history, top, top_stride = rhvd.arm_cfg(arm)
    reach = rhvd.ctx_reach(coarse, fine, stride, history, top, top_stride)
    return coarse * 16 + (fine + history) * 64 + top * 4, reach


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    out, marker = Path(args.out), Path(args.marker)
    if out.exists() or marker.exists():
        raise FileExistsError("E6 CPU report/marker is immutable")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["status"] not in {"PREPARATION_FROZEN", "FROZEN"}:
        raise ValueError("E6 config is not preparation-frozen")
    if config["statistics"]["equivalence"]["enabled"]:
        raise ValueError("E6 equivalence must remain disabled")
    flat_tokens, flat_reach = _tokens("mmL-flat")
    hier_tokens, hier_reach = _tokens("mmL-hier")
    if (flat_tokens, hier_tokens, flat_reach, hier_reach) != (1152, 1152, 18, 48):
        raise ValueError("E6 registered context budgets/reaches changed")
    budget = config["budget"]
    allocated = (
        config["codec"]["max_gpu_hours"]
        + config["classifier"]["max_gpu_hours"]
        + config["validation"]["evaluator_reliability_max_gpu_hours"]
        + 6 * config["training"]["per_cell_max_gpu_hours"]
        + 6 * config["evaluation"]["per_cell_max_gpu_hours"]
    )
    if abs(allocated - budget["allocated_max_gpu_hours"]) > 1e-9:
        raise ValueError(f"E6 allocated budget arithmetic differs: {allocated}")
    if allocated > budget["authorized_max_gpu_hours"]:
        raise ValueError("E6 cell allocations exceed the programme maximum")
    legacy = Path("scripts/run_phaseC_mmnist.sh").read_text(encoding="utf-8")
    if "check_experiment_gate.py phase-c-legacy" not in legacy or "exit 97" not in legacy:
        raise ValueError("legacy Phase C path is not visibly fail-closed")
    for path in IMPLEMENTATION:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    command = [
        str(Path("venv/Scripts/python.exe")),
        "-m", "pytest", "-q",
        "tests/test_object_assignment.py",
        "tests/test_phasec_v2.py",
        "tests/test_moving_mnist_schema.py",
        "tests/test_classifier_and_trust.py",
        "tests/test_complete_evaluation.py",
    ]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"E6 CPU tests failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    report = {
        "schema_version": 1,
        "status": "CPU_VALIDATION_PASS",
        "source_commit": args.source_commit,
        "config_sha256": sha256_file(args.config),
        "context": {
            "flat_tokens": flat_tokens, "hierarchy_tokens": hier_tokens,
            "flat_reach": flat_reach, "hierarchy_reach": hier_reach,
        },
        "budget": {
            "allocated_max_gpu_hours": allocated,
            "authorized_max_gpu_hours": budget["authorized_max_gpu_hours"],
        },
        "tests": {
            "command": command,
            "stdout": completed.stdout.strip(),
        },
        "implementation_hashes": {path: sha256_file(path) for path in IMPLEMENTATION},
    }
    atomic_write_json(out, report)
    atomic_write_json(marker, {
        "schema_version": 1,
        "status": "CPU_VALIDATION_PASS",
        "subject": out.as_posix(),
        "subject_sha256": sha256_file(out),
        "source_commit": args.source_commit,
    })
    print(f"PHASEC_V2_CPU_VALIDATION_PASS {completed.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
