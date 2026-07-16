"""Run E7a's complete CPU-only prelaunch gate and bind the reports to the tracked diff."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    python = str(Path("venv/Scripts/python.exe")) if Path("venv/Scripts/python.exe").exists() else "python"
    config = "experiments/configs/alternate_codec_robustness.json"
    commands = [
        [python, "-m", "pytest", "-q"],
        [python, "scripts/validate_scientific_configs.py"],
        [python, "scripts/check_rft_consistency.py"],
        [python, "scripts/validate_e7a_design.py", "--config", config],
        [python, "scripts/run_e7a_ae_parity.py"],
        [python, "-m", "compileall", "-q", "hvp_rev", "scripts", "tests"],
        ["git", "diff", "--check"],
    ]
    for command in commands:
        run(command, env)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], env=env)
    implementation = [
        "hvp_rev/hier_convae.py", "hvp_rev/hier_convae_v2.py", "hvp_rev/rhvd.py",
        "hvp_rev/rhvd_train_v2.py", "scripts/generate_loopmom_confirmation.py",
        "scripts/generate_e7a_codec_set.ps1", "scripts/evaluate_e7a_codec_floor.py",
        "scripts/analyze_e7a_codec_robustness.py", "scripts/validate_e7a_design.py",
        "scripts/validate_e7a_completion.py", "scripts/run_e7a_codec_robustness.ps1",
        "tests/test_e7a_codec.py",
    ]
    report = {
        "schema_version": 1,
        "status": "CPU_VALIDATION_PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, env=env
        ).splitlines(),
        "cuda_visible_devices": "-1",
        "commands": [" ".join(command) for command in commands],
        "config_sha256": sha256_file(config),
        "protocol_sha256": sha256_file("docs/E7A_ALTERNATE_CODEC_ROBUSTNESS_PROTOCOL_2026-07-16.md"),
        "ae_parity_sha256": sha256_file("gates/e7a_ae_parity_report.json"),
        "implementation_hashes": {path: sha256_file(path) for path in implementation},
    }
    atomic_write_json("gates/e7a_cpu_validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
