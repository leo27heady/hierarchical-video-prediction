"""Run the complete CPU-only E8a prelaunch gate."""
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
    commands = [
        [python, "-m", "pytest", "-q"],
        [python, "scripts/validate_scientific_configs.py"],
        [python, "scripts/check_rft_consistency.py"],
        [python, "scripts/validate_e8a_design.py", "--config",
         "experiments/configs/pose_indexed_retrieval.json", "--protocol",
         "docs/E8A_POSE_MOSAIC_PROTOCOL_2026-07-17.md"],
        [python, "scripts/run_e8a_label_parity.py"],
        [python, "-m", "compileall", "-q", "hvp_rev", "scripts", "tests"],
        ["git", "diff", "--check"],
    ]
    for command in commands:
        run(command, env)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], env=env)
    report = {
        "schema_version": 1,
        "status": "CPU_VALIDATION_PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "cuda_visible_devices": "-1",
        "commands": [" ".join(command) for command in commands],
        "config_sha256": sha256_file("experiments/configs/pose_indexed_retrieval.json"),
        "protocol_sha256": sha256_file("docs/E8A_POSE_MOSAIC_PROTOCOL_2026-07-17.md"),
        "label_parity_sha256": sha256_file("gates/e8a_label_parity_report.json"),
    }
    atomic_write_json("gates/e8a_cpu_validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
