"""Run the complete CPU gate and emit a commit/diff-bound marker on success."""
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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    python = str(Path("venv/Scripts/python.exe")) if Path("venv/Scripts/python.exe").exists() else "python"
    run([python, "-m", "pytest", "-q"], env)
    run([python, "scripts/validate_scientific_configs.py"], env)
    run([python, "scripts/check_rft_consistency.py"], env)
    run([python, "scripts/validate_e2_design.py", "--config",
         "experiments/configs/e2_untouched_confirmation.json"], env)
    run([python, "-m", "compileall", "-q", "hvp_rev", "scripts", "tests"], env)
    run(["git", "diff", "--check"], env)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], env=env)
    untracked = git("ls-files", "--others", "--exclude-standard")
    marker = (
        "CPU_VALIDATION_PASS\n"
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"commit={git('rev-parse', 'HEAD')}\n"
        f"tracked_diff_sha256={hashlib.sha256(diff).hexdigest()}\n"
        f"untracked_files_sha256={hashlib.sha256(untracked.encode()).hexdigest()}\n"
        "cuda_visible_devices=-1\n"
    )
    path = Path("gates/cpu_validation.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(marker, encoding="utf-8")
    print(marker, end="")
    report_path = Path("gates/e2_cpu_validation_report.json")
    atomic_write_json(report_path, {
        "schema_version": 1,
        "status": "CPU_VALIDATION_PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files_sha256": hashlib.sha256(untracked.encode()).hexdigest(),
        "cuda_visible_devices": "-1",
        "commands": [
            "python -m pytest -q",
            "python scripts/validate_scientific_configs.py",
            "python scripts/check_rft_consistency.py",
            "python scripts/validate_e2_design.py --config experiments/configs/e2_untouched_confirmation.json",
            "python -m compileall -q hvp_rev scripts tests",
            "git diff --check",
        ],
        "e2_config_sha256": sha256_file("experiments/configs/e2_untouched_confirmation.json"),
    })
    print(json.dumps(json.loads(report_path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
