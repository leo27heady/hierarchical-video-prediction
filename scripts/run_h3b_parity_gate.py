"""Run and record the mandatory CPU numerical-parity gate for new RHVD training."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


COMMANDS = (
    ["python", "-m", "pytest", "tests/test_h3b_parity.py", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["python", "-m", "compileall", "-q", "hvp_rev", "scripts", "tests"],
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    python = root / "venv" / "Scripts" / "python.exe"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    completed = []
    for command in COMMANDS:
        resolved = [str(python) if command[0] == "python" else command[0], *command[1:]]
        print("+", " ".join(resolved), flush=True)
        result = subprocess.run(resolved, cwd=root, env=environment, text=True)
        completed.append({"command": command, "exit_code": result.returncode})
        if result.returncode:
            raise SystemExit(result.returncode)
    diff = subprocess.run(["git", "diff", "--check"], cwd=root).returncode
    if diff:
        raise SystemExit(diff)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_files = [
        root / "hvp_rev" / "rhvd.py",
        root / "hvp_rev" / "rhvd_train_v2.py",
        root / "hvp_rev" / "provenance.py",
        root / "hvp_rev" / "arm_registry.py",
        root / "tests" / "test_h3b_parity.py",
    ]
    report = {
        "schema_version": 1,
        "status": "H3B_PARITY_PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "cuda_visible_devices": "-1",
        "commands": completed,
        "source_hashes": {path.relative_to(root).as_posix(): _sha(path) for path in source_files},
        "parity_scope": [
            "sampled clip and target-start indices",
            "batch tensors",
            "model construction and initial parameters",
            "first fixed-seed loss",
            "AdamW warm-up, gradient clipping and parameter update",
            "EMA update",
            "arm-registry legacy mapping",
            "RNG-complete uninterrupted-versus-resumed trajectory",
        ],
        "consequence": "v2 wrapper is the same numerical regime as the guarded legacy kernel",
    }
    out = root / "gates" / "h3b_parity_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
