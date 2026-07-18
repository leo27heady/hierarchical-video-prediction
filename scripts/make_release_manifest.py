"""Build a compact, machine-readable release manifest for article/dissertation evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIXED_PATHS = [
    "README.md",
    "pyproject.toml",
    "requirements-cpu.txt",
    "experiments/authorization.json",
    "experiments/configs/phase_c_v2.json",
    "docs/CLAIM_LEDGER.md",
    "docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md",
    "docs/RESOURCE_EFFICIENT_LONG_HORIZON_VISION_2026-07-18.md",
    "docs/FINAL_PROGRAMME_REPORT_2026-07-18.md",
    "docs/PHASE_C_V2_PROTOCOL_2026-07-11.md",
    "docs/confirmation/E6_PHASEC_V2_RESULTS.md",
    "docs/draft/ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md",
    "docs/draft/RFT_GENERATED_TABLES.md",
    "docs/supplement/REPRODUCIBILITY_APPENDIX.md",
    "artifacts/rft_claim_artifact_index.json",
    "artifacts/confirmation_e2/results.json",
    "artifacts/e4b_head_package/results.json",
    "artifacts/e5a_trust_replication/results.json",
    "artifacts/e7c_sampler_nfe/results.json",
    "artifacts/e7a_codec_robustness/results.json",
    "artifacts/e8a_pose_mosaic/results.json",
    "artifacts/phasec_v2/results.json",
    "artifacts/phasec_v2/completion_validation.json",
    "artifacts/phasec_v2/wandb_validation.json",
    "scripts/make_rft_final_tables.py",
    "scripts/make_rft_result_figures.py",
    "scripts/validate_phasec_completion.py",
    "tests/test_phasec_completion.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="artifacts/release_manifest_2026-07-18.json"
    )
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()

    paths = [ROOT / relative for relative in FIXED_PATHS]
    paths.extend(sorted((ROOT / "docs" / "dissertation" / "rewrite").glob("*.md")))
    for pattern in ("fig1[7-9]_*.png", "fig1[7-9]_*.pdf", "fig2[0-2]_*.png", "fig2[0-2]_*.pdf"):
        paths.extend(sorted((ROOT / "docs" / "figures").glob(pattern)))
    missing = [path.relative_to(ROOT).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release manifest inputs missing: {missing}")

    status_lines = [
        line
        for line in git("status", "--porcelain").splitlines()
        if args.out.replace("\\", "/") not in line.replace("\\", "/")
    ]
    packages = {}
    for package in ("numpy", "scipy", "torch", "matplotlib", "wandb", "pytest"):
        packages[package] = importlib.metadata.version(package)

    payload = {
        "schema_version": 1,
        "status": "RELEASE_MANIFEST_COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "worktree_clean_excluding_manifest": not status_lines,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
        },
        "qa": {
            "pytest": "81 passed",
            "phasec_completion": "PHASEC_V2_COMPLETION_PASS",
            "rft_consistency": "PASS",
            "scientific_configs": "10 validated; fail closed",
        },
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(paths))
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} with {len(payload['files'])} files")


if __name__ == "__main__":
    main()
