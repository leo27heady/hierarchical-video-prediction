"""Build a compact, source-only manifest for article/dissertation evidence.

The manifest intentionally excludes bytecode, caches, local virtual environments, rendered review
files, and wheels. Those artifacts are platform- or build-dependent and belong in a separately
generated distribution manifest after a clean build.
"""
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
    ".github/workflows/cpu-tests.yml",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "requirements-cpu.txt",
    "experiments/authorization.json",
    "experiments/configs/phase_c_v2.json",
    "docs/CLAIM_LEDGER.md",
    "docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md",
    "docs/RESOURCE_EFFICIENT_LONG_HORIZON_VISION_2026-07-18.md",
    "docs/FINAL_PROGRAMME_REPORT_2026-07-18.md",
    "docs/CODE_ARCHITECTURE.md",
    "docs/DATA_CARD_LOOPMOM.md",
    "docs/EXPERIMENT_AND_SCRIPT_INDEX.md",
    "docs/MODEL_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASE_QA_REPORT_2026-07-18.md",
    "docs/CONTENT_FINALIZATION_EXECUTION_PLAN_2026-07-19.md",
    "docs/CONTENT_FINALIZATION_CLAIM_MATRIX_2026-07-19.md",
    "docs/CONTENT_FINALIZATION_REPORT_2026-07-19.md",
    "docs/PHASE_C_V2_PROTOCOL_2026-07-11.md",
    "docs/confirmation/E6_PHASEC_V2_RESULTS.md",
    "docs/draft/RFT_GENERATED_TABLES.md",
    "docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md",
    "docs/dissertation/AUTHOR_INPUT_CHECKLIST.md",
    "docs/dissertation/final/DISSERTATION_MANUSCRIPT.md",
    "docs/dissertation/final/MANUSCRIPT_ASSEMBLY_REPORT.md",
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
    "scripts/assemble_final_manuscripts.py",
    "scripts/bind_dissertation_references.py",
    "scripts/build_final_documents.py",
    "scripts/release_check.py",
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
        "--out", default="artifacts/release_manifest_2026-07-19.json"
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail unless all manifest inputs are tracked and the worktree is clean",
    )
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()

    paths = [ROOT / relative for relative in FIXED_PATHS]
    paths.extend(sorted((ROOT / "docs" / "dissertation" / "rewrite").glob("*.md")))
    paths.extend(sorted((ROOT / "hvp").rglob("*.py")))
    py_typed = ROOT / "hvp" / "py.typed"
    if py_typed.is_file():
        paths.append(py_typed)
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
    tracked = set(git("ls-files").replace("\\", "/").splitlines())
    untracked_inputs = sorted(
        path.relative_to(ROOT).as_posix()
        for path in set(paths)
        if path.relative_to(ROOT).as_posix() not in tracked
    )
    if args.require_clean and (status_lines or untracked_inputs):
        raise RuntimeError(
            "clean release manifest requested with a dirty/untracked source state: "
            f"status={status_lines}, untracked_inputs={untracked_inputs}"
        )
    packages = {}
    for package in ("numpy", "scipy", "torch", "matplotlib", "wandb", "pytest"):
        packages[package] = importlib.metadata.version(package)

    payload = {
        "schema_version": 3,
        "status": (
            "CLEAN_SOURCE_RELEASE_MANIFEST"
            if not status_lines and not untracked_inputs
            else "WORKING_SOURCE_SNAPSHOT"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "worktree_clean_excluding_manifest": not status_lines,
        "all_inputs_git_tracked": not untracked_inputs,
        "untracked_inputs": untracked_inputs,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
        },
        "qa": {
            "scope": "source/evidence manifest; rendered files and wheels require separate clean-build manifests",
            "report": "docs/RELEASE_QA_REPORT_2026-07-18.md",
            "assembly_report": "docs/dissertation/final/MANUSCRIPT_ASSEMBLY_REPORT.md",
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
