"""Frozen E7c analysis across NFE and sampler-seed variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import wb
from hvp_rev.analysis import paired_effect, simultaneous_cluster_intervals
from hvp_rev.provenance import atomic_write_bytes, atomic_write_json, sha256_file


ALIGN_KEYS = ("clip", "h", "hard", "dt", "seen", "fg_nonempty")
HIGHER_BETTER = {"fg_precision", "fg_recall", "fg_f1"}


def _discover(root: Path, dataset_hash: str) -> dict[tuple[str, int, int], Path]:
    found = {}
    for spec_path in root.rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "robustness-e7c" or spec.get("dataset_hash") != dataset_hash:
            continue
        sampler = spec["sampler"]
        key = (spec["arm"], int(sampler["nfe"]), int(sampler["sampler_seed"]))
        if key in found:
            raise RuntimeError(f"duplicate completed E7c cell: {key}")
        result = run_dir / "metrics/rollout_complete.npz"
        if not result.is_file():
            raise FileNotFoundError(result)
        found[key] = result
    return found


def _aligned(base, candidate) -> None:
    for key in ALIGN_KEYS:
        if not np.array_equal(base[key], candidate[key]):
            raise ValueError(f"paired E7c key differs: {key}")


def _markdown(result: dict) -> str:
    lines = [
        "# E7c sampler/NFE robustness results", "",
        f"> Dataset: `{result['dataset_id']}`. Positive favors the registered candidate.", "",
        f"Registered verdict: **{result['decision']['verdict']}**.", "",
        "| Contrast | NFE | Sampler seed | Effect | Simultaneous 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["primary"]:
        lines.append(
            f"| {row['contrast']} | {row['nfe']} | {row['sampler_seed']} | {row['mean']:+.4f} | "
            f"[{row['lo']:+.4f}, {row['hi']:+.4f}] |"
        )
    lines += ["", "No variant was selected or added after results were observed."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dataset_hash = manifest["artifacts"]["clips"]["sha256"]
    paths = _discover(Path(args.run_root), dataset_hash)
    expected = {
        (arm["arm"], int(v["nfe"]), int(v["sampler_seed"]))
        for arm in config["arms"] for v in config["variants"]
    }
    if set(paths) != expected:
        raise ValueError(f"E7c cell inventory mismatch: missing={sorted(expected-set(paths))}; extra={sorted(set(paths)-expected)}")

    deltas: dict[str, np.ndarray] = {}
    clips = None
    corroborating = []
    key_rows = []
    for contrast in config["contrasts"]:
        for variant in config["variants"]:
            nfe, sampler_seed = int(variant["nfe"]), int(variant["sampler_seed"])
            base = np.load(paths[(contrast["baseline"], nfe, sampler_seed)])
            candidate = np.load(paths[(contrast["candidate"], nfe, sampler_seed)])
            _aligned(base, candidate)
            if clips is None:
                clips = base["clip"]
            elif not np.array_equal(clips, base["clip"]):
                raise ValueError("E7c family populations are not aligned")
            name = f"{contrast['id']}::{variant['id']}"
            deltas[name] = base["fg"] - candidate["fg"]
            key_rows.append((name, contrast, variant))
            for metric in config["decision"]["corroborating_metrics"]:
                effect = paired_effect(
                    base[metric], candidate[metric], base["clip"],
                    lower_is_better=metric not in HIGHER_BETTER,
                    n_boot=config["evaluation"]["bootstrap_replicates"],
                    seed=config["evaluation"]["bootstrap_seed"],
                )
                corroborating.append({
                    "contrast": contrast["id"], "variant": variant["id"],
                    "nfe": nfe, "sampler_seed": sampler_seed, "metric": metric, **effect,
                })

    family = simultaneous_cluster_intervals(
        deltas, clips,
        n_boot=config["evaluation"]["bootstrap_replicates"],
        seed=config["evaluation"]["bootstrap_seed"],
        confidence=config["evaluation"]["family_confidence"],
    )
    primary = []
    for name, contrast, variant in key_rows:
        primary.append({
            "contrast": contrast["id"], "candidate": contrast["candidate"],
            "baseline": contrast["baseline"], "variant": variant["id"],
            "nfe": int(variant["nfe"]), "sampler_seed": int(variant["sampler_seed"]),
            "total_nfe_per_clip": 38 * int(variant["nfe"]),
            **family["contrasts"][name],
        })
    all_positive = all(row["mean"] > 0 for row in primary)
    all_clean = all(row["lo"] > 0 for row in primary)
    verdict = ("ROBUST_CI_CLEAN" if all_positive and all_clean else
               "DIRECTIONALLY_ROBUST" if all_positive else "SAMPLER_OR_NFE_SENSITIVE")
    result = {
        "schema_version": 1,
        "study": "robustness-e7c",
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "config_sha256": sha256_file(args.config),
        "family": {key: value for key, value in family.items() if key != "contrasts"},
        "primary": primary,
        "corroborating": corroborating,
        "decision": {
            "verdict": verdict,
            "all_point_estimates_expected_direction": all_positive,
            "all_simultaneous_lower_bounds_positive": all_clean,
            "rule": config["decision"],
        },
        "input_artifacts": {"::".join(map(str, key)): sha256_file(path) for key, path in sorted(paths.items())},
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, _markdown(result).encode("utf-8"))

    run_id = f"e7c-analysis-{manifest['dataset_id'].split('-')[-1]}"
    wb.init(
        run_id, project=args.wandb_project, run_id=run_id, resume="never",
        group=args.wandb_group, job_type="robustness-analysis", required=True,
        tags=["loopmom", "sampler-nfe-robustness", "analysis"],
        config={"config_sha256": result["config_sha256"],
                "manifest_sha256": result["dataset_manifest_sha256"]},
    )
    wb.summary({"analysis/verdict": verdict, "analysis/cells": len(paths),
                "analysis/contrasts": len(primary),
                "analysis/max_stat_critical": family["critical_value"]})
    wb.artifact(run_id, "robustness-analysis",
                [args.out_json, args.out_md, args.config, args.manifest],
                metadata={"dataset_id": manifest["dataset_id"], "verdict": verdict},
                aliases=["latest"])
    wb.finish()
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
