"""Registered simultaneous analysis for the E6 persistent-object Moving-MNIST study."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hvp_rev import wb
from hvp_rev.analysis import paired_effect, simultaneous_cluster_intervals
from hvp_rev.provenance import atomic_write_bytes, atomic_write_json, sha256_file


ERROR_METRICS = {
    "foreground_mse": "mse_gt_fg_nonempty",
    "union_mse": "mse_union",
    "trajectory_error": "object_trajectory_error",
}
BENEFIT_METRICS = {
    "identity_accuracy": "object_identity_accuracy",
    "object_coverage": "object_coverage",
}
SUMMARY_METRICS = (
    "mse_gt_fg_nonempty", "mse_full", "mse_gt_bg", "mse_union", "fg_precision", "fg_recall",
    "fg_f1", "object_trajectory_error", "object_identity_accuracy", "object_identity_coverage",
    "object_coverage", "object_missed", "object_extra",
)


def _discover(root: Path, dataset_id: str) -> dict[tuple[str, int], Path]:
    found = {}
    base = root / "phase-c-v2-evaluation"
    for spec_path in base.rglob("run_spec.json") if base.exists() else []:
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec["arm_config"].get("dataset_id") != dataset_id:
            continue
        key = (spec["arm"], int(spec["seed"]))
        result = run_dir / "metrics" / "phasec_v2_records.npz"
        if key in found or not result.is_file():
            raise ValueError(f"duplicate or incomplete E6 evaluation: {key}")
        found[key] = result
    return found


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    key = np.stack([arrays["clip"], arrays["h"]], axis=1)
    if len(np.unique(key, axis=0)) != len(key):
        raise ValueError(f"duplicate E6 paired key: {path}")
    order = np.lexsort((arrays["h"], arrays["clip"]))
    return {name: value[order] for name, value in arrays.items()}


def _aligned(reference: dict, candidate: dict) -> None:
    for key in ("clip", "h", "frame", "overlap_any"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"E6 paired alignment differs for {key}")


def _decision(primary: list[dict]) -> str:
    by_seed = {
        seed: {row["endpoint"]: row for row in primary if row["seed"] == seed}
        for seed in (0, 1, 2)
    }
    fg_positive = all(by_seed[seed]["foreground_mse"]["lo"] > 0 for seed in by_seed)
    fg_negative = all(by_seed[seed]["foreground_mse"]["hi"] < 0 for seed in by_seed)
    other = [key for key in (*ERROR_METRICS, *BENEFIT_METRICS) if key != "foreground_mse"]
    hierarchy_support = all(
        sum(by_seed[seed][endpoint]["mean"] > 0 for seed in by_seed) >= 2
        and not any(by_seed[seed][endpoint]["hi"] < 0 for seed in by_seed)
        for endpoint in other
    )
    flat_support = all(
        sum(by_seed[seed][endpoint]["mean"] < 0 for seed in by_seed) >= 2
        and not any(by_seed[seed][endpoint]["lo"] > 0 for seed in by_seed)
        for endpoint in other
    )
    if fg_positive and hierarchy_support:
        return "HIERARCHY_ADVANTAGE"
    if fg_negative and flat_support:
        return "FLAT_ADVANTAGE"
    return "MIXED_BOUNDARY"


def _markdown(result: dict) -> str:
    lines = [
        "# E6 persistent-object long Moving-MNIST results",
        "",
        f"Registered verdict: **{result['decision']['verdict']}**.",
        "",
        "Positive effects favor the hierarchical 1,152-token package. Intervals are one "
        "simultaneous 95% max-statistic clip-cluster family over all 15 co-primary effects.",
        "",
        "| Seed | Endpoint | Effect | Simultaneous 95% CI |",
        "|---:|---|---:|---:|",
    ]
    for row in result["primary"]:
        lines.append(
            f"| {row['seed']} | {row['endpoint']} | {row['mean']:+.4f} | "
            f"[{row['lo']:+.4f}, {row['hi']:+.4f}] |"
        )
    lines += [
        "",
        "| Seed | Arm | FG MSE | Union MSE | Identity | Trajectory px | Coverage | Missed | Extra |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["arm_summary"]:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['mse_gt_fg_nonempty']:.4f} | "
            f"{row['mse_union']:.4f} | {row['object_identity_accuracy']:.4f} | "
            f"{row['object_trajectory_error']:.2f} | {row['object_coverage']:.4f} | "
            f"{row['object_missed']:.3f} | {row['object_extra']:.3f} |"
        )
    lines += [
        "",
        "Equivalence was disabled before protected access. Moving-MNIST is an external boundary, "
        "not evidence for LoopMom's exact re-entry mechanism or natural-video transfer.",
    ]
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
    paths = _discover(Path(args.run_root), manifest["dataset_id"])
    expected = {(arm, seed) for arm in ("mmL-flat", "mmL-hier") for seed in (0, 1, 2)}
    if set(paths) != expected:
        raise ValueError(f"E6 evaluation inventory mismatch: {set(paths)} != {expected}")
    arrays = {key: _load(path) for key, path in paths.items()}
    reference = arrays[("mmL-flat", 0)]
    for value in arrays.values():
        _aligned(reference, value)
    clusters = reference["clip"]

    contrasts = {}
    for seed in (0, 1, 2):
        flat, hierarchy = arrays[("mmL-flat", seed)], arrays[("mmL-hier", seed)]
        for endpoint, metric in ERROR_METRICS.items():
            contrasts[f"s{seed}:{endpoint}"] = flat[metric] - hierarchy[metric]
        for endpoint, metric in BENEFIT_METRICS.items():
            contrasts[f"s{seed}:{endpoint}"] = hierarchy[metric] - flat[metric]
    stats = config["statistics"]["bootstrap"]
    family = simultaneous_cluster_intervals(
        contrasts, clusters,
        n_boot=int(stats["replicates"]),
        seed=int(stats["seed"]),
        confidence=0.95,
    )
    primary = []
    for seed in (0, 1, 2):
        for endpoint in (*ERROR_METRICS, *BENEFIT_METRICS):
            primary.append({
                "seed": seed,
                "endpoint": endpoint,
                **family["contrasts"][f"s{seed}:{endpoint}"],
            })

    arm_summary = []
    for seed in (0, 1, 2):
        for arm in ("mmL-flat", "mmL-hier"):
            value = arrays[(arm, seed)]
            arm_summary.append({
                "seed": seed,
                "arm": arm,
                **{metric: float(np.nanmean(value[metric])) for metric in SUMMARY_METRICS},
            })
    overlap = {}
    for population, mask in (
        ("nonoverlap", ~reference["overlap_any"].astype(bool)),
        ("overlap", reference["overlap_any"].astype(bool)),
    ):
        overlap[population] = []
        for seed in (0, 1, 2):
            flat, hierarchy = arrays[("mmL-flat", seed)], arrays[("mmL-hier", seed)]
            for endpoint, metric in ERROR_METRICS.items():
                overlap[population].append({
                    "seed": seed, "endpoint": endpoint,
                    **paired_effect(
                        flat[metric][mask], hierarchy[metric][mask], clusters[mask],
                        lower_is_better=True, n_boot=int(stats["replicates"]), seed=int(stats["seed"]),
                    ),
                })
            for endpoint, metric in BENEFIT_METRICS.items():
                overlap[population].append({
                    "seed": seed, "endpoint": endpoint,
                    **paired_effect(
                        flat[metric][mask], hierarchy[metric][mask], clusters[mask],
                        lower_is_better=False, n_boot=int(stats["replicates"]), seed=int(stats["seed"]),
                    ),
                })
    verdict = _decision(primary)
    result = {
        "schema_version": 1,
        "study": "phase-c-persistent-object-v2",
        "dataset_id": manifest["dataset_id"],
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "family": {key: value for key, value in family.items() if key != "contrasts"},
        "primary": primary,
        "arm_summary": arm_summary,
        "overlap_strata": overlap,
        "decision": {
            "verdict": verdict,
            "equivalence_enabled": False,
            "rule": config["statistics"]["registered_verdict"],
        },
        "input_artifacts": {
            f"{arm}-s{seed}": sha256_file(path) for (arm, seed), path in sorted(paths.items())
        },
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, _markdown(result).encode("utf-8"))
    run_id = f"phasec-v2-analysis-{manifest['dataset_id'].split('-')[-1]}"
    wb.init(
        run_id,
        config={"result": result},
        project=args.wandb_project,
        run_id=run_id,
        group=args.wandb_group,
        tags=["moving-mnist", "phase-c-v2", "registered-analysis"],
        job_type="phasec-v2-analysis",
        required=True,
        resume="never",
    )
    for row in primary:
        prefix = f"primary/s{row['seed']}/{row['endpoint']}"
        wb.summary({
            f"{prefix}/mean": row["mean"],
            f"{prefix}/lo": row["lo"],
            f"{prefix}/hi": row["hi"],
        })
    wb.summary({"decision/verdict": verdict})
    wb.artifact(
        f"phasec-v2-analysis-{manifest['dataset_id'].split('-')[-1]}",
        "phasec-v2-analysis",
        [args.out_json, args.out_md, args.config, args.manifest],
        aliases=["latest"],
    )
    wb.finish()
    print(f"PHASEC_V2_ANALYSIS_DONE {verdict}", flush=True)


if __name__ == "__main__":
    main()
