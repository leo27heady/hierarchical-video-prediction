"""Frozen paired analysis for the E8a pose-indexed raw-memory reference."""
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
HIGHER_BETTER = {
    "fg_precision", "fg_recall", "fg_f1", "current_crop_observed_fraction"
}


def registered_verdict(primary: list[dict], neural_effect_ids: set[str] | None = None) -> str:
    """Apply the ceiling rule to neural-to-raw effects, not raw-span saturation controls."""
    selected = (
        [row for row in primary if row["effect"] in neural_effect_ids]
        if neural_effect_ids is not None
        else primary
    )
    if not selected or (neural_effect_ids is not None and len(selected) != len(neural_effect_ids)):
        raise ValueError("registered neural-reference effect inventory is incomplete")
    if not all(row["mean"] > 0 for row in selected):
        return "RAW_ACCESS_CEILING_NOT_ESTABLISHED"
    if all(row["lo"] > 0 for row in selected):
        return "RAW_ACCESS_CEILING_CI_CLEAN"
    return "RAW_ACCESS_DIRECTIONAL_REFERENCE"


def simultaneous_family_allowing_degenerate(
    contrasts: dict[str, np.ndarray],
    clusters: np.ndarray,
    **bootstrap,
) -> dict:
    """Give exact intervals to cluster-degenerate controls and studentize the rest.

    K38 can already cover the complete 192-column world. In that registered saturation case its
    contrast with K48 is identically zero, so its sampling interval is exactly [0, 0] and it
    contributes no random max statistic.
    """
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    degenerate: dict[str, float] = {}
    nondegenerate: dict[str, np.ndarray] = {}
    for name, values in contrasts.items():
        values = np.asarray(values, dtype=np.float64)
        means = np.asarray([values[clusters == cluster].mean() for cluster in unique])
        if np.ptp(means) <= np.finfo(np.float64).eps:
            degenerate[name] = float(means[0])
        else:
            nondegenerate[name] = values
    if not nondegenerate:
        raise ValueError("all registered E8a contrasts are cluster-degenerate")
    family = simultaneous_cluster_intervals(nondegenerate, clusters, **bootstrap)
    for name, mean in degenerate.items():
        family["contrasts"][name] = {
            "mean": mean,
            "lo": mean,
            "hi": mean,
            "bootstrap_se": 0.0,
            "degenerate": True,
        }
    family["degenerate_contrasts"] = sorted(degenerate)
    return family


def population_mask(reference, rule: dict) -> np.ndarray:
    """Resolve a frozen geometric population without evaluating a free-form expression."""
    mask = np.ones(len(reference["dt"]), dtype=bool)
    mask &= reference["dt"] > float(rule["dt_lower_exclusive"])
    mask &= reference["dt"] <= float(rule["dt_upper_inclusive"])
    mask &= reference["hard"] > float(rule["hard_lower_exclusive"])
    mask &= reference["seen"] >= float(rule["seen_lower_inclusive"])
    return mask


def _discover(root: Path, dataset_hash: str) -> tuple[dict[str, Path], dict[str, Path]]:
    neural: dict[str, Path] = {}
    mosaics: dict[str, Path] = {}
    for spec_path in root.rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "reference-e8a" or spec.get("dataset_hash") != dataset_hash:
            continue
        if spec["arm"] == "pose-mosaic":
            for path in (run_dir / "metrics").glob("pose-mosaic-k*.npz"):
                if path.stem in mosaics:
                    raise RuntimeError(f"duplicate E8a mosaic result: {path.stem}")
                mosaics[path.stem] = path
        else:
            if spec["arm"] in neural:
                raise RuntimeError(f"duplicate E8a neural result: {spec['arm']}")
            result = run_dir / "metrics/rollout_complete.npz"
            if not result.is_file():
                raise FileNotFoundError(result)
            neural[spec["arm"]] = result
    return neural, mosaics


def _aligned(reference, other) -> None:
    for key in ALIGN_KEYS:
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"paired E8a key differs: {key}")


def _markdown(result: dict) -> str:
    lines = [
        "# E8a pose-indexed raw-memory reference results",
        "",
        f"> Dataset: `{result['dataset_id']}`. Positive favors `pose-mosaic-k48`.",
        "",
        f"Registered verdict: **{result['decision']['verdict']}**.",
        "",
        "The pose mosaic receives exact positions and raw-pixel storage. It is a privileged "
        "access reference, not a token-matched neural competitor.",
        "",
        "| Effect | Estimate | Simultaneous 95% CI |",
        "|---|---:|---:|",
    ]
    for row in result["primary"]:
        lines.append(
            f"| {row['effect']} | {row['mean']:+.4f} | "
            f"[{row['lo']:+.4f}, {row['hi']:+.4f}] |"
        )
    lines += [
        "",
        "| Method | FG MSE | Full MSE | Background MSE | Union MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["method_summary"]:
        lines.append(
            f"| {row['method']} | {row['fg']:.4f} | {row['mse_full']:.4f} | "
            f"{row['mse_gt_bg']:.4f} | {row['mse_union']:.4f} |"
        )
    lines += [
        "",
        "| Pose mosaic | Stored K | Median stored columns | Mean crop coverage | Pixel payload | Pose payload |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["mosaic_storage"]:
        lines.append(
            f"| {row['method']} | {row['span']} | {row['median_stored_world_columns']:.1f} | "
            f"{row['mean_current_crop_observed_fraction']:.3f} | "
            f"{row['median_stored_pixel_payload_bytes']:.0f} B | "
            f"{row['pose_bytes']} B |"
        )
    lines += [
        "",
        "No memory span, fill rule, checkpoint, sampler, threshold or endpoint was added after "
        "the reference set was generated.",
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
    dataset_hash = manifest["artifacts"]["clips"]["sha256"]
    neural_paths, mosaic_paths = _discover(Path(args.run_root), dataset_hash)
    expected_neural = {row["arm"] for row in config["neural_reference_runs"]}
    expected_mosaics = {row["id"] for row in config["memory"]["variants"]}
    if set(neural_paths) != expected_neural:
        raise ValueError(
            f"E8a neural inventory mismatch: missing={sorted(expected_neural-set(neural_paths))}; "
            f"extra={sorted(set(neural_paths)-expected_neural)}"
        )
    if set(mosaic_paths) != expected_mosaics:
        raise ValueError(
            f"E8a mosaic inventory mismatch: missing={sorted(expected_mosaics-set(mosaic_paths))}; "
            f"extra={sorted(set(mosaic_paths)-expected_mosaics)}"
        )
    paths = {**neural_paths, **mosaic_paths}
    arrays = {method: np.load(path) for method, path in paths.items()}
    reference = arrays["pose-mosaic-k48"]
    for values in arrays.values():
        _aligned(reference, values)
    clips = reference["clip"]

    deltas = {}
    registered_rows = []
    corroborating = []
    for effect in config["registered_effects"]:
        baseline = arrays[effect["baseline"]]
        candidate = arrays[effect["candidate"]]
        deltas[effect["id"]] = baseline["fg"] - candidate["fg"]
        registered_rows.append(effect)
        for metric in config["decision"]["corroborating_metrics"]:
            if metric not in baseline.files or metric not in candidate.files:
                continue
            corroborating.append({
                "effect": effect["id"],
                "metric": metric,
                **paired_effect(
                    baseline[metric],
                    candidate[metric],
                    clips,
                    lower_is_better=metric not in HIGHER_BETTER,
                    n_boot=int(config["evaluation"]["bootstrap_replicates"]),
                    seed=int(config["evaluation"]["bootstrap_seed"]),
                ),
            })
    family = simultaneous_family_allowing_degenerate(
        deltas,
        clips,
        n_boot=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        confidence=float(config["evaluation"]["family_confidence"]),
    )
    primary = [{
        "effect": effect["id"],
        "baseline": effect["baseline"],
        "candidate": effect["candidate"],
        **family["contrasts"][effect["id"]],
    } for effect in registered_rows]
    neural_effect_ids = set(config["decision"]["neural_reference_effect_ids"])
    verdict = registered_verdict(primary, neural_effect_ids)

    method_summary = []
    for method in [row["id"] for row in config["memory"]["variants"]] + [
        row["arm"] for row in config["neural_reference_runs"]
    ]:
        values = arrays[method]
        method_summary.append({
            "method": method,
            **{
                key: float(np.nanmean(values[key]))
                for key in ("fg", "mse_full", "mse_gt_bg", "mse_union", "fg_precision",
                            "fg_recall", "missed_components", "extra_components")
            },
        })
    spans = {row["id"]: int(row["warmup_frames_stored"])
             for row in config["memory"]["variants"]}
    mosaic_storage = []
    for method in spans:
        values = arrays[method]
        mosaic_storage.append({
            "method": method,
            "span": spans[method],
            "median_stored_world_columns":
                float(np.median(values["stored_world_coverage"]) * config["dataset"]["world"]),
            "mean_current_crop_observed_fraction":
                float(np.mean(values["current_crop_observed_fraction"])),
            "median_stored_pixel_payload_bytes":
                float(np.median(values["stored_pixel_payload_bytes"])),
            "allocated_mosaic_bytes": int(np.max(values["allocated_mosaic_bytes"])),
            "stored_pose_bytes": int(np.max(values["stored_pose_bytes"])),
            "query_pose_bytes": int(np.max(values["query_pose_bytes"])),
            "pose_bytes": int(np.max(values["pose_bytes"])),
            "mask_bytes": int(np.max(values["mask_bytes"])),
        })
    storage_by_method = {row["method"]: row for row in mosaic_storage}
    raw38 = next(row for row in primary if row["effect"] == "raw-reach-38-to-48")
    coverage_saturated_by_k38 = bool(
        storage_by_method["pose-mosaic-k38"]["median_stored_world_columns"]
        == storage_by_method["pose-mosaic-k48"]["median_stored_world_columns"]
        and abs(raw38["mean"]) <= 1e-12
    )
    secondary_populations = {}
    for population_id, rule in config["evaluation"]["secondary_populations"].items():
        selected = population_mask(reference, rule)
        if not selected.any():
            raise ValueError(f"E8a secondary population is empty: {population_id}")
        effects = []
        for effect in registered_rows:
            baseline = arrays[effect["baseline"]]
            candidate = arrays[effect["candidate"]]
            effects.append({
                "effect": effect["id"],
                **paired_effect(
                    baseline["fg"][selected],
                    candidate["fg"][selected],
                    clips[selected],
                    lower_is_better=True,
                    n_boot=int(config["evaluation"]["bootstrap_replicates"]),
                    seed=int(config["evaluation"]["bootstrap_seed"]),
                ),
            })
        secondary_populations[population_id] = {
            "rule": rule,
            "records": int(selected.sum()),
            "clips": int(np.unique(clips[selected]).size),
            "effects": effects,
        }

    result = {
        "schema_version": 1,
        "study": "reference-e8a",
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "config_sha256": sha256_file(args.config),
        "family": {key: value for key, value in family.items() if key != "contrasts"},
        "primary": primary,
        "secondary_populations": secondary_populations,
        "corroborating": corroborating,
        "method_summary": method_summary,
        "mosaic_storage": mosaic_storage,
        "decision": {
            "verdict": verdict,
            "neural_reference_effect_ids": sorted(neural_effect_ids),
            "all_neural_point_estimates_positive": all(
                row["mean"] > 0 for row in primary if row["effect"] in neural_effect_ids
            ),
            "all_neural_simultaneous_lower_bounds_positive": all(
                row["lo"] > 0 for row in primary if row["effect"] in neural_effect_ids
            ),
            "coverage_saturated_by_k38": coverage_saturated_by_k38,
            "rule": config["decision"],
        },
        "input_artifacts": {
            method: sha256_file(path) for method, path in sorted(paths.items())
        },
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, _markdown(result).encode("utf-8"))
    run_id = f"e8a-analysis-{manifest['dataset_id'].split('-')[-1]}"
    wb.init(
        run_id,
        project=args.wandb_project,
        run_id=run_id,
        resume="never",
        group=args.wandb_group,
        job_type="pose-mosaic-analysis",
        required=True,
        tags=["loopmom", "e8a", "pose-indexed", "analysis"],
        config={
            "config_sha256": result["config_sha256"],
            "manifest_sha256": result["dataset_manifest_sha256"],
        },
    )
    wb.summary({
        "analysis/verdict": verdict,
        "analysis/methods": len(paths),
        "analysis/effects": len(primary),
        "analysis/max_stat_critical": family["critical_value"],
    })
    wb.artifact(
        run_id,
        "pose-mosaic-analysis",
        [args.out_json, args.out_md, args.config, args.manifest],
        metadata={"dataset_id": manifest["dataset_id"], "verdict": verdict},
        aliases=["latest"],
    )
    wb.finish()
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
