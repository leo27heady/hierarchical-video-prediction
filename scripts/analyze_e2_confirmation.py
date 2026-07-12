"""Frozen E2 confirmation analysis: selected primaries plus max-statistic frontier family."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import wb
from hvp_rev.analysis import paired_effect, simultaneous_cluster_intervals
from hvp_rev.evaluation.reach import ContextGeometry, pair_specific_reach_mask
from hvp_rev.provenance import atomic_write_json, atomic_write_bytes, sha256_file


def _discover(root: Path, dataset_hash: str) -> dict[tuple, Path]:
    found: dict[tuple, Path] = {}
    for spec_path in root.rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "confirmation-e2" or spec.get("dataset_hash") != dataset_hash:
            continue
        key = (
            spec["arm"], int(spec["seed"]), int(spec["optimizer_updates"]),
            spec["arm_config"]["checkpoint_sha256"],
        )
        if key in found:
            raise RuntimeError(f"multiple completed E2 runs for {key}")
        result = run_dir / "metrics/rollout_complete.npz"
        if not result.is_file():
            raise FileNotFoundError(result)
        found[key] = result
    return found


def _effect(base: np.ndarray, candidate: np.ndarray, clips: np.ndarray, mask=None,
            *, higher_is_better=False) -> dict:
    if mask is not None:
        base, candidate, clips = base[mask], candidate[mask], clips[mask]
    return paired_effect(
        base, candidate, clips, lower_is_better=not higher_is_better, n_boot=10000, seed=0
    )


def _component_coverage(data) -> np.ndarray:
    gt = data["gt_component_count"].astype(np.float64)
    matched = gt - data["missed_components"].astype(np.float64)
    return np.divide(matched, gt, out=np.ones_like(gt), where=gt > 0)


def _aligned(base, candidate) -> None:
    for key in ("clip", "h", "hard", "dt", "seen", "fg_nonempty"):
        if not np.array_equal(base[key], candidate[key]):
            raise ValueError(f"paired result key differs: {key}")


def _markdown(result: dict) -> str:
    lines = [
        "# E2 untouched confirmation results",
        "",
        f"> Dataset: `{result['dataset_id']}`. Positive effects favor the candidate package.",
        "",
        "## Selected primary contrasts",
        "",
        "| Pair | Seed | Population | Metric | Effect | 95% CI | Records | Clips |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in result["primary"]:
        lines.append(
            f"| {row['pair']} | {row['seed']} | {row['population']} | {row['metric']} | "
            f"{row['mean']:+.4f} | [{row['lo']:+.4f}, {row['hi']:+.4f}] | "
            f"{row['records']} | {row['clusters']} |"
        )
    lines += [
        "",
        "## Frozen frontier family",
        "",
        f"Method: {result['frontier']['method']}; critical value "
        f"{result['frontier']['critical_value']:.3f}.",
        "",
        "| Contrast | Effect | Simultaneous 95% CI |",
        "|---|---:|---:|",
    ]
    for name, row in result["frontier"]["contrasts"].items():
        lines.append(f"| {name} | {row['mean']:+.4f} | [{row['lo']:+.4f}, {row['hi']:+.4f}] |")
    lines += [
        "",
        "No interpolation or post-result primary selection is permitted. c2 regimes remain separate.",
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
    discovered = _discover(Path(args.run_root), dataset_hash)
    run_paths = {}
    for run in config["runs"]:
        key = (run["arm"], run["seed"], run["steps"], run["checkpoint_sha256"])
        if key not in discovered:
            raise FileNotFoundError(f"missing completed E2 result {key}")
        run_paths[(run["arm"], run["seed"], run["steps"])] = discovered[key]

    primary_rows = []
    for pair in config["primary_pairs"]:
        for seed in pair["seeds"]:
            base = np.load(run_paths[(pair["baseline"], seed, 30000)])
            candidate = np.load(run_paths[(pair["candidate"], seed, 30000)])
            _aligned(base, candidate)
            all_effect = _effect(base["fg"], candidate["fg"], base["clip"])
            primary_rows.append({
                "pair": pair["id"], "seed": seed, "population": "legacy-all",
                "metric": "legacy_fg", **all_effect,
            })
            mask = pair_specific_reach_mask(
                base["dt"],
                ContextGeometry(pair["candidate"], coarse_slots=pair["candidate_reach"]),
                ContextGeometry(pair["baseline"], fine_recent=pair["baseline_reach"]),
                answerable=base["seen"] >= 0.5,
                foreground_nonempty=base["fg_nonempty"],
            )
            for metric in ("fg", "mse_gt_fg_nonempty", "mse_union", "mse_full", "mse_gt_bg"):
                effect = _effect(base[metric], candidate[metric], base["clip"], mask)
                primary_rows.append({
                    "pair": pair["id"], "seed": seed, "population": pair["primary_band"],
                    "metric": metric, **effect,
                })
            for metric in ("fg_precision", "fg_recall", "fg_f1"):
                effect = _effect(
                    base[metric], candidate[metric], base["clip"], mask, higher_is_better=True
                )
                primary_rows.append({
                    "pair": pair["id"], "seed": seed, "population": pair["primary_band"],
                    "metric": metric, **effect,
                })
            for metric in ("missed_components", "extra_components"):
                effect = _effect(base[metric], candidate[metric], base["clip"], mask)
                primary_rows.append({
                    "pair": pair["id"], "seed": seed, "population": pair["primary_band"],
                    "metric": metric, **effect,
                })
            coverage = _effect(
                _component_coverage(base), _component_coverage(candidate), base["clip"], mask,
                higher_is_better=True,
            )
            primary_rows.append({
                "pair": pair["id"], "seed": seed, "population": pair["primary_band"],
                "metric": "component_coverage", **coverage,
            })

    frontier_deltas = {}
    frontier_clips = None
    for pair in config["frontier_pairs"]:
        steps = int(pair.get("steps", 30000 if pair["regime"] == "B16x30k" else 60000))
        base = np.load(run_paths[(pair["baseline"], pair["seed"], steps)])
        candidate = np.load(run_paths[(pair["candidate"], pair["seed"], steps)])
        _aligned(base, candidate)
        if frontier_clips is None:
            frontier_clips = base["clip"]
        elif not np.array_equal(frontier_clips, base["clip"]):
            raise ValueError("frontier record populations are not aligned")
        frontier_deltas[pair["id"]] = base["fg"] - candidate["fg"]
    frontier = simultaneous_cluster_intervals(
        frontier_deltas, frontier_clips, n_boot=10000, seed=0, confidence=0.95
    )
    result = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "config_sha256": sha256_file(args.config),
        "effect_convention": config["evaluation"]["effect_convention"],
        "primary": primary_rows,
        "frontier": frontier,
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, _markdown(result).encode("utf-8"))

    run_id = f"e2-analysis-{manifest['dataset_id'].split('-')[-1]}"
    wb.init(
        run_id, project=args.wandb_project, run_id=run_id, resume="never",
        group=args.wandb_group, job_type="confirmation-analysis", required=True,
        tags=["loopmom", "untouched-confirmation", "analysis"],
        config={"config_sha256": result["config_sha256"],
                "manifest_sha256": result["dataset_manifest_sha256"]},
    )
    wb.summary({
        "analysis/primary_rows": len(primary_rows),
        "analysis/frontier_contrasts": len(frontier_deltas),
        "analysis/max_stat_critical": frontier["critical_value"],
    })
    wb.artifact(
        run_id, "confirmation-analysis",
        [args.out_json, args.out_md, args.config, args.manifest],
        metadata={"dataset_id": manifest["dataset_id"]}, aliases=["latest"],
    )
    wb.finish()
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
