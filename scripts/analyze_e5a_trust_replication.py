"""Registered E5a analysis: early self-fed benefit followed by late inversion."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import wb
from hvp_rev.provenance import atomic_write_bytes, atomic_write_json, sha256_file


T_CRITICAL_DF2_975 = 4.302652729696142


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def _find_v2(root: Path, study: str, arm: str, seed: int) -> Path:
    found = []
    for spec_path in root.rglob("run_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") == study and spec.get("arm") == arm and int(spec.get("seed")) == seed:
            run_dir = spec_path.parent
            if (run_dir / "COMPLETE").is_file():
                found.append(run_dir / "metrics" / "rollout_complete.npz")
    if len(found) != 1 or not found[0].is_file():
        raise RuntimeError(f"expected one completed v2 result for {study}/{arm}/s{seed}: {found}")
    return found[0]


def _paths(config: dict, root: Path, seed: int) -> tuple[Path, Path]:
    cells = config["cells"]
    if seed == 0:
        prior = config["prior_evidence"]["seed0_rollout"]
        return Path(prior["long_reach_metrics"]), Path(prior["comparator_metrics"])
    return (
        _find_v2(root, config["study"], cells["long_reach_hierarchy"], seed),
        _find_v2(root, config["study"], cells["contiguous_fine_comparator"], seed),
    )


def _aligned(hierarchy: dict[str, np.ndarray], comparator: dict[str, np.ndarray]) -> None:
    required = {"clip", "h", "hard", "dt", "seen", "fg"}
    for name, run in (("hierarchy", hierarchy), ("comparator", comparator)):
        missing = required - set(run)
        if missing:
            raise ValueError(f"{name} is missing pairing fields: {sorted(missing)}")
        shape = run["clip"].shape
        if any(run[key].shape != shape for key in required):
            raise ValueError(f"{name} record arrays are not aligned")
    for key in ("clip", "h", "hard", "dt", "seen"):
        if not np.array_equal(hierarchy[key], comparator[key]):
            raise ValueError(f"pair is not aligned on {key}")
    if not np.isfinite(hierarchy["fg"]).all() or not np.isfinite(comparator["fg"]).all():
        raise ValueError("primary foreground metric contains non-finite values")


def _nonempty_mask(run: dict[str, np.ndarray], test_path: str, warm: int, threshold: float) -> np.ndarray:
    frames = np.load(test_path, mmap_mode="r")
    cutoff = threshold * 255.0
    return np.asarray([
        frames[int(clip), warm + int(horizon) - 1].max() > cutoff
        for clip, horizon in zip(run["clip"], run["h"])
    ], dtype=bool)


def _interval(delta: np.ndarray, clips: np.ndarray, *, n_boot: int = 10000, seed: int = 0) -> dict:
    finite = np.isfinite(delta)
    delta, clips = delta[finite].astype(np.float64), clips[finite]
    unique = np.unique(clips)
    if not len(delta) or not len(unique):
        raise ValueError("empty or non-finite registered analysis cell")
    sums = np.asarray([delta[clips == clip].sum() for clip in unique], dtype=np.float64)
    counts = np.asarray([(clips == clip).sum() for clip in unique], dtype=np.float64)
    picks = np.random.RandomState(seed).randint(0, len(unique), (n_boot, len(unique)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "mean": float(delta.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "records": int(len(delta)),
        "clips": int(len(unique)),
        "bootstrap_replicates": int(n_boot),
    }


def _effect(
    hierarchy: dict[str, np.ndarray], comparator: dict[str, np.ndarray], mask: np.ndarray,
    metric: str = "fg", *, higher_is_better: bool = False,
) -> dict:
    if metric not in hierarchy or metric not in comparator:
        raise ValueError(f"complete metric missing: {metric}")
    # Positive always favors the long-reach hierarchy.
    delta = (
        hierarchy[metric][mask] - comparator[metric][mask]
        if higher_is_better else comparator[metric][mask] - hierarchy[metric][mask]
    )
    return _interval(delta, hierarchy["clip"][mask])


def _seed_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("E5a seed summary requires exactly three finite seed effects")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRITICAL_DF2_975 * sd / math.sqrt(3)
    return {
        "mean": mean, "sample_sd": sd, "lo": mean - half, "hi": mean + half,
        "seeds": 3, "df": 2,
        "method": "descriptive two-sided 95% Student-t interval across three training seeds",
    }


def _horizon_table(hierarchy: dict, comparator: dict, nonempty: np.ndarray) -> list[dict]:
    rows = []
    horizon_bins = ((1, 8), (9, 24), (25, 60), (61, 120), (121, 192), (193, 256))
    hardness_bins = (
        ("easy", lambda x: x < 0.1),
        ("mid", lambda x: (x >= 0.1) & (x < 0.5)),
        ("HARD", lambda x: x >= 0.5),
    )
    for lo, hi in horizon_bins:
        for label, hard_fn in hardness_bins:
            mask = nonempty & (hierarchy["h"] >= lo) & (hierarchy["h"] <= hi)
            mask &= hard_fn(hierarchy["hard"])
            if mask.any():
                rows.append({"horizon": f"{lo}-{hi}", "hardness": label,
                             "effect": _effect(hierarchy, comparator, mask)})
    return rows


def _complete_metric_rows(hierarchy: dict, comparator: dict, windows: dict[str, np.ndarray]) -> dict:
    metrics = {
        "mse_full": False,
        "mse_union": False,
        "mse_gt_bg": False,
        "fg_precision": True,
        "fg_recall": True,
        "missed_components": False,
        "extra_components": False,
    }
    return {
        window: {
            metric: _effect(hierarchy, comparator, mask, metric, higher_is_better=higher)
            for metric, higher in metrics.items()
        }
        for window, mask in windows.items()
    }


def _decision(rows: list[dict]) -> dict:
    new = [row for row in rows if row["seed"] in (1, 2)]
    clean = [row["ci_clean_boundary"] for row in new]
    directional = [row["directional_boundary"] for row in new]
    if all(clean):
        verdict = "REPLICATED"
        e5b = "FULL_E5B_DESIGN_MAY_BE_FROZEN_AFTER_REVIEW"
    elif all(directional):
        verdict = "DIRECTIONALLY_REPLICATED"
        e5b = "REDUCED_E5B_ONLY_AFTER_NEW_FREEZE"
    elif sum(directional) == 1:
        verdict = "HETEROGENEOUS_PARTIAL"
        e5b = "FULL_E5B_NOT_AUTHORIZED"
    else:
        verdict = "NOT_REPLICATED"
        e5b = "FULL_E5B_NOT_AUTHORIZED"
    return {
        "verdict": verdict,
        "new_seed_ci_clean_count": int(sum(clean)),
        "new_seed_directional_count": int(sum(directional)),
        "e5b_gate": e5b,
        "terminal_rule": "No E5a seed beyond 2 is authorized regardless of outcome.",
    }


def _markdown(result: dict) -> str:
    lines = [
        "# E5a trust-boundary replication", "",
        "> Positive effects favor the 224-frame-reach hierarchical configuration. Primary rows "
        "exclude empty ground-truth foreground records.", "",
        "| Seed | Early h1–24 effect [95% CI] | Late h61–256 effect [95% CI] | Directional | CI-clean |",
        "|---:|---:|---:|:---:|:---:|",
    ]
    for row in result["seeds"]:
        early, late = row["primary"]["early_benefit"], row["primary"]["late_inversion"]
        lines.append(
            f"| {row['seed']} | {early['mean']:+.4f} [{early['lo']:+.4f}, {early['hi']:+.4f}] "
            f"| {late['mean']:+.4f} [{late['lo']:+.4f}, {late['hi']:+.4f}] "
            f"| {'yes' if row['directional_boundary'] else 'no'} "
            f"| {'yes' if row['ci_clean_boundary'] else 'no'} |"
        )
    lines += ["", f"Registered E5a verdict: **{result['decision']['verdict']}**.",
              f"E5b gate: **{result['decision']['e5b_gate']}**.", "",
              "## Mechanistic secondary rows", "",
              "| Seed | Early HARD | Late HARD | Answerable HARD dt>48 |",
              "|---:|---:|---:|---:|"]
    for row in result["seeds"]:
        sec = row["secondary"]
        cells = []
        for key in ("early_hard", "late_hard", "answerable_hard_dt_gt48"):
            value = sec[key]
            cells.append(f"{value['mean']:+.4f} [{value['lo']:+.4f}, {value['hi']:+.4f}]")
        lines.append(f"| {row['seed']} | {' | '.join(cells)} |")
    lines += ["", "## Across-training-seed descriptions", ""]
    for name, summary in result["training_seed_summaries"].items():
        lines.append(
            f"- {name.replace('_', ' ')}: {summary['mean']:+.4f} "
            f"[{summary['lo']:+.4f}, {summary['hi']:+.4f}], SD {summary['sample_sd']:.4f}."
        )
    lines += [
        "", "The three-seed intervals are descriptive because n=3 is small. The H-grid3 pool is "
        "adaptively used development evidence. E5a does not re-estimate the separate GT-fed "
        "single-step addressability result and does not establish a universal trust law.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/trust_replication_seeds1_2.json")
    parser.add_argument("--run-root", default="artifacts/training")
    parser.add_argument("--out-json", default="artifacts/e5a_trust_replication/results.json")
    parser.add_argument("--out-md", default="docs/confirmation/E5A_RESULTS.md")
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", default="e5a-trust-replication-seeds1-2")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows, inputs = [], []
    seed_effects = {"early_benefit": [], "late_inversion": []}
    for seed in range(3):
        hierarchy_path, comparator_path = _paths(config, Path(args.run_root), seed)
        for cell, path in (("long_reach_hierarchy", hierarchy_path),
                           ("contiguous_fine_comparator", comparator_path)):
            inputs.append({"seed": seed, "cell": cell, "path": path.as_posix(),
                           "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        hierarchy, comparator = _load(hierarchy_path), _load(comparator_path)
        _aligned(hierarchy, comparator)
        nonempty = _nonempty_mask(
            hierarchy, config["dataset"]["test_clips"],
            config["evaluation"]["warmup_frames"], config["evaluation"]["metric_threshold"],
        )
        h, hard = hierarchy["h"], hierarchy["hard"] >= 0.5
        early = nonempty & (h >= 1) & (h <= 24)
        late = nonempty & (h >= 61) & (h <= 256)
        primary = {
            "early_benefit": _effect(hierarchy, comparator, early),
            "late_inversion": _effect(hierarchy, comparator, late),
        }
        secondary = {
            "early_hard": _effect(hierarchy, comparator, early & hard),
            "late_hard": _effect(hierarchy, comparator, late & hard),
            "answerable_hard_dt_gt48": _effect(
                hierarchy, comparator,
                nonempty & hard & (hierarchy["seen"] >= 0.5) & (hierarchy["dt"] > 48),
            ),
        }
        directional = primary["early_benefit"]["mean"] > 0 and primary["late_inversion"]["mean"] < 0
        clean = primary["early_benefit"]["lo"] > 0 and primary["late_inversion"]["hi"] < 0
        for name in seed_effects:
            seed_effects[name].append(primary[name]["mean"])
        row = {
            "seed": seed, "records": int(len(h)), "nonempty_records": int(nonempty.sum()),
            "primary": primary, "secondary": secondary,
            "directional_boundary": bool(directional), "ci_clean_boundary": bool(clean),
            "horizon_by_hardness": _horizon_table(hierarchy, comparator, nonempty),
        }
        if seed in (1, 2):
            row["complete_metric_corroboration"] = _complete_metric_rows(
                hierarchy, comparator, {"early_benefit": early, "late_inversion": late},
            )
        rows.append(row)

    result = {
        "schema_version": 1, "study": config["study"],
        "config_sha256": sha256_file(args.config),
        "effect_convention": config["evaluation"]["effect_convention"],
        "seeds": rows, "input_artifacts": inputs,
        "training_seed_summaries": {
            name: _seed_summary(values) for name, values in seed_effects.items()
        },
        "decision": _decision(rows),
        "scope": config["scope_guard"],
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, _markdown(result).encode("utf-8"))

    run_id = f"e5a-analysis-{result['config_sha256'][:12]}"
    wb.init(
        run_id, project=args.wandb_project, run_id=run_id, resume="allow",
        group=args.wandb_group, job_type="trust-replication-analysis", required=True,
        tags=["hgrid3", "trust-boundary", "training-seed-analysis", "e5a"],
        config={"config_sha256": result["config_sha256"]},
    )
    wb.summary({
        "analysis/verdict": result["decision"]["verdict"],
        "analysis/new_seed_ci_clean_count": result["decision"]["new_seed_ci_clean_count"],
        "analysis/new_seed_directional_count": result["decision"]["new_seed_directional_count"],
        "analysis/early_seed_mean": result["training_seed_summaries"]["early_benefit"]["mean"],
        "analysis/late_seed_mean": result["training_seed_summaries"]["late_inversion"]["mean"],
    })
    wb.artifact(run_id, "trust-replication-analysis", [args.out_json, args.out_md, args.config],
                aliases=["latest"])
    wb.finish()
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
