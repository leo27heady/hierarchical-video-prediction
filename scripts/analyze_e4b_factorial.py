"""Five-training-seed E4b head-by-package analysis."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import wb
from hvp_rev.analysis import head_package_interaction, paired_effect
from hvp_rev.provenance import atomic_write_bytes, atomic_write_json, sha256_file

T_CRITICAL_DF4_975 = 2.7764451051977987


def _legacy(arm: str, seed: int) -> Path:
    path = Path(f"logs_rhcp/rhvd_eval_{arm}_s{seed}.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _v2(root: Path, study: str, arm: str, seed: int) -> Path:
    found = []
    for spec_path in root.rglob("run_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") == study and spec.get("arm") == arm and int(spec.get("seed")) == seed:
            run_dir = spec_path.parent
            if (run_dir / "COMPLETE").is_file():
                found.append(run_dir / "metrics" / "rollout_complete.npz")
    if len(found) != 1 or not found[0].is_file():
        raise RuntimeError(f"expected one completed v2 result for {study}/{arm}/seed-{seed}: {found}")
    return found[0]


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def _aligned(*runs: dict[str, np.ndarray]) -> None:
    if not runs:
        raise ValueError("at least one factorial cell is required")
    for index, run in enumerate(runs):
        missing = {"clip", "h", "hard", "fg"} - set(run)
        if missing:
            raise ValueError(f"factorial cell {index} missing required fields: {sorted(missing)}")
        shape = run["clip"].shape
        if any(run[key].shape != shape for key in ("h", "hard", "fg")):
            raise ValueError(f"factorial cell {index} has unaligned record fields")
    for key in ("clip", "h", "hard"):
        if any(not np.array_equal(runs[0][key], run[key]) for run in runs[1:]):
            raise ValueError(f"factorial cells are not paired on {key}")
    for key in ("dt", "seen"):
        available = [run[key] for run in runs if key in run]
        if len(available) > 1 and any(
            not np.array_equal(available[0], values) for values in available[1:]
        ):
            raise ValueError(f"available factorial labels are not paired on {key}")


def training_seed_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,) or not np.isfinite(array).all():
        raise ValueError("training-seed summary requires five finite effects")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half_width = T_CRITICAL_DF4_975 * sd / math.sqrt(len(array))
    return {
        "mean": mean,
        "sample_sd": sd,
        "lo": mean - half_width,
        "hi": mean + half_width,
        "seeds": 5,
        "df": 4,
        "method": "two-sided 95% Student-t interval across training seeds",
    }


def _paths(config: dict, root: Path, seed: int) -> dict[str, Path]:
    cells = config["cells"]
    if seed <= 2:
        paths = {
            "diffusion_temporal": _legacy(cells["diffusion_temporal"], seed),
            "diffusion_spatial": _legacy(cells["diffusion_spatial"], seed),
        }
    else:
        paths = {
            name: _v2(root, config["study"], arm, seed)
            for name, arm in cells.items()
        }
    if seed < 2:
        paths.update({
            "regression_temporal": _legacy(cells["regression_temporal"], seed),
            "regression_spatial": _legacy(cells["regression_spatial"], seed),
        })
    elif seed == 2:
        paths.update({
            "regression_temporal": _v2(
                root, config["prior_studies"]["seed_2_regression"],
                cells["regression_temporal"], seed,
            ),
            "regression_spatial": _v2(
                root, config["prior_studies"]["seed_2_regression"],
                cells["regression_spatial"], seed,
            ),
        })
    return paths


def _markdown(result: dict) -> str:
    lines = [
        "# E4b five-seed head × package factorial", "",
        "> Positive package effects favor temporally sparse full-resolution context. "
        "Negative interactions mean regression shifts toward spatial coarsening.", "",
        "| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['diffusion']['mean']:+.4f} "
            f"[{row['diffusion']['lo']:+.4f}, {row['diffusion']['hi']:+.4f}] | "
            f"{row['regression']['mean']:+.4f} "
            f"[{row['regression']['lo']:+.4f}, {row['regression']['hi']:+.4f}] | "
            f"{row['interaction']['mean']:+.4f} "
            f"[{row['interaction']['lo']:+.4f}, {row['interaction']['hi']:+.4f}] |"
        )
    lines += ["", "## Across-training-seed summaries", ""]
    for name, summary in result["training_seed_summaries"].items():
        lines.append(
            f"- {name.replace('_', ' ')}: {summary['mean']:+.4f} "
            f"[{summary['lo']:+.4f}, {summary['hi']:+.4f}], SD {summary['sample_sd']:.4f}."
        )
    lines += [
        "", f"Robust negative head shift: **{result['decision']['robust_negative_head_shift']}**.",
        f"Regression material-sign counts: **{result['decision']['regression_sign_counts']}**.",
        "", "The development pool is adaptively used. Clip intervals condition on each trained model; "
        "the Student-t summaries use training seeds as units.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/regression_factorial_seeds3_4.json")
    parser.add_argument("--run-root", default="artifacts/training")
    parser.add_argument("--out-json", default="artifacts/e4b_head_package/results.json")
    parser.add_argument("--out-md", default="docs/confirmation/E4B_RESULTS.md")
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", default="e4b-head-package-seeds3-4")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = []
    effects = {"diffusion": [], "regression": [], "interaction": []}
    input_artifacts = []
    for seed in range(5):
        paths = _paths(config, Path(args.run_root), seed)
        for cell, path in paths.items():
            input_artifacts.append({
                "seed": seed, "cell": cell, "path": path.as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
        loaded = {name: _load(path) for name, path in paths.items()}
        _aligned(*loaded.values())
        clips = loaded["diffusion_temporal"]["clip"]
        diffusion = paired_effect(
            loaded["diffusion_spatial"]["fg"], loaded["diffusion_temporal"]["fg"],
            clips, lower_is_better=True, n_boot=10000, seed=0,
        )
        regression = paired_effect(
            loaded["regression_spatial"]["fg"], loaded["regression_temporal"]["fg"],
            clips, lower_is_better=True, n_boot=10000, seed=0,
        )
        interaction = head_package_interaction(
            loaded["diffusion_temporal"]["fg"], loaded["diffusion_spatial"]["fg"],
            loaded["regression_temporal"]["fg"], loaded["regression_spatial"]["fg"],
            clips, n_boot=10000, seed=0,
        )
        for name, value in (("diffusion", diffusion), ("regression", regression),
                            ("interaction", interaction)):
            effects[name].append(value["mean"])
        rows.append({"seed": seed, "diffusion": diffusion, "regression": regression,
                     "interaction": interaction})

    summaries = {name: training_seed_summary(values) for name, values in effects.items()}
    regression = np.asarray(effects["regression"])
    material = {
        "temporal": int((regression >= 0.005).sum()),
        "spatial": int((regression <= -0.005).sum()),
        "small": int((np.abs(regression) < 0.005).sum()),
    }
    robust_shift = bool(
        all(value < 0 for value in effects["interaction"])
        and summaries["interaction"]["hi"] < 0
    )
    result = {
        "schema_version": 1,
        "study": config["study"],
        "config_sha256": sha256_file(args.config),
        "effect_convention": config["evaluation"]["effect_convention"],
        "seeds": rows,
        "input_artifacts": input_artifacts,
        "training_seed_summaries": summaries,
        "decision": {
            "robust_negative_head_shift": robust_shift,
            "all_interactions_negative": all(value < 0 for value in effects["interaction"]),
            "interaction_seed_interval_below_zero": summaries["interaction"]["hi"] < 0,
            "regression_seed_stable": material["temporal"] == 5 or material["spatial"] == 5,
            "regression_sign_counts": material,
            "terminal_rule": "No seeds 5+ authorized regardless of outcome.",
        },
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, _markdown(result).encode("utf-8"))

    run_id = f"e4b-analysis-{result['config_sha256'][:12]}"
    wb.init(
        run_id, project=args.wandb_project, run_id=run_id, resume="allow",
        group=args.wandb_group, job_type="factorial-analysis", required=True,
        tags=["loopmom", "head-package", "training-seed-analysis", "e4b"],
        config={"config_sha256": result["config_sha256"]},
    )
    wb.summary({
        "analysis/robust_negative_head_shift": int(robust_shift),
        "analysis/interaction_seed_mean": summaries["interaction"]["mean"],
        "analysis/interaction_seed_sd": summaries["interaction"]["sample_sd"],
        "analysis/regression_temporal_signs": material["temporal"],
        "analysis/regression_spatial_signs": material["spatial"],
    })
    wb.artifact(run_id, "factorial-analysis", [args.out_json, args.out_md, args.config], aliases=["latest"])
    wb.finish()
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
