"""Seed-resolved E4 head-by-package factorial analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import wb
from hvp_rev.analysis import head_package_interaction, paired_effect
from hvp_rev.provenance import atomic_write_bytes, atomic_write_json, sha256_file


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
    # clip/h/hard define the paired record geometry used by every E4 estimand. The oldest
    # hierarchical rollout files predate the arm-independent dt/seen instrumentation, so those
    # labels are corroborating alignment checks when available rather than required inputs.
    for index, run in enumerate(runs):
        missing = {"clip", "h", "hard", "fg"} - set(run)
        if missing:
            raise ValueError(f"factorial cell {index} is missing required fields: {sorted(missing)}")
        expected_shape = run["clip"].shape
        for key in ("h", "hard", "fg"):
            if run[key].shape != expected_shape:
                raise ValueError(f"factorial cell {index} has unaligned {key} shape")
    for key in ("clip", "h", "hard"):
        first = runs[0][key]
        if any(not np.array_equal(first, run[key]) for run in runs[1:]):
            raise ValueError(f"factorial cells are not paired on {key}")
    for key in ("dt", "seen"):
        available = [run[key] for run in runs if key in run]
        if len(available) > 1 and any(
            not np.array_equal(available[0], values) for values in available[1:]
        ):
            raise ValueError(f"available factorial labels are not paired on {key}")


def _markdown(result: dict) -> str:
    lines = [
        "# E4 head × package factorial", "",
        "> Positive package effects favor temporally sparse full-resolution context. "
        "Negative interactions mean regression shifts the package contrast toward spatial coarsening.",
        "", "| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |",
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
    lines += [
        "", f"Consistent negative interaction across all seeds: **{result['consistent_negative_interaction']}**.",
        f"E4b training-seed heterogeneity trigger: **{result['e4b_trigger']['fired']}** "
        f"({result['e4b_trigger']['reason']}).", "",
        "Clip-bootstrap intervals condition on each trained model. The E4b trigger uses variation "
        "across training seeds, not those intervals.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/regression_factorial_seed2.json")
    parser.add_argument("--run-root", default="artifacts/training")
    parser.add_argument("--out-json", default="artifacts/e4_head_package/results.json")
    parser.add_argument("--out-md", default="docs/confirmation/E4_RESULTS.md")
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", default="e4-head-package-seed2")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cells = config["cells"]
    rows = []
    regression_effects = []
    for seed in (0, 1, 2):
        diffusion_temporal = _load(_legacy(cells["diffusion_temporal"], seed))
        diffusion_spatial = _load(_legacy(cells["diffusion_spatial"], seed))
        if seed < 2:
            regression_temporal = _load(_legacy(cells["regression_temporal"], seed))
            regression_spatial = _load(_legacy(cells["regression_spatial"], seed))
        else:
            regression_temporal = _load(_v2(Path(args.run_root), config["study"], cells["regression_temporal"], seed))
            regression_spatial = _load(_v2(Path(args.run_root), config["study"], cells["regression_spatial"], seed))
        _aligned(diffusion_temporal, diffusion_spatial, regression_temporal, regression_spatial)
        clips = diffusion_temporal["clip"]
        diffusion = paired_effect(
            diffusion_spatial["fg"], diffusion_temporal["fg"], clips,
            lower_is_better=True, n_boot=10000, seed=0,
        )
        regression = paired_effect(
            regression_spatial["fg"], regression_temporal["fg"], clips,
            lower_is_better=True, n_boot=10000, seed=0,
        )
        interaction = head_package_interaction(
            diffusion_temporal["fg"], diffusion_spatial["fg"],
            regression_temporal["fg"], regression_spatial["fg"], clips,
            n_boot=10000, seed=0,
        )
        regression_effects.append(regression["mean"])
        rows.append({"seed": seed, "diffusion": diffusion, "regression": regression,
                     "interaction": interaction})

    regression_effects_array = np.asarray(regression_effects)
    sign_trigger = bool(
        (regression_effects_array >= 0.005).any()
        and (regression_effects_array <= -0.005).any()
    )
    seed_sd = float(regression_effects_array.std(ddof=1))
    sd_trigger = seed_sd > 0.010
    reasons = []
    if sign_trigger:
        reasons.append("opposite material seed effects")
    if sd_trigger:
        reasons.append(f"across-seed SD {seed_sd:.4f} > 0.010")
    result = {
        "schema_version": 1,
        "study": config["study"],
        "config_sha256": sha256_file(args.config),
        "effect_convention": config["evaluation"]["effect_convention"],
        "seeds": rows,
        "consistent_negative_interaction": all(row["interaction"]["mean"] < 0 for row in rows),
        "e4b_trigger": {
            "fired": sign_trigger or sd_trigger,
            "reason": "; ".join(reasons) if reasons else "no preregistered heterogeneity threshold crossed",
            "regression_effects": regression_effects,
            "sample_sd": seed_sd,
        },
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, _markdown(result).encode("utf-8"))

    run_id = f"e4-analysis-{result['config_sha256'][:12]}"
    wb.init(
        run_id, project=args.wandb_project, run_id=run_id, resume="never",
        group=args.wandb_group, job_type="factorial-analysis", required=True,
        tags=["loopmom", "head-package", "training-seed-analysis"],
        config={"config_sha256": result["config_sha256"]},
    )
    wb.summary({
        "analysis/consistent_negative_interaction": int(result["consistent_negative_interaction"]),
        "analysis/e4b_trigger": int(result["e4b_trigger"]["fired"]),
        "analysis/regression_seed_sd": seed_sd,
    })
    wb.artifact(run_id, "factorial-analysis", [args.out_json, args.out_md, args.config], aliases=["latest"])
    wb.finish()
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
