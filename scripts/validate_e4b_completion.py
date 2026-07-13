"""Independent immutable-artifact and statistics completion gate for E4b."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file, sha256_value

T_CRITICAL_DF4_975 = 2.7764451051977987


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def _legacy(arm: str, seed: int) -> Path:
    return Path(f"logs_rhcp/rhvd_eval_{arm}_s{seed}.npz")


def _find_v2(root: Path, study: str, arm: str, seed: int) -> tuple[Path, dict]:
    matches = []
    for spec_path in root.rglob("run_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") == study and spec.get("arm") == arm and int(spec.get("seed")) == seed:
            matches.append((spec_path.parent, spec))
    if len(matches) != 1:
        raise ValueError(f"expected one RunSpec for {study}/{arm}/s{seed}: {matches}")
    run_dir, spec = matches[0]
    if sha256_value(spec)[:16] != run_dir.name:
        raise ValueError(f"RunSpec identity mismatch: {run_dir}")
    if (run_dir / "COMPLETE").read_text(encoding="utf-8").strip() != run_dir.name:
        raise ValueError(f"invalid COMPLETE marker: {run_dir}")
    return run_dir, spec


def _verify_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_spec_hash") != run_dir.name:
        raise ValueError(f"manifest identity mismatch: {run_dir}")
    for item in manifest["artifacts"]:
        path = run_dir / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"manifest artifact mismatch: {path}")
    return {
        "run_spec_hash": run_dir.name,
        "artifact_count": len(manifest["artifacts"]),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _alignment(*runs: dict[str, np.ndarray]) -> None:
    for key in ("clip", "h", "hard"):
        if any(key not in run for run in runs):
            raise ValueError(f"missing pairing field {key}")
        if any(not np.array_equal(runs[0][key], run[key]) for run in runs[1:]):
            raise ValueError(f"factorial cells not paired on {key}")
    shape = runs[0]["clip"].shape
    for run in runs:
        if run["fg"].shape != shape or not np.isfinite(run["fg"]).all():
            raise ValueError("foreground metric is incomplete or unaligned")
    for key in ("dt", "seen"):
        available = [run[key] for run in runs if key in run]
        if len(available) > 1 and any(
            not np.array_equal(available[0], values) for values in available[1:]
        ):
            raise ValueError(f"available labels not paired on {key}")


def _interval(values: np.ndarray, clips: np.ndarray) -> dict:
    unique = np.unique(clips)
    sums = np.asarray([values[clips == clip].sum() for clip in unique], dtype=np.float64)
    counts = np.asarray([(clips == clip).sum() for clip in unique], dtype=np.float64)
    picks = np.random.RandomState(0).randint(0, len(unique), (10000, len(unique)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
    }


def _seed_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRITICAL_DF4_975 * sd / math.sqrt(5)
    return {"mean": mean, "sample_sd": sd, "lo": mean - half, "hi": mean + half}


def _assert_stats(observed: dict, expected: dict, label: str) -> None:
    for key in ("mean", "lo", "hi"):
        if not np.isclose(observed[key], expected[key], rtol=0.0, atol=1e-15):
            raise ValueError(f"{label} {key} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/regression_factorial_seeds3_4.json")
    parser.add_argument("--results", default="artifacts/e4b_head_package/results.json")
    parser.add_argument("--run-root", default="artifacts/training")
    parser.add_argument("--gpu-ledger", default="artifacts/e4b_head_package/gpu_hours.jsonl")
    parser.add_argument("--out", default="artifacts/e4b_head_package/completion_validation.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    config_hash = sha256_file(args.config)
    if results.get("config_sha256") != config_hash:
        raise ValueError("results are not bound to the frozen config")

    root = Path(args.run_root)
    cells = config["cells"]
    new_runs = {}
    manifests = {}
    for declared in config["new_training_runs"]:
        key = (declared["arm"], int(declared["seed"]))
        run_dir, spec = _find_v2(root, config["study"], *key)
        frozen = {
            "steps": spec["optimizer_updates"], "batch": spec["batch_size"],
            "lr": spec["optimizer"]["lr"], "warmup": spec["optimizer"]["warmup_updates"],
            "ema": spec["ema"]["decay"], "precision": spec["precision"],
        }
        if any(frozen[name] != declared[name] for name in frozen):
            raise ValueError(f"frozen RunSpec mismatch: {key}")
        new_runs[key] = run_dir
        manifests[f"{key[0]}-s{key[1]}"] = _verify_manifest(run_dir)

    ledger = [
        json.loads(line)
        for line in Path(args.gpu_ledger).read_text(encoding="utf-8-sig").splitlines()
        if line
    ]
    total_gpu_hours = float(sum(row["gpu_hours"] for row in ledger))
    if total_gpu_hours > config["budget"]["authorized_max_gpu_hours"]:
        raise ValueError("E4b exceeded its 36-hour replacement allocation")
    for row in ledger:
        if row["gpu_hours"] > row["attempt_authorized_gpu_hours"]:
            raise ValueError(f"attempt exceeded authorization: {row['arm']}/s{row['seed']}")
    for key in new_runs:
        if not any(row["arm"] == key[0] and int(row["seed"]) == key[1]
                   and row["exit_code"] == 0 for row in ledger):
            raise ValueError(f"no successful GPU-ledger row: {key}")

    independent = []
    input_artifacts = []
    effects = {"diffusion": [], "regression": [], "interaction": []}
    for seed in range(5):
        if seed <= 2:
            paths = {
                "diffusion_temporal": _legacy(cells["diffusion_temporal"], seed),
                "diffusion_spatial": _legacy(cells["diffusion_spatial"], seed),
            }
        else:
            paths = {
                name: new_runs[(arm, seed)] / "metrics" / "rollout_complete.npz"
                for name, arm in cells.items()
            }
        if seed < 2:
            paths.update({
                "regression_temporal": _legacy(cells["regression_temporal"], seed),
                "regression_spatial": _legacy(cells["regression_spatial"], seed),
            })
        elif seed == 2:
            for name in ("regression_temporal", "regression_spatial"):
                run_dir, _ = _find_v2(
                    root, config["prior_studies"]["seed_2_regression"], cells[name], seed,
                )
                paths[name] = run_dir / "metrics" / "rollout_complete.npz"
        for cell, path in paths.items():
            input_artifacts.append({
                "seed": seed, "cell": cell, "path": path.as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
        loaded = {name: _load(path) for name, path in paths.items()}
        _alignment(*loaded.values())
        clips = loaded["diffusion_temporal"]["clip"]
        diffusion = _interval(
            loaded["diffusion_spatial"]["fg"] - loaded["diffusion_temporal"]["fg"], clips,
        )
        regression = _interval(
            loaded["regression_spatial"]["fg"] - loaded["regression_temporal"]["fg"], clips,
        )
        interaction = _interval(
            (loaded["regression_spatial"]["fg"] - loaded["regression_temporal"]["fg"])
            - (loaded["diffusion_spatial"]["fg"] - loaded["diffusion_temporal"]["fg"]),
            clips,
        )
        row = results["seeds"][seed]
        for name, expected in (("diffusion", diffusion), ("regression", regression),
                               ("interaction", interaction)):
            _assert_stats(row[name], expected, f"seed {seed} {name}")
            effects[name].append(expected["mean"])
        independent.append({
            "seed": seed, "records": int(len(clips)), "clips": int(len(np.unique(clips))),
            "diffusion": diffusion, "regression": regression, "interaction": interaction,
        })

    seed_summaries = {name: _seed_summary(values) for name, values in effects.items()}
    for name, expected in seed_summaries.items():
        observed = results["training_seed_summaries"][name]
        for key in ("mean", "sample_sd", "lo", "hi"):
            if not np.isclose(observed[key], expected[key], rtol=0.0, atol=1e-15):
                raise ValueError(f"across-seed {name} {key} mismatch")
    robust = all(value < 0 for value in effects["interaction"]) and seed_summaries["interaction"]["hi"] < 0
    if robust != results["decision"]["robust_negative_head_shift"]:
        raise ValueError("registered robust-head-shift decision mismatch")

    report = {
        "schema_version": 1, "status": "PASS", "study": config["study"],
        "config_sha256": config_hash, "results_sha256": sha256_file(args.results),
        "gpu_hours": total_gpu_hours,
        "authorized_max_gpu_hours": config["budget"]["authorized_max_gpu_hours"],
        "manifests": manifests, "input_artifacts": input_artifacts,
        "independent_statistics": independent,
        "training_seed_summaries": seed_summaries,
        "robust_negative_head_shift": robust,
    }
    atomic_write_json(args.out, report)
    print(f"E4B_COMPLETION_PASS: {args.out}")


if __name__ == "__main__":
    main()
