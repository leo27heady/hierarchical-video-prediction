"""Independent completion and statistics gate for the E4 seed-2 factorial."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file, sha256_value


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def _legacy(arm: str, seed: int) -> Path:
    return Path(f"logs_rhcp/rhvd_eval_{arm}_s{seed}.npz")


def _v2(root: Path, study: str, arm: str, seed: int) -> tuple[Path, dict]:
    matches = []
    for spec_path in root.rglob("run_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") == study and spec.get("arm") == arm and int(spec.get("seed")) == seed:
            matches.append((spec_path.parent, spec))
    if len(matches) != 1:
        raise ValueError(f"expected one v2 run for {study}/{arm}/seed-{seed}: {matches}")
    run_dir, spec = matches[0]
    run_hash = run_dir.name
    if (run_dir / "COMPLETE").read_text(encoding="utf-8").strip() != run_hash:
        raise ValueError(f"invalid COMPLETE marker for {run_dir}")
    if sha256_value(spec)[:16] != run_hash:
        raise ValueError(f"RunSpec hash mismatch for {run_dir}")
    return run_dir, spec


def _verify_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_spec_hash") != run_dir.name:
        raise ValueError(f"manifest RunSpec mismatch for {run_dir}")
    for item in manifest["artifacts"]:
        path = run_dir / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"byte-count mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return {
        "run_spec_hash": run_dir.name,
        "artifact_count": len(manifest["artifacts"]),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _check_alignment(*runs: dict[str, np.ndarray]) -> None:
    for key in ("clip", "h", "hard"):
        if any(key not in run for run in runs):
            raise ValueError(f"missing required pairing field {key}")
        if any(not np.array_equal(runs[0][key], run[key]) for run in runs[1:]):
            raise ValueError(f"factorial cells are not paired on {key}")
    shape = runs[0]["clip"].shape
    for run in runs:
        if run["fg"].shape != shape or not np.isfinite(run["fg"]).all():
            raise ValueError("foreground metric is incomplete or unaligned")
    for key in ("dt", "seen"):
        available = [run[key] for run in runs if key in run]
        if len(available) > 1 and any(
            not np.array_equal(available[0], values) for values in available[1:]
        ):
            raise ValueError(f"available labels are not paired on {key}")


def _cluster_interval(values: np.ndarray, clips: np.ndarray) -> dict:
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


def _assert_result(observed: dict, expected: dict, label: str) -> None:
    for key in ("mean", "lo", "hi"):
        if not np.isclose(observed[key], expected[key], rtol=0.0, atol=1e-15):
            raise ValueError(f"{label} {key} mismatch: {observed[key]} != {expected[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/regression_factorial_seed2.json")
    parser.add_argument("--results", default="artifacts/e4_head_package/results.json")
    parser.add_argument("--run-root", default="artifacts/training")
    parser.add_argument("--gpu-ledger", default="artifacts/e4_head_package/gpu_hours.jsonl")
    parser.add_argument("--out", default="artifacts/e4_head_package/completion_validation.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    config_hash = sha256_file(args.config)
    if results.get("config_sha256") != config_hash:
        raise ValueError("analysis result is not bound to the frozen E4 config")

    cells = config["cells"]
    run_root = Path(args.run_root)
    v2_runs = {}
    manifests = {}
    for arm in (cells["regression_spatial"], cells["regression_temporal"]):
        run_dir, spec = _v2(run_root, config["study"], arm, 2)
        declared = next(item for item in config["new_training_runs"] if item["arm"] == arm)
        frozen_values = {
            "seed": spec["seed"],
            "steps": spec["optimizer_updates"],
            "batch": spec["batch_size"],
            "lr": spec["optimizer"]["lr"],
            "warmup": spec["optimizer"]["warmup_updates"],
            "ema": spec["ema"]["decay"],
            "precision": spec["precision"],
        }
        for key, value in frozen_values.items():
            if value != declared[key]:
                raise ValueError(f"frozen RunSpec mismatch for {arm}: {key}")
        v2_runs[arm] = run_dir
        manifests[arm] = _verify_manifest(run_dir)

    ledger = [
        json.loads(line)
        for line in Path(args.gpu_ledger).read_text(encoding="utf-8-sig").splitlines()
        if line
    ]
    if len(ledger) != 2 or {row["arm"] for row in ledger} != set(v2_runs):
        raise ValueError("GPU ledger does not contain exactly the two frozen E4 cells")
    for row in ledger:
        if row["exit_code"] != 0 or row["gpu_hours"] > row["authorized_max_gpu_hours"]:
            raise ValueError(f"failed or over-budget GPU ledger row: {row['arm']}")
    total_gpu_hours = float(sum(row["gpu_hours"] for row in ledger))
    if total_gpu_hours > config["budget"]["authorized_max_gpu_hours"]:
        raise ValueError("E4 exceeded its authorized study maximum")

    independent = []
    input_artifacts = []
    for seed in (0, 1, 2):
        paths = {
            "diffusion_temporal": _legacy(cells["diffusion_temporal"], seed),
            "diffusion_spatial": _legacy(cells["diffusion_spatial"], seed),
        }
        if seed < 2:
            paths.update({
                "regression_temporal": _legacy(cells["regression_temporal"], seed),
                "regression_spatial": _legacy(cells["regression_spatial"], seed),
            })
        else:
            paths.update({
                "regression_temporal": v2_runs[cells["regression_temporal"]]
                / "metrics" / "rollout_complete.npz",
                "regression_spatial": v2_runs[cells["regression_spatial"]]
                / "metrics" / "rollout_complete.npz",
            })
        for cell, path in paths.items():
            input_artifacts.append({
                "seed": seed,
                "cell": cell,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        dt = _load(paths["diffusion_temporal"])
        ds = _load(paths["diffusion_spatial"])
        rt = _load(paths["regression_temporal"])
        rs = _load(paths["regression_spatial"])
        _check_alignment(dt, ds, rt, rs)
        clips = dt["clip"]
        diffusion = _cluster_interval(ds["fg"] - dt["fg"], clips)
        regression = _cluster_interval(rs["fg"] - rt["fg"], clips)
        interaction = _cluster_interval((rs["fg"] - rt["fg"]) - (ds["fg"] - dt["fg"]), clips)
        row = results["seeds"][seed]
        if row["seed"] != seed:
            raise ValueError("analysis seed rows are out of order")
        _assert_result(row["diffusion"], diffusion, f"seed {seed} diffusion")
        _assert_result(row["regression"], regression, f"seed {seed} regression")
        _assert_result(row["interaction"], interaction, f"seed {seed} interaction")
        independent.append({
            "seed": seed,
            "records": int(len(clips)),
            "clips": int(len(np.unique(clips))),
            "diffusion": diffusion,
            "regression": regression,
            "interaction": interaction,
        })

    regression_effects = np.asarray([row["regression"]["mean"] for row in independent])
    seed_sd = float(regression_effects.std(ddof=1))
    fired = bool(
        ((regression_effects >= 0.005).any() and (regression_effects <= -0.005).any())
        or seed_sd > 0.010
    )
    if fired != results["e4b_trigger"]["fired"] or not np.isclose(
        seed_sd, results["e4b_trigger"]["sample_sd"], rtol=0.0, atol=1e-15
    ):
        raise ValueError("E4b training-seed trigger mismatch")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "study": config["study"],
        "config_sha256": config_hash,
        "results_sha256": sha256_file(args.results),
        "gpu_hours": total_gpu_hours,
        "authorized_max_gpu_hours": config["budget"]["authorized_max_gpu_hours"],
        "manifests": manifests,
        "input_artifacts": input_artifacts,
        "independent_statistics": independent,
        "e4b_trigger": {"fired": fired, "sample_sd": seed_sd},
    }
    atomic_write_json(args.out, report)
    print(f"E4_COMPLETION_PASS: {args.out}")


if __name__ == "__main__":
    main()
