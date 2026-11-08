"""Independent immutable-artifact, budget, pairing, and decision gate for E5a."""
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


def _find_v2(root: Path, study: str, arm: str, seed: int) -> tuple[Path, dict]:
    matches = []
    for spec_path in root.rglob("run_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") == study and spec.get("arm") == arm and int(spec.get("seed")) == seed:
            matches.append((spec_path.parent, spec))
    if len(matches) != 1:
        raise ValueError(f"expected one E5a RunSpec for {arm}/s{seed}: {matches}")
    run_dir, spec = matches[0]
    if sha256_value(spec)[:16] != run_dir.name:
        raise ValueError(f"RunSpec identity mismatch: {run_dir}")
    if (run_dir / "COMPLETE").read_text(encoding="utf-8").strip() != run_dir.name:
        raise ValueError(f"invalid COMPLETE marker: {run_dir}")
    return run_dir, spec


def _verify_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("run_spec_hash") != run_dir.name:
        raise ValueError(f"manifest identity mismatch: {run_dir}")
    for item in manifest["artifacts"]:
        artifact = run_dir / item["path"]
        if artifact.stat().st_size != item["bytes"] or sha256_file(artifact) != item["sha256"]:
            raise ValueError(f"manifest artifact mismatch: {artifact}")
    return {"run_spec_hash": run_dir.name, "artifact_count": len(manifest["artifacts"]),
            "manifest_sha256": sha256_file(path)}


def _interval(delta: np.ndarray, clips: np.ndarray) -> dict:
    unique = np.unique(clips)
    sums = np.asarray([delta[clips == clip].sum() for clip in unique], dtype=np.float64)
    counts = np.asarray([(clips == clip).sum() for clip in unique], dtype=np.float64)
    picks = np.random.RandomState(0).randint(0, len(unique), (10000, len(unique)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"mean": float(delta.mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "records": int(len(delta)),
            "clips": int(len(unique)), "bootstrap_replicates": 10000}


def _assert_interval(observed: dict, expected: dict, label: str) -> None:
    for key in ("mean", "lo", "hi"):
        if not np.isclose(observed[key], expected[key], rtol=0.0, atol=1e-15):
            raise ValueError(f"{label} {key} differs from independent calculation")
    for key in ("records", "clips", "bootstrap_replicates"):
        if observed[key] != expected[key]:
            raise ValueError(f"{label} {key} differs from independent calculation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/trust_replication_seeds1_2.json")
    parser.add_argument("--results", default="artifacts/e5a_trust_replication/results.json")
    parser.add_argument("--run-root", default="artifacts/training")
    parser.add_argument("--gpu-ledger", default="artifacts/e5a_trust_replication/gpu_hours.jsonl")
    parser.add_argument("--out", default="artifacts/e5a_trust_replication/completion_validation.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    config_hash = sha256_file(args.config)
    if results.get("config_sha256") != config_hash:
        raise ValueError("E5a results are not bound to the frozen config")

    root = Path(args.run_root)
    cells = config["cells"]
    run_dirs, manifests = {}, {}
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
        if spec["warmup_frames"] != 224 or spec["rollout_frames"] != 256:
            raise ValueError(f"E5a rollout geometry changed: {key}")
        if spec["evaluation_clips"] != 100 or spec["nfe"] != 16:
            raise ValueError(f"E5a evaluation protocol changed: {key}")
        data_artifacts = spec["arm_config"].get("data_artifacts", {})
        if set(data_artifacts) != {
            "train_clips", "train_actions", "train_positions",
            "test_clips", "test_actions", "test_positions",
        }:
            raise ValueError(f"E5a RunSpec does not hash geometric labels: {key}")
        if spec.get("tag") != config["wandb"]["provenance_tag"]:
            raise ValueError(f"E5a provenance tag changed: {key}")
        run_dirs[key] = run_dir
        manifests[f"{key[0]}-s{key[1]}"] = _verify_manifest(run_dir)

    ledger = [json.loads(line) for line in Path(args.gpu_ledger).read_text(
        encoding="utf-8-sig").splitlines() if line]
    total_gpu_hours = float(sum(row["gpu_hours"] for row in ledger))
    if total_gpu_hours > config["budget"]["authorized_max_gpu_hours"]:
        raise ValueError("E5a exceeded its 28-hour authorization")
    for row in ledger:
        if row["gpu_hours"] > row["attempt_authorized_gpu_hours"]:
            raise ValueError(f"attempt exceeded authorization: {row['arm']}/s{row['seed']}")
    for key in run_dirs:
        if not any(row["arm"] == key[0] and int(row["seed"]) == key[1]
                   and row["exit_code"] == 0 for row in ledger):
            raise ValueError(f"no successful GPU-ledger row for {key}")

    seed0 = config["prior_evidence"]["seed0_rollout"]
    test = np.load(config["dataset"]["test_clips"], mmap_mode="r")
    independent_rows, input_artifacts = [], []
    clean, directional = [], []
    for seed in range(3):
        if seed == 0:
            hierarchy_path = Path(seed0["long_reach_metrics"])
            comparator_path = Path(seed0["comparator_metrics"])
        else:
            hierarchy_path = run_dirs[(cells["long_reach_hierarchy"], seed)] / "metrics/rollout_complete.npz"
            comparator_path = run_dirs[(cells["contiguous_fine_comparator"], seed)] / "metrics/rollout_complete.npz"
        for cell, path in (("long_reach_hierarchy", hierarchy_path),
                           ("contiguous_fine_comparator", comparator_path)):
            input_artifacts.append({"seed": seed, "cell": cell, "path": path.as_posix(),
                                    "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        hierarchy, comparator = _load(hierarchy_path), _load(comparator_path)
        for key in ("clip", "h", "hard", "dt", "seen"):
            if not np.array_equal(hierarchy[key], comparator[key]):
                raise ValueError(f"E5a seed {seed} pair is not aligned on {key}")
        if len(hierarchy["fg"]) != 25600 or not np.isfinite(hierarchy["fg"]).all():
            raise ValueError(f"E5a seed {seed} hierarchy primary metric is incomplete")
        if len(comparator["fg"]) != 25600 or not np.isfinite(comparator["fg"]).all():
            raise ValueError(f"E5a seed {seed} comparator primary metric is incomplete")
        if seed in (1, 2):
            complete = {"mse_full", "mse_union", "mse_gt_bg", "fg_precision", "fg_recall",
                        "missed_components", "extra_components"}
            if not complete.issubset(hierarchy) or not complete.issubset(comparator):
                raise ValueError(f"E5a seed {seed} lacks complete-v2 corroborating metrics")
        nonempty = np.asarray([
            test[int(clip), 224 + int(horizon) - 1].max() > 127.5
            for clip, horizon in zip(hierarchy["clip"], hierarchy["h"])
        ])
        early = nonempty & (hierarchy["h"] <= 24)
        late = nonempty & (hierarchy["h"] >= 61)
        delta = comparator["fg"] - hierarchy["fg"]
        early_effect = _interval(delta[early], hierarchy["clip"][early])
        late_effect = _interval(delta[late], hierarchy["clip"][late])
        observed = results["seeds"][seed]
        if int(observed["seed"]) != seed:
            raise ValueError("E5a result seed ordering changed")
        _assert_interval(observed["primary"]["early_benefit"], early_effect,
                         f"seed {seed} early")
        _assert_interval(observed["primary"]["late_inversion"], late_effect,
                         f"seed {seed} late")
        this_directional = early_effect["mean"] > 0 and late_effect["mean"] < 0
        this_clean = early_effect["lo"] > 0 and late_effect["hi"] < 0
        if observed["directional_boundary"] != this_directional or observed["ci_clean_boundary"] != this_clean:
            raise ValueError(f"seed {seed} registered boundary decision mismatch")
        if seed in (1, 2):
            directional.append(this_directional); clean.append(this_clean)
        independent_rows.append({"seed": seed, "early_benefit": early_effect,
                                 "late_inversion": late_effect,
                                 "directional_boundary": this_directional,
                                 "ci_clean_boundary": this_clean})

    if all(clean):
        verdict, e5b = "REPLICATED", "FULL_E5B_DESIGN_MAY_BE_FROZEN_AFTER_REVIEW"
    elif all(directional):
        verdict, e5b = "DIRECTIONALLY_REPLICATED", "REDUCED_E5B_ONLY_AFTER_NEW_FREEZE"
    elif sum(directional) == 1:
        verdict, e5b = "HETEROGENEOUS_PARTIAL", "FULL_E5B_NOT_AUTHORIZED"
    else:
        verdict, e5b = "NOT_REPLICATED", "FULL_E5B_NOT_AUTHORIZED"
    if results["decision"]["verdict"] != verdict or results["decision"]["e5b_gate"] != e5b:
        raise ValueError("E5a terminal decision mismatch")
    if results["input_artifacts"] != input_artifacts:
        raise ValueError("E5a input artifact inventory mismatch")

    report = {
        "schema_version": 1, "status": "PASS", "study": config["study"],
        "config_sha256": config_hash, "results_sha256": sha256_file(args.results),
        "results_markdown_sha256": sha256_file("docs/confirmation/E5A_RESULTS.md"),
        "gpu_hours": total_gpu_hours,
        "authorized_max_gpu_hours": config["budget"]["authorized_max_gpu_hours"],
        "manifests": manifests, "input_artifacts": input_artifacts,
        "independent_primary_statistics": independent_rows,
        "decision": {"verdict": verdict, "e5b_gate": e5b},
        "terminal_rule": "No E5a seed beyond 2 is authorized.",
    }
    atomic_write_json(args.out, report)
    print(f"E5A_COMPLETION_PASS: {args.out}")


if __name__ == "__main__":
    main()
