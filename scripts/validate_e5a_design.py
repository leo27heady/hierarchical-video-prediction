"""Fail-closed design, data, legacy-evidence, parity, and budget gate for E5a."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import arm_registry, rhvd
from hvp_rev.provenance import atomic_write_json, sha256_file


def _load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def _interval(delta: np.ndarray, clips: np.ndarray) -> dict:
    unique = np.unique(clips)
    sums = np.asarray([delta[clips == clip].sum() for clip in unique], dtype=np.float64)
    counts = np.asarray([(clips == clip).sum() for clip in unique], dtype=np.float64)
    picks = np.random.RandomState(0).randint(0, len(unique), (10000, len(unique)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "mean": float(delta.mean()), "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)), "records": int(len(delta)),
        "clips": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/trust_replication_seeds1_2.json")
    parser.add_argument("--out", default="gates/e5a_design_validation.json")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    failures: list[str] = []
    if config.get("status") != "FROZEN":
        failures.append("E5a config is not frozen")
    runs = config.get("new_training_runs", [])
    expected = {(arm, seed) for seed in (1, 2) for arm in ("d1-56-diff", "d0-diff-long")}
    actual = {(run.get("arm"), int(run.get("seed", -1))) for run in runs}
    if len(runs) != 4 or actual != expected:
        failures.append(f"wrong E5a training cells: {sorted(actual)}")
    allocated = sum(float(run.get("max_gpu_hours", 0)) for run in runs)
    maximum = float(config.get("budget", {}).get("authorized_max_gpu_hours", 0))
    if allocated != 28.0 or maximum != 28.0:
        failures.append(f"E5a must allocate exactly 28 GPU-hours: {allocated}/{maximum}")
    if config.get("budget", {}).get("protected_reserve_allowed") is not False:
        failures.append("E5a protected-reserve prohibition is missing")
    for run in runs:
        if run.get("steps") != 40000 or run.get("batch") != 8:
            failures.append(f"{run.get('arm')}/s{run.get('seed')} is not legacy B8x40k")
        if run.get("lr") != 0.0003 or run.get("warmup") != 2000 or run.get("ema") != 0.999:
            failures.append(f"{run.get('arm')}/s{run.get('seed')} optimizer/EMA changed")
        expected_hours = 4.5 if run.get("arm") == "d1-56-diff" else 9.5
        if run.get("precision") != "bf16" or run.get("max_gpu_hours") != expected_hours:
            failures.append(f"{run.get('arm')}/s{run.get('seed')} precision/budget changed")

    registry = arm_registry.from_legacy_mapping(rhvd.ARMS)
    resolved = {arm: registry[arm].resolved() for arm in ("d1-56-diff", "d0-diff-long")}
    expected_interfaces = {
        "d1-56-diff": {"context_reach": 224, "context_tokens": 1280,
                        "target_tokens": 272, "total_tokens": 1552, "head": "diff"},
        "d0-diff-long": {"context_reach": 48, "context_tokens": 3072,
                         "target_tokens": 320, "total_tokens": 3392, "head": "diff"},
    }
    for arm, fields in expected_interfaces.items():
        for field, expected_value in fields.items():
            if resolved[arm].get(field) != expected_value:
                failures.append(f"{arm} {field} changed: {resolved[arm].get(field)}")

    dataset = config["dataset"]
    path_keys = ("train_clips", "train_actions", "train_positions",
                 "test_clips", "test_actions", "test_positions")
    data_hashes = {}
    for key in path_keys:
        path = Path(dataset[key])
        if not path.is_file():
            failures.append(f"missing E5a dataset artifact: {path}")
            continue
        data_hashes[key] = sha256_file(path)
        if data_hashes[key] != dataset["hashes"][key]:
            failures.append(f"E5a dataset hash changed: {key}")
    if Path(dataset["train_clips"]).is_file():
        if list(np.load(dataset["train_clips"], mmap_mode="r").shape) != dataset["train_shape"]:
            failures.append("H-grid3 train shape changed")
        if list(np.load(dataset["train_actions"], mmap_mode="r").shape) != [1200, 480]:
            failures.append("H-grid3 train action shape changed")
        if list(np.load(dataset["train_positions"], mmap_mode="r").shape) != [1200, 480]:
            failures.append("H-grid3 train position shape changed")
    if Path(dataset["test_clips"]).is_file():
        if list(np.load(dataset["test_clips"], mmap_mode="r").shape) != dataset["test_shape"]:
            failures.append("H-grid3 test shape changed")
        if list(np.load(dataset["test_actions"], mmap_mode="r").shape) != [300, 480]:
            failures.append("H-grid3 test action shape changed")
        if list(np.load(dataset["test_positions"], mmap_mode="r").shape) != [300, 480]:
            failures.append("H-grid3 test position shape changed")
    expected_eval = np.arange(300 - dataset["n_val"], 300)[:config["evaluation"]["clips"]]
    if [int(expected_eval[0]), int(expected_eval[-1])] != dataset["evaluation_clip_indices"]:
        failures.append("E5a evaluation clip pool changed")

    ae_path = Path(config["autoencoder"]["path"])
    ae_hash = sha256_file(ae_path) if ae_path.is_file() else ""
    if not ae_path.is_file() or ae_hash != config["autoencoder"]["sha256"]:
        failures.append("H-grid3 autoencoder missing or changed")

    prior = config["prior_evidence"]
    pinned = {
        prior["single_step_addressability"]["design"]:
            prior["single_step_addressability"]["design_sha256"],
        prior["single_step_addressability"]["seed0_log"]:
            prior["single_step_addressability"]["seed0_log_sha256"],
        prior["single_step_addressability"]["seed1_log"]:
            prior["single_step_addressability"]["seed1_log_sha256"],
    }
    seed0 = prior["seed0_rollout"]
    for stem in ("long_reach_metrics", "comparator_metrics", "long_reach_checkpoint",
                 "comparator_checkpoint", "long_reach_oracle", "comparator_oracle"):
        pinned[seed0[stem]] = seed0[f"{stem}_sha256"]
    for path, expected_hash in pinned.items():
        if not Path(path).is_file() or sha256_file(path) != expected_hash:
            failures.append(f"pinned prior evidence missing or changed: {path}")

    reference = {}
    try:
        hierarchy = _load(seed0["long_reach_metrics"])
        comparator = _load(seed0["comparator_metrics"])
        for key in ("clip", "h", "hard", "dt", "seen"):
            if not np.array_equal(hierarchy[key], comparator[key]):
                failures.append(f"seed-0 rollout pair is not aligned on {key}")
        test = np.load(dataset["test_clips"], mmap_mode="r")
        nonempty = np.asarray([
            test[int(clip), config["evaluation"]["warmup_frames"] + int(h) - 1].max()
            > config["evaluation"]["metric_threshold"] * 255.0
            for clip, h in zip(hierarchy["clip"], hierarchy["h"])
        ])
        early = nonempty & (hierarchy["h"] <= 24)
        late = nonempty & (hierarchy["h"] >= 61)
        delta = comparator["fg"] - hierarchy["fg"]
        reference = {
            "records": int(len(delta)), "nonempty_records": int(nonempty.sum()),
            "early_benefit": _interval(delta[early], hierarchy["clip"][early]),
            "late_inversion": _interval(delta[late], hierarchy["clip"][late]),
        }
        if reference["records"] != 25600 or reference["nonempty_records"] != 18177:
            failures.append("seed-0 record geometry changed")
        if reference["early_benefit"]["lo"] <= 0 or reference["late_inversion"]["hi"] >= 0:
            failures.append("seed-0 reference no longer has the registered early/late boundary")
        oracle_means = {}
        for name in ("long_reach_oracle", "comparator_oracle"):
            oracle = _load(seed0[name])
            oracle_means[name] = float(oracle["fg"].mean())
            if len(oracle["fg"]) < 4096 or oracle_means[name] >= 0.012:
                failures.append(f"legacy oracle gate failed: {name}")
        reference["legacy_oracle_means"] = oracle_means
    except Exception as error:
        failures.append(f"seed-0 reference validation failed: {error}")

    h3b_path = Path("gates/h3b_parity_report.json")
    h3b = json.loads(h3b_path.read_text(encoding="utf-8")) if h3b_path.is_file() else {}
    if h3b.get("status") != "H3B_PARITY_PASS":
        failures.append("H3b parity report is missing or failed")
    for path, expected_hash in h3b.get("source_hashes", {}).items():
        if not Path(path).is_file() or sha256_file(path) != expected_hash:
            failures.append(f"H3b-parity source changed: {path}")

    required = [
        "hvp_rev/rhvd_train_v2.py", "scripts/analyze_e5a_trust_replication.py",
        "scripts/run_e5a_trust_replication.ps1", "scripts/validate_e5a_completion.py",
        "docs/confirmation/E5A_TABLE_SHELL.md",
    ]
    for path in required:
        if not Path(path).is_file():
            failures.append(f"missing E5a implementation artifact: {path}")
    existing = []
    run_root = Path("artifacts/training") / config["study"]
    if run_root.exists():
        existing = [path.as_posix() for path in run_root.rglob("run_spec.json")]
    if existing and not args.allow_existing:
        failures.append(f"pre-existing E5a RunSpecs require explicit resume audit: {existing}")

    report = {
        "schema_version": 1, "status": "DESIGN_VALIDATION_PASS" if not failures else "FAIL",
        "config_sha256": sha256_file(args.config), "runs": len(runs),
        "allocated_gpu_hours": allocated, "authorized_gpu_hours": maximum,
        "resolved_arms": resolved, "data_hashes": data_hashes, "ae_sha256": ae_hash,
        "pinned_prior_hashes": pinned, "seed0_reference": reference,
        "h3b_parity_sha256": sha256_file(h3b_path) if h3b_path.is_file() else "",
        "existing_runs": existing, "failures": failures,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
