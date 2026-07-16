"""Frozen E7a joint original/alternate-codec analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import wb
from hvp_rev.analysis import cluster_bootstrap_mean, paired_effect, simultaneous_cluster_intervals
from hvp_rev.provenance import atomic_write_bytes, atomic_write_json, sha256_file


ALIGN_KEYS = ("clip", "h", "hard", "dt", "seen", "fg_nonempty")
HIGHER_BETTER = {"fg_precision", "fg_recall", "fg_f1"}


def discover(root: Path, dataset_hash: str, codec_hashes: dict[str, str]) -> dict[tuple[str, str], Path]:
    by_hash = {value: key for key, value in codec_hashes.items()}
    found: dict[tuple[str, str], Path] = {}
    for spec_path in root.rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "robustness-e7a" or spec.get("dataset_hash") != dataset_hash:
            continue
        codec = by_hash.get(spec.get("ae_hash"))
        if codec is None:
            raise ValueError(f"unexpected codec hash in completed E7a RunSpec: {spec_path}")
        key = (codec, spec["arm"])
        if key in found:
            raise RuntimeError(f"duplicate completed E7a cell: {key}")
        result = run_dir / "metrics/rollout_complete.npz"
        if not result.is_file():
            raise FileNotFoundError(result)
        found[key] = result
    return found


def aligned(reference, other) -> None:
    for key in ALIGN_KEYS:
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"paired E7a key differs: {key}")


def registered_verdict(primary: list[dict]) -> str:
    original = [row for row in primary if row["codec"] == "original"]
    alternate = [row for row in primary if row["codec"] == "alternate"]
    controls_positive = all(row["mean"] > 0 for row in original)
    alternate_positive = all(row["mean"] > 0 for row in alternate)
    alternate_clean = all(row["lo"] > 0 for row in alternate)
    if not controls_positive:
        return "HOLDOUT_CONTROL_FAILURE"
    if not alternate_positive:
        return "CODEC_SENSITIVE"
    if alternate_clean:
        return "CODEC_ROBUST_CI_CLEAN"
    return "CODEC_DIRECTIONALLY_ROBUST"


def markdown(result: dict) -> str:
    lines = [
        "# E7a alternate-codec robustness results", "",
        f"> Dataset: `{result['dataset_id']}`. Positive favors the registered candidate.", "",
        f"Registered verdict: **{result['decision']['verdict']}**.", "",
        "| Codec | Contrast | Effect | Simultaneous 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in result["primary"]:
        lines.append(
            f"| {row['codec']} | {row['contrast']} | {row['mean']:+.4f} | "
            f"[{row['lo']:+.4f}, {row['hi']:+.4f}] |"
        )
    lines += ["", "Descriptive alternate-minus-original effect changes:", "",
              "| Contrast | Change | Clip-bootstrap 95% CI |", "|---|---:|---:|"]
    for row in result["effect_change_diagnostic"]:
        lines.append(f"| {row['contrast']} | {row['mean']:+.4f} | [{row['lo']:+.4f}, {row['hi']:+.4f}] |")
    lines += ["", "No codec, predictor seed, arm, threshold or retry was selected after results were observed."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--alternate-ae", required=True)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    codec_hashes = {
        "original": config["codecs"]["original"]["sha256"],
        "alternate": sha256_file(args.alternate_ae),
    }
    paths = discover(Path(args.run_root), manifest["artifacts"]["clips"]["sha256"], codec_hashes)
    expected = {(codec, arm["arm"]) for codec in codec_hashes for arm in config["predictor_runs"]}
    if set(paths) != expected:
        raise ValueError(f"E7a inventory mismatch: missing={sorted(expected-set(paths))}; extra={sorted(set(paths)-expected)}")

    arrays = {key: np.load(path) for key, path in paths.items()}
    reference = arrays[("original", config["predictor_runs"][0]["arm"])]
    for values in arrays.values():
        aligned(reference, values)
    clips = reference["clip"]
    deltas: dict[str, np.ndarray] = {}
    rows = []
    corroborating = []
    diagnostic = []
    for codec in ("original", "alternate"):
        for contrast in config["contrasts"]:
            base = arrays[(codec, contrast["baseline"])]
            candidate = arrays[(codec, contrast["candidate"])]
            key = f"{codec}::{contrast['id']}"
            deltas[key] = base["fg"] - candidate["fg"]
            rows.append((key, codec, contrast))
            for metric in config["decision"]["corroborating_metrics"]:
                effect = paired_effect(
                    base[metric], candidate[metric], clips,
                    lower_is_better=metric not in HIGHER_BETTER,
                    n_boot=config["evaluation"]["bootstrap_replicates"],
                    seed=config["evaluation"]["bootstrap_seed"],
                )
                corroborating.append({
                    "codec": codec, "contrast": contrast["id"], "metric": metric, **effect,
                })
    family = simultaneous_cluster_intervals(
        deltas, clips,
        n_boot=config["evaluation"]["bootstrap_replicates"],
        seed=config["evaluation"]["bootstrap_seed"],
        confidence=config["evaluation"]["family_confidence"],
    )
    primary = [{
        "codec": codec, "contrast": contrast["id"],
        "candidate": contrast["candidate"], "baseline": contrast["baseline"],
        **family["contrasts"][key],
    } for key, codec, contrast in rows]
    for contrast in config["contrasts"]:
        change = deltas[f"alternate::{contrast['id']}"] - deltas[f"original::{contrast['id']}"]
        diagnostic.append({
            "contrast": contrast["id"],
            **cluster_bootstrap_mean(
                change, clips, n_boot=config["evaluation"]["bootstrap_replicates"],
                seed=config["evaluation"]["bootstrap_seed"],
            ),
        })

    original_rows = [row for row in primary if row["codec"] == "original"]
    alternate_rows = [row for row in primary if row["codec"] == "alternate"]
    controls_positive = all(row["mean"] > 0 for row in original_rows)
    alternate_positive = all(row["mean"] > 0 for row in alternate_rows)
    alternate_clean = all(row["lo"] > 0 for row in alternate_rows)
    verdict = registered_verdict(primary)
    result = {
        "schema_version": 1,
        "study": "robustness-e7a",
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "config_sha256": sha256_file(args.config),
        "codec_hashes": codec_hashes,
        "family": {key: value for key, value in family.items() if key != "contrasts"},
        "primary": primary,
        "corroborating": corroborating,
        "effect_change_diagnostic": diagnostic,
        "decision": {
            "verdict": verdict,
            "original_controls_positive": controls_positive,
            "alternate_point_estimates_positive": alternate_positive,
            "alternate_simultaneous_lower_bounds_positive": alternate_clean,
            "rule": config["decision"],
        },
        "input_artifacts": {
            "::".join(key): sha256_file(path) for key, path in sorted(paths.items())
        },
    }
    atomic_write_json(args.out_json, result)
    atomic_write_bytes(args.out_md, markdown(result).encode("utf-8"))
    run_id = f"e7a-analysis-{manifest['dataset_id'].split('-')[-1]}"
    wb.init(
        run_id, project=args.wandb_project, run_id=run_id, resume="never",
        group=args.wandb_group, job_type="codec-robustness-analysis", required=True,
        tags=["loopmom", "alternate-codec-robustness", "analysis"],
        config={"config_sha256": result["config_sha256"],
                "manifest_sha256": result["dataset_manifest_sha256"]},
    )
    wb.summary({
        "analysis/verdict": verdict, "analysis/evaluations": len(paths),
        "analysis/effects": len(primary), "analysis/max_stat_critical": family["critical_value"],
    })
    wb.artifact(
        run_id, "codec-robustness-analysis",
        [args.out_json, args.out_md, args.config, args.manifest],
        metadata={"dataset_id": manifest["dataset_id"], "verdict": verdict}, aliases=["latest"],
    )
    wb.finish()
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
