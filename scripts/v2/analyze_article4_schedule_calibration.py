"""Verify and analyze the paired Article 4 30k schedule calibrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.article4_run import load_article4_run_spec


CANDIDATES = (8000, 12000, 16000, 20000, 24000, 30000)
PRIMARY = "horizon4_foreground_mse"
TOLERANCE = 0.02
QUALITY_METRICS = (
    "diffusion_loss",
    "latent_mse",
    "pixel_mse",
    "foreground_mse",
    "horizon4_pixel_mse",
    "horizon4_foreground_mse",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_run(run_dir: Path, spec_path: Path) -> dict[str, Any]:
    spec = load_article4_run_spec(spec_path)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = load_jsonl(run_dir / "validation_metrics.jsonl")
    if status.get("status") != "complete" or int(status.get("step", -1)) != 30000:
        raise ValueError(f"incomplete calibration: {run_dir}")
    for source, name in ((status, "status"), (manifest, "manifest")):
        if source.get("config_hash") != spec.content_hash:
            raise ValueError(f"{name} config hash mismatch for {spec.experiment_id}")
    steps = [int(row["step"]) for row in metrics]
    expected = list(range(2000, 30001, 2000))
    if steps != expected:
        raise ValueError(f"validation milestones {steps} != {expected}")
    milestones = sorted(run_dir.glob("milestone_step*.pt"))
    if len(milestones) != len(expected):
        raise ValueError(f"expected {len(expected)} milestone checkpoints, found {len(milestones)}")
    milestone_hashes = {path.name: sha256_file(path) for path in milestones}
    return {
        "experiment_id": spec.experiment_id,
        "policy": spec.policy,
        "config_hash": spec.content_hash,
        "git_commit": manifest["git_commit"],
        "runtime_seconds": status["runtime_seconds"],
        "peak_allocated_gib": status["peak_allocated_gib"],
        "peak_reserved_gib": status["peak_reserved_gib"],
        "metrics_path": str((run_dir / "validation_metrics.jsonl").resolve()),
        "metrics_sha256": sha256_file(run_dir / "validation_metrics.jsonl"),
        "milestone_sha256": milestone_hashes,
        "metrics": metrics,
    }


def candidate_analysis(
    runs: list[dict[str, Any]], metric: str = PRIMARY
) -> tuple[list[dict[str, Any]], int]:
    by_run = {
        run["policy"]: {int(row["step"]): row for row in run["metrics"]}
        for run in runs
    }
    rows = []
    directions = {}
    for step in CANDIDATES:
        flat = by_run["flat_recent"][step][metric]
        hierarchy = by_run["balanced_recent"][step][metric]
        directions[step] = 1 if hierarchy > flat else (-1 if hierarchy < flat else 0)
    for step in CANDIDATES:
        per_arm = {}
        for policy, records in by_run.items():
            current = records[step][metric]
            best_later = min(
                row[metric] for later, row in records.items() if later >= step
            )
            relative_gap = (current - best_later) / best_later
            per_arm[policy] = {
                "value": current,
                "best_at_or_after": best_later,
                "relative_gap": relative_gap,
                "within_2_percent": relative_gap <= TOLERANCE,
            }
        later_directions = [directions[later] for later in CANDIDATES if later >= step]
        nonzero = [direction for direction in later_directions if direction]
        stable = len(set(nonzero)) <= 1
        rows.append(
            {
                "step": step,
                "arms": per_arm,
                "hierarchy_minus_flat": (
                    per_arm["balanced_recent"]["value"] - per_arm["flat_recent"]["value"]
                ),
                "ranking_direction_stable_from_here": stable,
                "passes": stable and all(
                    arm["within_2_percent"] for arm in per_arm.values()
                ),
            }
        )
    recommendation = next(row["step"] for row in rows if row["passes"])
    return rows, recommendation


def plot_curves(runs: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"flat_recent": "#2b6cb0", "balanced_recent": "#c05621"}
    labels = {"flat_recent": "Flat recent", "balanced_recent": "Balanced hierarchy"}
    series = {}
    for run in runs:
        steps = [int(row["step"]) for row in run["metrics"]]
        values = [row[PRIMARY] for row in run["metrics"]]
        series[run["policy"]] = dict(zip(steps, values))
        axes[0].plot(steps, values, marker="o", ms=3, color=colors[run["policy"]],
                     label=labels[run["policy"]])
    axes[0].set_xlabel("Training updates")
    axes[0].set_ylabel("Horizon-4 foreground MSE (lower is better)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    candidate_effects = [
        100 * (series["balanced_recent"][step] - series["flat_recent"][step])
        / series["flat_recent"][step]
        for step in CANDIDATES
    ]
    axes[1].axhline(0, color="black", lw=1)
    axes[1].plot(CANDIDATES, candidate_effects, marker="o", color="#6b46c1")
    axes[1].set_xlabel("Training updates")
    axes[1].set_ylabel("Hierarchy vs flat, relative H4 FG-MSE (%)")
    axes[1].grid(alpha=0.25)
    axes[1].annotate("ranking changes", xy=(16000, candidate_effects[2]),
                     xytext=(18500, 2.0),
                     arrowprops={"arrowstyle": "->", "lw": 1})
    fig.suptitle("A4-COR schedule calibration (seed 0, fixed validation windows)")
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def markdown_report(result: dict[str, Any], figure_path: Path) -> str:
    lines = [
        "# Article 4 schedule-calibration analysis",
        "",
        "**Decision status:** recommendation awaiting Leonid's confirmation.",
        "",
        f"Operational headline endpoint: `{PRIMARY}` (lower is better). Registered tolerance: 2%.",
        "The pre-run documents named the required metrics but did not explicitly designate one as",
        "primary. To avoid a post-hoc metric choice, the recommendation is also required to agree",
        "across every recorded validation metric; all six independently select 30k.",
        "",
        "| Updates | Flat H4 FG-MSE | Hierarchy H4 FG-MSE | Flat gap to best later | Hierarchy gap to best later | Ranking stable | Pass |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in result["candidate_analysis"]:
        flat = row["arms"]["flat_recent"]
        hierarchy = row["arms"]["balanced_recent"]
        lines.append(
            f"| {row['step']:,} | {flat['value']:.6f} | {hierarchy['value']:.6f} | "
            f"{100 * flat['relative_gap']:.2f}% | {100 * hierarchy['relative_gap']:.2f}% | "
            f"{'yes' if row['ranking_direction_stable_from_here'] else 'no'} | "
            f"{'yes' if row['passes'] else 'no'} |"
        )
    lines += [
        "",
        f"## Recommendation: {result['recommended_updates']:,} updates",
        "",
        "The original 8k and 12k options fail decisively. The decoded foreground endpoint continues",
        "to improve, and the flat-versus-hierarchy ordering reverses between 12k and 16k. At 24k,",
        "the flat arm is within 2% of its best later value, but the balanced hierarchy remains more",
        "than 2% away. The 30k endpoint is within 2% of each arm's best late milestone and preserves",
        "the post-16k ordering. Therefore 30k is the earliest registered candidate satisfying the",
        "paired rule. This is a schedule recommendation, not confirmation of a hierarchy advantage.",
        f"At 30k, each endpoint is also within 2% of its arm's best value in the 24k--30k late",
        f"window (worst final gap: {100 * result['late_window_stability']['worst_final_gap']:.2f}%).",
        "",
        "The result also invalidates the original one-GPU/one-week full-matrix estimate. Running every",
        "agreed two-seed cell for 30k requires a revised staged scope, more wall time, or additional",
        "comparable GPUs. That execution choice must be confirmed before the campaign is generated.",
        "",
        f"Figure: `{figure_path.as_posix()}`",
        "",
        "## Provenance",
        "",
    ]
    for run in result["runs"]:
        lines += [
            f"- `{run['experiment_id']}`: config `{run['config_hash']}`, commit `{run['git_commit']}`, "
            f"runtime {run['runtime_seconds'] / 3600:.2f} h, peak allocated {run['peak_allocated_gib']:.2f} GiB, "
            f"metrics SHA-256 `{run['metrics_sha256']}`.",
        ]
    lines += [
        "",
        "No test split was accessed, and these seed-0 calibration runs are not final two-seed Article 4 results.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-run", type=Path, required=True)
    parser.add_argument("--hierarchy-run", type=Path, required=True)
    parser.add_argument("--flat-spec", type=Path, required=True)
    parser.add_argument("--hierarchy-spec", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    args = parser.parse_args()

    runs = [
        verify_run(args.flat_run, args.flat_spec),
        verify_run(args.hierarchy_run, args.hierarchy_spec),
    ]
    if {run["policy"] for run in runs} != {"flat_recent", "balanced_recent"}:
        raise ValueError("calibration must contain flat_recent and balanced_recent")
    candidates, recommendation = candidate_analysis(runs)
    metric_recommendations = {
        metric: candidate_analysis(runs, metric)[1] for metric in QUALITY_METRICS
    }
    if set(metric_recommendations.values()) != {recommendation}:
        raise ValueError(f"metric-dependent schedule recommendation: {metric_recommendations}")
    late_rows = {
        run["policy"]: [row for row in run["metrics"] if int(row["step"]) >= 24000]
        for run in runs
    }
    final_gaps = {
        policy: (
            rows[-1][PRIMARY] / min(row[PRIMARY] for row in rows) - 1
        )
        for policy, rows in late_rows.items()
    }
    result = {
        "schema_version": 1,
        "status": "recommendation_awaiting_user_confirmation",
        "primary_metric": PRIMARY,
        "relative_tolerance": TOLERANCE,
        "candidate_steps": list(CANDIDATES),
        "recommended_updates": recommendation,
        "metric_robustness_recommendations": metric_recommendations,
        "late_window_stability": {
            "window": [24000, 30000],
            "final_relative_gap_by_policy": final_gaps,
            "worst_final_gap": max(final_gaps.values()),
            "within_2_percent": max(final_gaps.values()) <= TOLERANCE,
        },
        "candidate_analysis": candidates,
        "runs": [{key: value for key, value in run.items() if key != "metrics"} for run in runs],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_curves(runs, args.figure_output)
    args.markdown_output.write_text(markdown_report(result, args.figure_output), encoding="utf-8")
    print(json.dumps({"recommended_updates": recommendation, "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
