"""Resolve the six A4-LVL/A4-OVR runs from completed two-seed A4-ALC evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.allocation import overlap_1792_plan
from hvp_hca.article4_campaign import (
    Article4CampaignRunSpec,
    file_sha256,
    load_campaign_run_spec,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "experiments/v2/campaigns/article4_primary_30k_2026-09-01.json"
DEFAULT_OUTPUT = ROOT / "artifacts/v2/article4_primary_30k/dependency_resolution.json"
TOP_AE = "C:/Users/leoni/Documents/projects/hierarchical-video-prediction/logs_rhcp/hier_ae_mom_top.pt"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def final_metric(spec: Article4CampaignRunSpec) -> tuple[float, Path, str]:
    run_dir = ROOT / spec.output_root / f"{spec.experiment_id}__{spec.content_hash[:12]}"
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "validation_metrics.jsonl"
    if not status_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"A4-ALC dependency is incomplete: {spec.experiment_id}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or int(status.get("step", 0)) != 30_000:
        raise RuntimeError(f"A4-ALC dependency is not complete at 30k: {spec.experiment_id}")
    matches = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("step", -1)) == 30_000:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"expected one final validation record: {spec.experiment_id}")
    return float(matches[0][spec.primary_metric]), metrics_path, file_sha256(metrics_path)


def load_alc_sources(campaign: dict[str, Any]) -> dict[str, dict[int, Article4CampaignRunSpec]]:
    selected: dict[str, dict[int, Article4CampaignRunSpec]] = {}
    for entry in campaign["run_order"]:
        spec = load_campaign_run_spec(ROOT / entry["config"])
        is_reused = (
            spec.family == "FRT"
            and spec.budget_tokens == 1792
            and spec.policy in {"flat_recent", "linear_age", "geometric_age"}
        )
        if spec.family == "ALC" or is_reused:
            selected.setdefault(spec.policy, {})[spec.seed] = spec
    expected = {
        "flat_recent",
        "fine_heavy",
        "balanced_recent",
        "coarse_heavy",
        "linear_age",
        "geometric_age",
        "reach_optimized",
    }
    if set(selected) != expected or any(set(by_seed) != {0, 1} for by_seed in selected.values()):
        raise ValueError("campaign does not provide both A4-ALC seeds for all seven policies")
    return selected


def choose_policies(means: dict[str, float], rule: dict[str, Any]) -> tuple[str, str]:
    tie_order = {policy: index for index, policy in enumerate(rule["tie_break_order"])}
    nonflat = [policy for policy in means if policy != "flat_recent"]
    if not nonflat:
        raise ValueError("dependency selection needs at least one non-flat policy")
    level_base = min(nonflat, key=lambda policy: (means[policy], tie_order[policy]))
    feasible = list(rule["overlap_feasible_policies"])
    if any(policy not in means for policy in feasible):
        raise ValueError("overlap-feasible policy is missing from A4-ALC means")
    overlap_base = min(feasible, key=lambda policy: (means[policy], tie_order[policy]))
    return level_base, overlap_base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    campaign_path = args.campaign.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_hash = file_sha256(campaign_path)
    sources = load_alc_sources(campaign)
    evidence: dict[str, Any] = {}
    means: dict[str, float] = {}
    for policy, by_seed in sources.items():
        values = []
        seed_evidence = {}
        for seed, spec in sorted(by_seed.items()):
            value, metrics_path, metrics_hash = final_metric(spec)
            values.append(value)
            seed_evidence[str(seed)] = {
                "experiment_id": spec.experiment_id,
                "config_hash": spec.content_hash,
                "metric": value,
                "metrics_file": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
                "metrics_sha256": metrics_hash,
            }
        means[policy] = sum(values) / len(values)
        evidence[policy] = {"seeds": seed_evidence, "mean": means[policy]}

    rule = campaign["dependency_resolution"]
    tie_order = {policy: index for index, policy in enumerate(rule["tie_break_order"])}
    level_base, overlap_base = choose_policies(means, rule)
    feasible = list(rule["overlap_feasible_policies"])
    for policy in feasible:
        for mode in ("nested", "exclusive", "residual"):
            overlap_1792_plan(policy, mode)

    template = sources[level_base][0]
    specs: list[Article4CampaignRunSpec] = []
    for seed in (0, 1):
        specs.append(
            replace(
                template,
                experiment_id=f"A4_LVL_B1792_L3_FROM_{level_base.upper()}_NEST_S{seed:02d}_T30K",
                family="LVL",
                policy=f"three_level_from_{level_base}",
                overlap="nested",
                seed=seed,
                allocation_kind="three_level_1792",
                allocation_args={"base_policy": level_base, "geometric_decay": 0.94},
                top_ae_path=TOP_AE,
                top_ae_sha256=file_sha256(TOP_AE),
            )
        )
    template = sources[overlap_base][0]
    for mode, label in (("exclusive", "EXCL"), ("residual", "RESID")):
        for seed in (0, 1):
            specs.append(
                replace(
                    template,
                    experiment_id=(
                        f"A4_OVR_B1792_L2_{overlap_base.upper()}_{label}_"
                        f"S{seed:02d}_T30K"
                    ),
                    family="OVR",
                    policy=overlap_base,
                    overlap=mode,
                    seed=seed,
                    allocation_kind="overlap_1792",
                    allocation_args={
                        "base_policy": overlap_base,
                        "mode": mode,
                        "geometric_decay": 0.94,
                    },
                    top_ae_path=None,
                    top_ae_sha256=None,
                )
            )
    if len(specs) != 6 or len({spec.experiment_id for spec in specs}) != 6:
        raise AssertionError("dependency resolver must produce exactly six unique runs")

    spec_dir = args.output.resolve().parent / "resolved_run_specs"
    run_order = []
    for spec in specs:
        path = spec_dir / f"{spec.experiment_id}.json"
        atomic_json(path, asdict(spec))
        run_order.append(
            {
                "experiment_id": spec.experiment_id,
                "config": str(path.relative_to(ROOT)).replace("\\", "/"),
                "config_hash": spec.content_hash,
                "family": spec.family,
                "seed": spec.seed,
            }
        )
    resolution = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "parent_campaign_sha256": campaign_hash,
        "status": "resolved",
        "selection_checkpoint": 30_000,
        "primary_metric": rule["primary_metric"],
        "aggregation": rule["aggregation"],
        "evidence": evidence,
        "selected": {
            "best_overall": min(means, key=lambda policy: (means[policy], tie_order.get(policy, -1))),
            "level_base_best_nonflat": level_base,
            "overlap_base_best_feasible": overlap_base,
        },
        "reused_runs": {
            "LVL_L1": [sources["flat_recent"][seed].experiment_id for seed in (0, 1)],
            "LVL_L2": [sources[level_base][seed].experiment_id for seed in (0, 1)],
            "OVR_nested": [sources[overlap_base][seed].experiment_id for seed in (0, 1)],
        },
        "run_order": run_order,
        "authorized_runs": {entry["experiment_id"]: entry["config_hash"] for entry in run_order},
    }
    atomic_json(args.output.resolve(), resolution)
    print(json.dumps({"output": str(args.output), "selected": resolution["selected"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
