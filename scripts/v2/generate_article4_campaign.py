"""Generate the deterministic wave-1 run specs and authorization ledger for Article 4."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.article4_campaign import Article4CampaignRunSpec, file_sha256


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "article4-primary-30k-2026-09-01"
PROJECT = "hierarchical-video-prediction-article4-v2"
GROUP = "article4-30k-primary-2026-09-01"
MANIFEST = "experiments/v2/manifests/trb_v2_development.manifest.json"
BASE_AE = "C:/Users/leoni/Documents/projects/hierarchical-video-prediction/logs_rhcp/hier_ae_mom.pt"
TOP_AE = "C:/Users/leoni/Documents/projects/hierarchical-video-prediction/logs_rhcp/hier_ae_mom_top.pt"
OUT_DIR = ROOT / "experiments/v2/runspecs/article4_primary_30k"
CAMPAIGN_PATH = ROOT / "experiments/v2/campaigns/article4_primary_30k_2026-09-01.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return
    path.write_text(rendered, encoding="utf-8")


def common() -> dict[str, Any]:
    manifest_path = ROOT / MANIFEST
    return {
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "phase": "primary_30k",
        "dataset_manifest": MANIFEST,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "ae_path": BASE_AE,
        "ae_sha256": file_sha256(BASE_AE),
        "top_ae_path": None,
        "top_ae_sha256": None,
        "predictor_mode": "diffusion_v_prediction",
        "width": 256,
        "depth": 6,
        "heads": 8,
        "temporal_embedding_positions": 100,
        "context_anchor": 96,
        "batch_size": 16,
        "train_updates": 30_000,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "warmup_updates": 2_000,
        "precision": "bf16_autocast_fp32_master",
        "mixed_precision_rmsnorm": True,
        "ema_decay": 0.999,
        "validation_every": 2_000,
        "validation_batches": 4,
        "validation_batch_size": 16,
        "validation_nfe": 16,
        "checkpoint_every": 1_000,
        "milestone_every": 10_000,
        "image_every": 2_000,
        "log_every": 200,
        "output_root": "artifacts/v2/article4_primary_30k/runs",
        "wandb_project": PROJECT,
        "wandb_group": GROUP,
        "primary_metric": "horizon4_foreground_mse",
    }


def make_spec(**values: Any) -> Article4CampaignRunSpec:
    payload = common()
    payload.update(values)
    return Article4CampaignRunSpec(**payload)


def wave1_specs() -> list[Article4CampaignRunSpec]:
    specs: list[Article4CampaignRunSpec] = []
    budgets = (320, 768, 1152, 1792, 2432, 3072)
    frontier_policies = (
        ("flat_recent", "FLAT_RECENT", "exclusive", 1),
        ("linear_age", "LINEAR_AGE", "nested", 2),
        ("geometric_age", "GEOM_AGE", "nested", 2),
    )
    for budget in budgets:
        for policy, label, overlap, levels in frontier_policies:
            for seed in (0, 1):
                specs.append(
                    make_spec(
                        experiment_id=(
                            f"A4_FRT_B{budget:04d}_L{levels}_{label}_"
                            f"{'EXCL' if overlap == 'exclusive' else 'NEST'}_S{seed:02d}_T30K"
                        ),
                        family="FRT",
                        policy=policy,
                        overlap=overlap,
                        budget_tokens=budget,
                        seed=seed,
                        allocation_kind="frontier",
                        allocation_args={
                            "budget": budget,
                            "policy": policy,
                            "geometric_decay": 0.94,
                        },
                    )
                )

    allocation_additions = (
        ("fine_heavy", "FINE_HEAVY"),
        ("balanced_recent", "BAL_RECENT"),
        ("coarse_heavy", "COARSE_HEAVY"),
        ("reach_optimized", "REACH_OPT"),
    )
    for policy, label in allocation_additions:
        for seed in (0, 1):
            specs.append(
                make_spec(
                    experiment_id=f"A4_ALC_B1792_L2_{label}_NEST_S{seed:02d}_T30K",
                    family="ALC",
                    policy=policy,
                    overlap="nested",
                    budget_tokens=1792,
                    seed=seed,
                    allocation_kind="allocation_1792",
                    allocation_args={"policy": policy, "geometric_decay": 0.94},
                )
            )

    for coarse_times in (0, 4, 8, 16, 32, 64):
        budget = 384 + coarse_times * 16
        levels = 1 if coarse_times == 0 else 2
        for seed in (0, 1):
            specs.append(
                make_spec(
                    experiment_id=(
                        f"A4_CEX_B{budget:04d}_L{levels}_C{coarse_times:02d}_"
                        f"EXCL_S{seed:02d}_T30K"
                    ),
                    family="CEX",
                    policy="coarse_expansion",
                    overlap="exclusive",
                    budget_tokens=budget,
                    seed=seed,
                    allocation_kind="coarse_expansion",
                    allocation_args={"coarse_times": coarse_times},
                )
            )

    temporal = (
        ("recent_detail", "RECENT"),
        ("uniform_time", "UNIFORM"),
        ("geometric_time", "GEOMETRIC"),
    )
    for quota in (64, 32, 16, 8):
        for policy, label in temporal:
            for seed in (0, 1):
                specs.append(
                    make_spec(
                        experiment_id=(
                            f"A4_STS_B1792_L2_K{quota:02d}_{label}_NEST_"
                            f"S{seed:02d}_T30K"
                        ),
                        family="STS",
                        policy=f"k{quota}_{policy}",
                        overlap="nested",
                        budget_tokens=1792,
                        seed=seed,
                        allocation_kind="spatial_temporal",
                        allocation_args={
                            "budget": 1792,
                            "quota": quota,
                            "temporal_policy": policy,
                        },
                    )
                )
    if len(specs) != 80 or len({spec.experiment_id for spec in specs}) != 80:
        raise AssertionError(f"expected 80 unique wave-1 runs, got {len(specs)}")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated files without writing")
    args = parser.parse_args()
    specs = wave1_specs()
    entries = []
    for spec in specs:
        path = OUT_DIR / f"{spec.experiment_id}.json"
        relative = path.relative_to(ROOT).as_posix()
        entries.append(
            {
                "experiment_id": spec.experiment_id,
                "config": relative,
                "config_hash": spec.content_hash,
                "family": spec.family,
                "seed": spec.seed,
            }
        )
        if args.check:
            if not path.exists():
                raise FileNotFoundError(path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if loaded != asdict(spec):
                raise ValueError(f"generated spec drift: {path}")
        else:
            write_json(path, asdict(spec))

    campaign = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "authorized_30k",
        "authorization": {
            "authorized_by": "Leonid Kulyk",
            "confirmed_utc_date": "2026-09-01",
            "decision": "30,000 updates for every primary Article 4 run",
        },
        "confirmed_train_updates": 30_000,
        "dataset_manifest": MANIFEST,
        "dataset_manifest_sha256": file_sha256(ROOT / MANIFEST),
        "wandb": {
            "project": PROJECT,
            "group": GROUP,
            "local_artifacts_authoritative": True,
            "runtime_failure_policy": "continue_locally_and_mark_unsynced",
        },
        "counts": {
            "wave1_runs": 80,
            "dependent_wave2_runs": 6,
            "total_runs": 86,
            "unique_configurations": 43,
            "seeds_per_configuration": 2,
        },
        "run_order": entries,
        "authorized_runs": {entry["experiment_id"]: entry["config_hash"] for entry in entries},
        "dependency_resolution": {
            "primary_metric": "horizon4_foreground_mse",
            "checkpoint_step": 30_000,
            "aggregation": "arithmetic mean across seeds 0 and 1",
            "level_base": "best non-flat A4-ALC policy",
            "overlap_base": (
                "best non-flat A4-ALC policy whose unchanged fine identities and coarse slot "
                "count fit disjointly within 96 ages"
            ),
            "overlap_feasible_policies": ["fine_heavy", "balanced_recent", "coarse_heavy"],
            "tie_break_order": [
                "balanced_recent",
                "fine_heavy",
                "coarse_heavy",
                "linear_age",
                "geometric_age",
                "reach_optimized",
            ],
            "resolver": "scripts/v2/resolve_article4_dependencies.py",
        },
        "post_training": {
            "selection_split": "development validation only",
            "selection_checkpoint": 30_000,
            "selection_aggregation": "arithmetic mean across seeds 0 and 1",
            "mechanism_roles": [
                "best_flat_1792",
                "best_hierarchy_1792",
                "failure_hierarchy_1792",
            ],
            "confirmation_roles": [
                "best_flat_1792",
                "best_hierarchy_1792",
                "memory_pareto_within_5pct",
            ],
            "memory_pareto_quality_tolerance_relative": 0.05,
            "confirmation": {
                "seed": 26090141,
                "clips": 500,
                "frames": 200,
                "world": 192,
                "viewport": 64,
                "digits": 3,
                "speed": 4,
                "p_cont": 0.9,
                "p_hold": 0.1,
                "mnist_path": (
                    "C:/Users/leoni/Documents/projects/hierarchical-video-prediction/"
                    "data/source/mnist/train-images-idx3-ubyte.gz"
                ),
                "mnist_sha256": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
                "evaluation_start": 196,
                "long_context": 96,
                "recent_reference": 28,
                "read_policy": "one locked evaluation after development-only selection",
            },
            "finalizer": "scripts/v2/finalize_article4_campaign.py",
        },
        "primary_checkpoint": 30_000,
        "validation_milestones": list(range(2_000, 30_001, 2_000)),
        "model_milestones": [10_000, 20_000, 30_000],
    }
    if args.check:
        if not CAMPAIGN_PATH.exists():
            raise FileNotFoundError(CAMPAIGN_PATH)
        if json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8")) != campaign:
            raise ValueError("campaign manifest drift")
    else:
        write_json(CAMPAIGN_PATH, campaign)
    print(
        json.dumps(
            {
                "campaign": str(CAMPAIGN_PATH.relative_to(ROOT)),
                "wave1_runs": len(specs),
                "authorized_hashes": len(campaign["authorized_runs"]),
                "mode": "check" if args.check else "write",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
