from __future__ import annotations

import json
from pathlib import Path

import torch

from hvp_hca.article4_campaign import load_campaign_run_spec, validate_campaign_authorization
from hvp_hca.postcampaign import pair_results, select_postcampaign
from scripts.v2.resolve_article4_dependencies import choose_policies
from scripts.v2.finalize_article4_campaign import attention_diagnostics
from hvp_rev.rhvd import WindowDiT


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/v2/campaigns/article4_primary_30k_2026-09-01.json"


def test_primary_campaign_authorizes_exactly_80_wave1_runs() -> None:
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    assert campaign["status"] == "authorized_30k"
    assert campaign["confirmed_train_updates"] == 30000
    assert campaign["counts"] == {
        "wave1_runs": 80,
        "dependent_wave2_runs": 6,
        "total_runs": 86,
        "unique_configurations": 43,
        "seeds_per_configuration": 2,
    }
    specs = [load_campaign_run_spec(ROOT / entry["config"]) for entry in campaign["run_order"]]
    assert len(specs) == 80
    assert len({spec.experiment_id for spec in specs}) == 80
    assert len({spec.content_hash for spec in specs}) == 80
    for spec in specs:
        validate_campaign_authorization(spec, CAMPAIGN)
        assert spec.train_updates == 30000
        assert spec.plan.token_count == spec.budget_tokens
        assert spec.seed in {0, 1}


def test_reduced_family_run_counts_are_locked() -> None:
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    families: dict[str, int] = {}
    for entry in campaign["run_order"]:
        families[entry["family"]] = families.get(entry["family"], 0) + 1
    assert families == {"FRT": 36, "ALC": 8, "CEX": 12, "STS": 24}


def test_dependency_selection_separates_best_nonflat_from_overlap_feasibility() -> None:
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    means = {
        "flat_recent": 0.020,
        "fine_heavy": 0.025,
        "balanced_recent": 0.024,
        "coarse_heavy": 0.026,
        "linear_age": 0.021,
        "geometric_age": 0.022,
        "reach_optimized": 0.023,
    }
    level, overlap = choose_policies(means, campaign["dependency_resolution"])
    assert level == "linear_age"
    assert overlap == "balanced_recent"


def _selection_row(policy: str, seed: int, metric: float, *, tokens: int = 1792) -> dict:
    family = "FRT" if policy in {"flat_recent", "linear_age", "geometric_age"} else "ALC"
    return {
        "experiment_id": f"{policy}-{seed}",
        "config": f"{policy}-{seed}.json",
        "config_hash": f"hash-{policy}-{seed}",
        "checkpoint": f"{policy}-{seed}.pt",
        "checkpoint_sha256": f"checkpoint-{policy}-{seed}",
        "spec": {
            "allocation_kind": "frontier" if family == "FRT" else "allocation_1792",
            "allocation_args": {"budget": 1792, "policy": policy},
            "budget_tokens": 1792,
            "policy": policy,
            "overlap": "exclusive" if policy == "flat_recent" else "nested",
            "family": family,
            "seed": seed,
        },
        "step": 30000,
        "metrics": {"horizon4_foreground_mse": metric},
        "levels": 1 if policy == "flat_recent" else 2,
        "context_tokens": tokens,
        "state_bytes_per_sample_bf16": tokens * 32,
    }


def test_postcampaign_selection_uses_seed_means_and_registered_tolerance() -> None:
    metrics = {
        "flat_recent": 0.020,
        "fine_heavy": 0.022,
        "balanced_recent": 0.019,
        "coarse_heavy": 0.024,
        "linear_age": 0.0205,
        "geometric_age": 0.0215,
        "reach_optimized": 0.023,
    }
    rows = [
        _selection_row(policy, seed, value + seed * 0.0002)
        for policy, value in metrics.items()
        for seed in (0, 1)
    ]
    selected = select_postcampaign(pair_results(rows))
    assert selected["best_flat_1792"].policy == "flat_recent"
    assert selected["best_hierarchy_1792"].policy == "balanced_recent"
    assert selected["failure_hierarchy_1792"].policy == "coarse_heavy"
    assert selected["memory_pareto_within_5pct"].policy in {
        "balanced_recent", "linear_age"
    }


def test_attention_diagnostics_records_normalized_entropy_and_source_mass() -> None:
    model = WindowDiT(
        n_ctx_c=0,
        n_ctx_f=1,
        fine_token_budget=8,
        d=32,
        depth=1,
        heads=4,
        temporal_embedding_positions=100,
        temporal_context_span=96,
    ).eval()
    spans = {
        "context_top": (0, 0),
        "context_coarse": (0, 0),
        "context_fine": (0, 8),
        "target_coarse": (8, 72),
        "target_fine": (72, 328),
    }
    with torch.no_grad(), attention_diagnostics(model, spans) as records:
        model(
            torch.randn(1, 4, 16, 4, 4),
            torch.randn(1, 4, 16, 8, 8),
            torch.empty(1, 0, 16, 4, 4),
            torch.empty(1, 0, 16, 8, 8),
            torch.full((1,), 0.5),
            torch.zeros(1, 4, dtype=torch.long),
            ctx_f_ids=torch.arange(8).view(1, 8),
            ctx_f_frame_ids=torch.zeros(1, 8, dtype=torch.long),
            ctx_f_packed=torch.randn(1, 8, 16),
        )
    assert len(records) == 1
    assert 0 <= records[0]["normalized_entropy"] <= 1
    assert abs(sum(records[0]["source_mass"].values()) - 1.0) < 1e-5
