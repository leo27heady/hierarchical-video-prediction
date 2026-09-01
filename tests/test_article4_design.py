from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hvp_hca.allocation import (
    allocation_1792_plan,
    coarse_expansion_plan,
    frontier_plan,
    spatial_temporal_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments" / "v2" / "designs" / "article4_matrix_2026-09-01.json"


def test_approved_article4_design_resolves_to_exact_allocation_contracts() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    assert design["training_seeds"] == [0, 1]
    assert not design["launch_authorized"]
    decay = design["allocation"]["geometric_decay"]

    frontier = design["series"]["A4-FRT"]
    for budget in frontier["budgets"]:
        for policy in frontier["policies"]:
            assert frontier_plan(budget, policy, geometric_decay=decay).token_count == budget

    allocation = design["series"]["A4-ALC"]
    for policy in allocation["policies"]:
        assert allocation_1792_plan(policy, geometric_decay=decay).token_count == 1792

    expansion = design["series"]["A4-CEX"]
    for coarse_times in expansion["coarse_time_positions"]:
        plan = coarse_expansion_plan(coarse_times)
        assert plan.token_count == 384 + 16 * coarse_times

    surface = design["series"]["A4-STS"]
    for budget in surface["budgets"]:
        for quota in surface["spatial_quotas"]:
            for policy in surface["temporal_policies"]:
                assert spatial_temporal_plan(budget, quota, policy).token_count == budget


def test_article4_design_pins_the_tracked_development_manifest() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    dataset = design["dataset"]
    manifest = ROOT / dataset["development_manifest"]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == dataset["development_manifest_sha256"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["purpose"] == "development_only"
    assert not payload["test_policy"]["included"]
    assert payload["split"]["train_count"] == 2700
    assert payload["split"]["validation_count"] == 300
