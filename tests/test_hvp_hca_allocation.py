from __future__ import annotations

import pytest

from hvp_hca.allocation import (
    FRONTIER_BUDGETS,
    allocation_1792_plan,
    cell_order_for_age,
    coarse_expansion_plan,
    frontier_plan,
    overlap_1792_plan,
    spatial_temporal_plan,
    three_level_1792_plan,
)


@pytest.mark.parametrize("budget", FRONTIER_BUDGETS)
@pytest.mark.parametrize(
    "policy", ["flat_recent", "balanced_recent", "linear_age", "geometric_age"]
)
def test_frontier_plans_use_exact_budget_and_valid_local_indices(budget: int, policy: str) -> None:
    plan = frontier_plan(budget, policy)
    assert plan.token_count == budget
    assert len(set(plan.fine_age_cells)) == plan.fine_tokens
    frames, cells = plan.local_fine_indices()
    assert len(frames) == len(cells) == plan.fine_tokens
    assert min(frames, default=0) >= 0
    assert max(frames, default=0) < max(plan.reach, 1)
    assert all(0 <= cell < 64 for cell in cells)


@pytest.mark.parametrize(
    "policy",
    [
        "flat_recent",
        "fine_heavy",
        "balanced_recent",
        "coarse_heavy",
        "linear_age",
        "geometric_age",
        "reach_optimized",
    ],
)
def test_all_seven_allocation_plans_are_exact(policy: str) -> None:
    assert allocation_1792_plan(policy).token_count == 1792


@pytest.mark.parametrize("coarse_times", [0, 4, 8, 16, 32, 64])
def test_coarse_expansion_keeps_six_fine_times_and_exclusive_support(coarse_times: int) -> None:
    plan = coarse_expansion_plan(coarse_times)
    assert plan.fine_tokens == 384
    assert plan.token_count == 384 + 16 * coarse_times
    assert not set(plan.coarse_ages).intersection(age for age, _ in plan.fine_age_cells)


@pytest.mark.parametrize("budget,coarse_times", [(1152, 56), (1792, 96)])
@pytest.mark.parametrize("quota,detail_times", [(64, 4), (32, 8), (16, 16), (8, 32)])
@pytest.mark.parametrize("policy", ["recent_detail", "uniform_time", "geometric_time"])
def test_spatial_temporal_surface_accounting(
    budget: int, coarse_times: int, quota: int, detail_times: int, policy: str
) -> None:
    plan = spatial_temporal_plan(budget, quota, policy)
    assert plan.token_count == budget
    assert len(plan.coarse_ages) == coarse_times
    assert plan.fine_tokens == 256
    assert len({age for age, _ in plan.fine_age_cells}) == detail_times


def test_spatial_cell_order_is_a_nested_permutation_for_each_age() -> None:
    for age in (1, 2, 17, 96):
        order = cell_order_for_age(age)
        assert len(order) == 64
        assert set(order) == set(range(64))
        assert set(order[:8]).issubset(order[:16])
        assert set(order[:16]).issubset(order[:32])


@pytest.mark.parametrize(
    "policy",
    [
        "fine_heavy",
        "balanced_recent",
        "coarse_heavy",
        "linear_age",
        "geometric_age",
        "reach_optimized",
    ],
)
def test_three_level_adaptation_is_exact_and_reaches_full_history(policy: str) -> None:
    plan = three_level_1792_plan(policy)
    assert plan.token_count == 1792
    assert plan.reach == 96
    assert plan.top_ages == tuple(range(1, 97))
    assert plan.fine_representation == "absolute"


@pytest.mark.parametrize("policy", ["fine_heavy", "balanced_recent", "coarse_heavy"])
def test_overlap_modes_are_exact_and_scientifically_distinct(policy: str) -> None:
    nested = overlap_1792_plan(policy, "nested")
    exclusive = overlap_1792_plan(policy, "exclusive")
    residual = overlap_1792_plan(policy, "residual")
    assert {nested.token_count, exclusive.token_count, residual.token_count} == {1792}
    assert set(exclusive.coarse_ages).isdisjoint(age for age, _ in exclusive.fine_age_cells)
    assert set(age for age, _ in residual.fine_age_cells).issubset(residual.coarse_ages)
    assert residual.fine_representation == "coarse_embedding_residual"
    assert exclusive.coarse_ages != nested.coarse_ages


@pytest.mark.parametrize("policy", ["linear_age", "geometric_age", "reach_optimized"])
def test_infeasible_exclusive_overlap_fails_closed(policy: str) -> None:
    with pytest.raises(ValueError, match="96-frame exclusive limit"):
        overlap_1792_plan(policy, "exclusive")
