"""Deterministic Article 4 context-allocation schedules.

Age ``1`` is the newest context time position.  Larger ages are older.  Fine
cells use row-major identities on the physical 8x8 latent grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


COARSE_TOKENS_PER_TIME = 16
FINE_CELLS_PER_TIME = 64
FRONTIER_BUDGETS = (320, 768, 1152, 1792, 2432, 3072)


@dataclass(frozen=True)
class AllocationPlan:
    coarse_ages: tuple[int, ...]
    fine_age_cells: tuple[tuple[int, int], ...]
    policy: str
    overlap: str = "nested"
    top_ages: tuple[int, ...] = ()
    fine_representation: str = "absolute"

    def __post_init__(self) -> None:
        if self.overlap not in {"nested", "exclusive", "residual"}:
            raise ValueError(f"unknown overlap: {self.overlap}")
        if self.fine_representation not in {"absolute", "coarse_embedding_residual"}:
            raise ValueError(f"unknown fine representation: {self.fine_representation}")
        if self.overlap == "residual" and self.fine_representation != "coarse_embedding_residual":
            raise ValueError("residual overlap requires coarse_embedding_residual fine tokens")
        if self.overlap != "residual" and self.fine_representation != "absolute":
            raise ValueError("only residual overlap may use residual fine tokens")
        if any(age <= 0 for age in self.top_ages):
            raise ValueError("top ages must be positive")
        if len(self.top_ages) != len(set(self.top_ages)):
            raise ValueError("top ages must be unique")
        if any(age <= 0 for age in self.coarse_ages):
            raise ValueError("coarse ages must be positive")
        if len(self.coarse_ages) != len(set(self.coarse_ages)):
            raise ValueError("coarse ages must be unique")
        if len(self.fine_age_cells) != len(set(self.fine_age_cells)):
            raise ValueError("fine (age,cell) identities must be unique")
        for age, cell in self.fine_age_cells:
            if age <= 0 or not 0 <= cell < FINE_CELLS_PER_TIME:
                raise ValueError("invalid fine age/cell identity")

    @property
    def token_count(self) -> int:
        return (
            len(self.top_ages) * 4
            + len(self.coarse_ages) * COARSE_TOKENS_PER_TIME
            + len(self.fine_age_cells)
        )

    @property
    def reach(self) -> int:
        ages = (
            list(self.top_ages)
            + list(self.coarse_ages)
            + [age for age, _ in self.fine_age_cells]
        )
        return max(ages, default=0)

    @property
    def fine_tokens(self) -> int:
        return len(self.fine_age_cells)

    def local_fine_indices(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return local oldest-to-newest frame ids and physical cell ids."""

        frames = tuple(self.reach - age for age, _ in self.fine_age_cells)
        cells = tuple(cell for _, cell in self.fine_age_cells)
        return frames, cells


def _farthest_cell_order() -> tuple[int, ...]:
    remaining = {(row, column) for row in range(8) for column in range(8)}
    selected: list[tuple[int, int]] = []
    while remaining:
        if not selected:
            choice = (3, 3)
        else:
            choice = max(
                remaining,
                key=lambda point: (
                    min(
                        (point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2
                        for old in selected
                    ),
                    -point[0],
                    -point[1],
                ),
            )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(row * 8 + column for row, column in selected)


_BASE_CELL_ORDER = _farthest_cell_order()


def cell_order_for_age(age: int) -> tuple[int, ...]:
    """A phase-cycled, nested spatial order independent of image content."""

    if age <= 0:
        raise ValueError("age must be positive")
    row_shift, column_shift = (3 * age) % 8, (5 * age) % 8
    return tuple(
        ((cell // 8 + row_shift) % 8) * 8 + ((cell % 8 + column_shift) % 8)
        for cell in _BASE_CELL_ORDER
    )


def _weighted_counts(total: int, weights: Iterable[float], cap: int = 64) -> tuple[int, ...]:
    values = tuple(float(weight) for weight in weights)
    if total < 0 or not values or any(weight < 0 for weight in values):
        raise ValueError("invalid weighted allocation")
    if total > len(values) * cap or sum(values) <= 0:
        raise ValueError("weighted allocation is infeasible")
    raw = [total * weight / sum(values) for weight in values]
    counts = [min(cap, int(math.floor(value))) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(values)), key=lambda i: (raw[i] - math.floor(raw[i]), -i), reverse=True)
    while remaining:
        progressed = False
        for index in order:
            if counts[index] < cap:
                counts[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("could not distribute the requested fine-token total")
    return tuple(counts)


def _fine_from_counts(counts_by_age: Iterable[int]) -> tuple[tuple[int, int], ...]:
    selected: list[tuple[int, int]] = []
    for age, count in enumerate(counts_by_age, start=1):
        if not 0 <= count <= FINE_CELLS_PER_TIME:
            raise ValueError("fine count per age must lie in [0,64]")
        selected.extend((age, cell) for cell in cell_order_for_age(age)[:count])
    return tuple(selected)


def _recent_fine(total: int, reach: int) -> tuple[tuple[int, int], ...]:
    counts = []
    remaining = total
    for _ in range(reach):
        count = min(FINE_CELLS_PER_TIME, remaining)
        counts.append(count)
        remaining -= count
    if remaining:
        raise ValueError("fine total exceeds temporal support")
    return _fine_from_counts(counts)


def frontier_plan(budget: int, policy: str, *, geometric_decay: float = 0.94) -> AllocationPlan:
    if budget not in FRONTIER_BUDGETS:
        raise ValueError(f"unregistered frontier budget: {budget}")
    if policy == "flat_recent":
        times = budget // FINE_CELLS_PER_TIME
        if times * FINE_CELLS_PER_TIME != budget:
            raise ValueError("flat budget must contain complete fine times")
        return AllocationPlan((), _recent_fine(budget, times), policy, overlap="exclusive")

    coarse_times = budget // 32
    fine_total = budget - coarse_times * COARSE_TOKENS_PER_TIME
    coarse_ages = tuple(range(1, coarse_times + 1))
    if policy == "balanced_recent":
        fine = _recent_fine(fine_total, coarse_times)
    elif policy == "linear_age":
        fine = _fine_from_counts(
            _weighted_counts(fine_total, range(coarse_times, 0, -1))
        )
    elif policy == "geometric_age":
        if not 0 < geometric_decay < 1:
            raise ValueError("geometric_decay must lie in (0,1)")
        fine = _fine_from_counts(
            _weighted_counts(
                fine_total,
                (geometric_decay ** index for index in range(coarse_times)),
            )
        )
    else:
        raise ValueError(f"unknown frontier policy: {policy}")
    plan = AllocationPlan(coarse_ages, fine, policy)
    if plan.token_count != budget:
        raise AssertionError("frontier plan violated exact budget")
    return plan


def allocation_1792_plan(policy: str, *, geometric_decay: float = 0.94) -> AllocationPlan:
    if policy in {"flat_recent", "balanced_recent", "linear_age", "geometric_age"}:
        return frontier_plan(1792, policy, geometric_decay=geometric_decay)
    definitions = {
        "fine_heavy": (28, 21),
        "coarse_heavy": (84, 7),
        "reach_optimized": (96, 4),
    }
    if policy not in definitions:
        raise ValueError(f"unknown 1792-token allocation: {policy}")
    coarse_times, complete_fine_times = definitions[policy]
    plan = AllocationPlan(
        tuple(range(1, coarse_times + 1)),
        _recent_fine(complete_fine_times * 64, coarse_times),
        policy,
    )
    if plan.token_count != 1792:
        raise AssertionError("ALC plan violated exact budget")
    return plan


def three_level_1792_plan(
    base_policy: str, *, geometric_decay: float = 0.94
) -> AllocationPlan:
    """Extend a selected two-level allocation to a fixed 2x2/4x4/8x8 pyramid.

    The top level spans all 96 available context positions (384 tokens). The remaining
    1,408 tokens preserve the selected two-level policy's coarse:fine payload ratio, with
    coarse payload rounded deterministically to the nearest 16-token slot. Fine cells are
    then placed using the selected policy's temporal rule inside the resulting coarse base.
    """

    if base_policy == "flat_recent":
        raise ValueError("three-level adaptation requires a non-flat hierarchy policy")
    base = allocation_1792_plan(base_policy, geometric_decay=geometric_decay)
    remaining = 1792 - 96 * 4
    base_coarse_payload = len(base.coarse_ages) * COARSE_TOKENS_PER_TIME
    coarse_payload = round(
        remaining * base_coarse_payload / (base_coarse_payload + base.fine_tokens) / 16
    ) * 16
    coarse_payload = min(max(coarse_payload, 16), remaining - 1)
    coarse_times = coarse_payload // COARSE_TOKENS_PER_TIME
    fine_total = remaining - coarse_payload
    if fine_total > coarse_times * FINE_CELLS_PER_TIME:
        raise ValueError("selected hierarchy ratio cannot be represented inside its coarse base")
    if base_policy == "linear_age":
        fine = _fine_from_counts(
            _weighted_counts(fine_total, range(coarse_times, 0, -1))
        )
    elif base_policy == "geometric_age":
        fine = _fine_from_counts(
            _weighted_counts(
                fine_total,
                (geometric_decay ** index for index in range(coarse_times)),
            )
        )
    else:
        fine = _recent_fine(fine_total, coarse_times)
    plan = AllocationPlan(
        coarse_ages=tuple(range(1, coarse_times + 1)),
        fine_age_cells=fine,
        policy=f"three_level_from_{base_policy}",
        overlap="nested",
        top_ages=tuple(range(1, 97)),
    )
    if plan.token_count != 1792 or plan.reach != 96:
        raise AssertionError("three-level plan violated the 1,792-token/96-position contract")
    return plan


def overlap_1792_plan(
    base_policy: str, mode: str, *, geometric_decay: float = 0.94
) -> AllocationPlan:
    """Apply a scientifically distinct overlap representation to one A4-ALC payload.

    Exclusive support preserves the base coarse/fine token counts and fine identities, then
    places the same number of coarse time slots at the newest ages not represented by fine
    tokens. It is feasible only when those disjoint supports fit inside the 96-frame TRB limit.
    Residual support keeps the nested token identities but represents fine content as the
    projected fine token minus its same-time coarse-parent projection.
    """

    if base_policy == "flat_recent":
        raise ValueError("overlap comparison requires a non-flat hierarchy policy")
    if mode not in {"nested", "exclusive", "residual"}:
        raise ValueError(f"unknown overlap mode: {mode}")
    base = allocation_1792_plan(base_policy, geometric_decay=geometric_decay)
    if mode == "nested":
        return AllocationPlan(
            base.coarse_ages, base.fine_age_cells, base.policy, overlap="nested"
        )
    if mode == "residual":
        fine_ages = {age for age, _ in base.fine_age_cells}
        if not fine_ages.issubset(base.coarse_ages):
            raise ValueError("residual fine tokens require a same-time coarse parent")
        return AllocationPlan(
            base.coarse_ages,
            base.fine_age_cells,
            base.policy,
            overlap="residual",
            fine_representation="coarse_embedding_residual",
        )
    fine_ages = {age for age, _ in base.fine_age_cells}
    available = tuple(age for age in range(1, 97) if age not in fine_ages)
    if len(available) < len(base.coarse_ages):
        raise ValueError(
            f"{base_policy} needs {len(base.coarse_ages)} coarse plus "
            f"{len(fine_ages)} fine-supported ages, exceeding the 96-frame exclusive limit"
        )
    plan = AllocationPlan(
        coarse_ages=available[: len(base.coarse_ages)],
        fine_age_cells=base.fine_age_cells,
        policy=base.policy,
        overlap="exclusive",
    )
    if plan.token_count != 1792:
        raise AssertionError("exclusive overlap plan violated exact budget")
    return plan


def coarse_expansion_plan(coarse_times: int) -> AllocationPlan:
    if coarse_times not in {0, 4, 8, 16, 32, 64}:
        raise ValueError("unregistered coarse-expansion position count")
    fine = _recent_fine(6 * 64, 6)
    # Exclusive support: coarse history begins immediately before the six fine times.
    coarse_ages = tuple(range(7, 7 + coarse_times))
    return AllocationPlan(coarse_ages, fine, "coarse_expansion", overlap="exclusive")


def _even_ages(count: int, reach: int) -> tuple[int, ...]:
    if count == 1:
        return (1,)
    return tuple(1 + (index * (reach - 1)) // (count - 1) for index in range(count))


def _geometric_ages(count: int, reach: int) -> tuple[int, ...]:
    if count == 1:
        return (1,)
    requested = [round(math.exp(index * math.log(reach) / (count - 1))) for index in range(count)]
    chosen: list[int] = []
    available = set(range(1, reach + 1))
    for value in requested:
        selection = min(available, key=lambda age: (abs(age - value), age))
        chosen.append(selection)
        available.remove(selection)
    return tuple(sorted(chosen))


def spatial_temporal_plan(budget: int, quota: int, temporal_policy: str) -> AllocationPlan:
    if budget not in {1152, 1792}:
        raise ValueError("STS budget must be 1152 or 1792")
    if quota not in {8, 16, 32, 64}:
        raise ValueError("STS quota must be 8, 16, 32, or 64")
    fine_total = 256
    coarse_times = (budget - fine_total) // COARSE_TOKENS_PER_TIME
    detail_times = fine_total // quota
    if temporal_policy == "recent_detail":
        ages = tuple(range(1, detail_times + 1))
    elif temporal_policy == "uniform_time":
        ages = _even_ages(detail_times, coarse_times)
    elif temporal_policy == "geometric_time":
        ages = _geometric_ages(detail_times, coarse_times)
    else:
        raise ValueError(f"unknown STS temporal policy: {temporal_policy}")
    fine = tuple(
        (age, cell)
        for age in ages
        for cell in cell_order_for_age(age)[:quota]
    )
    plan = AllocationPlan(tuple(range(1, coarse_times + 1)), fine, temporal_policy)
    if plan.token_count != budget or plan.fine_tokens != fine_total:
        raise AssertionError("STS plan violated exact budget")
    return plan
