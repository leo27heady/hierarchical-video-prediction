"""Locked selection rules for Article 4 post-training analysis.

The rules operate only on final 30k development-validation records. They deliberately identify
configuration pairs (both seeds), never a single lucky seed or an intermediate checkpoint.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Any, Iterable


PRIMARY_METRIC = "horizon4_foreground_mse"


@dataclass(frozen=True)
class ConfigurationResult:
    key: str
    policy: str
    budget_tokens: int
    levels: int
    overlap: str
    mean_primary: float
    seed_values: tuple[float, float]
    run_entries: tuple[dict[str, Any], dict[str, Any]]
    context_tokens: int
    state_bytes_per_sample_bf16: int


def pair_key(row: dict[str, Any]) -> str:
    """Make a seed-independent identity from one completed run record."""

    spec = row["spec"]
    return "|".join(
        (
            str(spec["allocation_kind"]),
            json.dumps(spec["allocation_args"], sort_keys=True, separators=(",", ":")),
            str(spec["budget_tokens"]),
            str(spec["policy"]),
            str(spec["overlap"]),
        )
    )


def pair_results(rows: Iterable[dict[str, Any]]) -> list[ConfigurationResult]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["step"]) != 30_000:
            raise ValueError("post-campaign selection accepts only final 30k records")
        grouped[pair_key(row)].append(row)
    results = []
    for key, members in grouped.items():
        members.sort(key=lambda item: int(item["spec"]["seed"]))
        seeds = [int(item["spec"]["seed"]) for item in members]
        if seeds != [0, 1]:
            raise ValueError(f"configuration {key} does not contain exactly seeds 0 and 1")
        values = tuple(float(item["metrics"][PRIMARY_METRIC]) for item in members)
        first = members[0]
        results.append(
            ConfigurationResult(
                key=key,
                policy=str(first["spec"]["policy"]),
                budget_tokens=int(first["spec"]["budget_tokens"]),
                levels=int(first["levels"]),
                overlap=str(first["spec"]["overlap"]),
                mean_primary=sum(values) / 2,
                seed_values=(values[0], values[1]),
                run_entries=(members[0], members[1]),
                context_tokens=int(first["context_tokens"]),
                state_bytes_per_sample_bf16=int(first["state_bytes_per_sample_bf16"]),
            )
        )
    return sorted(results, key=lambda result: result.key)


def select_postcampaign(results: Iterable[ConfigurationResult]) -> dict[str, ConfigurationResult]:
    """Apply the predeclared A4-MEC and sealed-confirmation selection rules."""

    values = list(results)
    # The mechanism contrast is matched at the single A4-ALC budget. FRT contributes the three
    # reused policies; the four ALC-only policies use their own family-labelled run files.
    alc = [
        result
        for result in values
        if result.budget_tokens == 1792
        and all(
            entry["spec"]["family"] in {"FRT", "ALC"}
            for entry in result.run_entries
        )
        and result.overlap in {"nested", "exclusive"}
    ]
    flat = [result for result in alc if result.policy == "flat_recent"]
    hierarchy = [result for result in alc if result.policy != "flat_recent"]
    if len(flat) != 1 or len(hierarchy) != 6:
        raise ValueError("the complete seven-policy A4-ALC comparison is unavailable")
    best_flat = flat[0]
    best_hierarchy = min(hierarchy, key=lambda result: (result.mean_primary, result.key))
    failure_hierarchy = max(hierarchy, key=lambda result: (result.mean_primary, result.key))

    # Registered 5% quality tolerance. Among hierarchical configurations whose mean final metric
    # is no worse than 1.05x the matched Flat reference, choose the smallest stored state; quality
    # and stable key break ties. This selection never consults the sealed holdout.
    eligible = [
        result
        for result in values
        if result.levels >= 2
        and result.mean_primary <= 1.05 * best_flat.mean_primary
    ]
    if not eligible:
        # Explicit fallback preserves a defined evaluation shortlist without pretending a Pareto
        # win exists. The ledger records the failure through the same selected checkpoint.
        memory_pareto = best_hierarchy
    else:
        memory_pareto = min(
            eligible,
            key=lambda result: (
                result.state_bytes_per_sample_bf16,
                result.context_tokens,
                result.mean_primary,
                result.key,
            ),
        )
    return {
        "best_flat_1792": best_flat,
        "best_hierarchy_1792": best_hierarchy,
        "failure_hierarchy_1792": failure_hierarchy,
        "memory_pareto_within_5pct": memory_pareto,
    }
