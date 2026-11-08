"""Scientific compatibility checks for paired run artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REQUIRED_EQUAL = (
    "dataset_hash",
    "ae_hash",
    "source_split",
    "metric_version",
    "warmup_frames",
    "rollout_frames",
    "evaluation_clips",
    "nfe",
    "sampler",
    "precision",
)


@dataclass(frozen=True)
class ComparisonReport:
    compatible: bool
    mismatches: Mapping[str, tuple[Any, Any]]
    matching_definitions: tuple[str, ...]

    def require(self) -> None:
        if not self.compatible:
            detail = ", ".join(f"{k}: {a!r} != {b!r}" for k, (a, b) in self.mismatches.items())
            raise ValueError(f"scientifically incompatible comparison: {detail}")


def validate_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> ComparisonReport:
    mismatches = {field: (left.get(field), right.get(field)) for field in REQUIRED_EQUAL
                  if left.get(field) != right.get(field)}
    matched = []
    for label, fields in {
        "context-token-matched": ("context_tokens",),
        "target-interface-matched": ("target_tokens", "target_slots", "loss_weights"),
        "parameter-matched": ("parameters",),
        "example-matched": ("train_examples",),
        "update-matched": ("optimizer_updates",),
        "batch-matched": ("batch_size",),
        "total-nfe-matched": ("total_nfe",),
    }.items():
        if all(field in left and field in right and left[field] == right[field] for field in fields):
            matched.append(label)
    return ComparisonReport(not mismatches, mismatches, tuple(matched))
