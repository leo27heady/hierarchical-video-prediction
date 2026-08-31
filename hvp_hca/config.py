"""Validated experiment identities for Articles 4 and 5.

The configuration object deliberately encodes the scientific comparison contract,
not just model hyperparameters.  It fails closed on test-based checkpoint selection,
ambiguous memory semantics, and incompatible dynamic-routing positions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_PHASES = {"correctness", "screen", "confirmation", "development", "locked_test"}
_OVERLAPS = {"exclusive", "nested", "residual"}
_ROUTING = {"none", "static", "deterministic", "learned"}
_POSITIONS = {"static_slot", "coordinate_fourier", "coordinate_rope"}
_REVERSIBLE = {"none", "internal_blocks"}


@dataclass(frozen=True)
class ContextLevel:
    """One retained level with an explicit spatial and temporal cost."""

    name: str
    tokens_per_slot: int
    slots: int
    temporal_stride: int

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid level name: {self.name!r}")
        for field_name in ("tokens_per_slot", "slots", "temporal_stride"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

    @property
    def token_count(self) -> int:
        return self.tokens_per_slot * self.slots

    @property
    def nominal_reach(self) -> int:
        return self.slots * self.temporal_stride


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved, hashable comparison configuration.

    ``launch_authorized`` is false in repository templates.  A later immutable
    authorization ledger may permit a resolved hash after data and hardware
    manifests exist.
    """

    schema_version: int
    experiment_id: str
    article: int
    phase: str
    family: str
    dataset: str
    seed: int
    split_manifest: str
    checkpoint_selection_split: str
    evaluation_split: str
    budget_tokens: int
    levels: tuple[ContextLevel, ...]
    overlap: str
    allocation_policy: str
    target_horizon: int
    target_stride: int
    target_tokens_per_step: int
    positional_encoding: str
    routing: str
    detail_quota: int | None
    reversible_mode: str
    memory_endpoint: str
    launch_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("schema_version must be 2")
        if self.article not in {4, 5}:
            raise ValueError("article must be 4 or 5")
        if self.phase not in _PHASES:
            raise ValueError(f"unknown phase: {self.phase}")
        if self.overlap not in _OVERLAPS:
            raise ValueError(f"unknown overlap mode: {self.overlap}")
        if self.routing not in _ROUTING:
            raise ValueError(f"unknown routing mode: {self.routing}")
        if self.positional_encoding not in _POSITIONS:
            raise ValueError(f"unknown positional encoding: {self.positional_encoding}")
        if self.reversible_mode not in _REVERSIBLE:
            raise ValueError(
                "reversible_mode must be 'none' or 'internal_blocks'; "
                "external second-order rollout is not a v2 memory-only comparator"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.phase == "confirmation" and self.seed not in {0, 1, 2, 3}:
            raise ValueError("confirmation seeds are registered as 0, 1, 2, 3")
        if not self.experiment_id or not self.family or not self.dataset:
            raise ValueError("experiment_id, family, and dataset are required")
        if not self.split_manifest:
            raise ValueError("an immutable split manifest identity is required")
        if self.checkpoint_selection_split != "validation":
            raise ValueError("checkpoint selection must use validation, never test")
        if self.phase == "locked_test":
            if self.evaluation_split != "test":
                raise ValueError("locked_test must evaluate the test split")
        elif self.evaluation_split != "validation":
            raise ValueError("only a locked_test config may evaluate the test split")
        if self.budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        if not self.levels:
            raise ValueError("at least one context level is required")
        names = [level.name for level in self.levels]
        if len(names) != len(set(names)):
            raise ValueError("context level names must be unique")
        if self.context_tokens != self.budget_tokens:
            raise ValueError(
                f"declared budget {self.budget_tokens} does not match "
                f"level allocation {self.context_tokens}"
            )
        for field_name in ("target_horizon", "target_stride", "target_tokens_per_step"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.routing == "none":
            if self.detail_quota is not None:
                raise ValueError("detail_quota must be null when routing is disabled")
        elif self.detail_quota is None or self.detail_quota <= 0:
            raise ValueError("routing requires a positive fixed detail_quota")
        if self.routing in {"deterministic", "learned"}:
            if self.positional_encoding == "static_slot":
                raise ValueError(
                    "moving routes require coordinate-aware positional encoding"
                )
        if self.memory_endpoint not in {
            "retained_state",
            "inference_peak",
            "training_peak",
        }:
            raise ValueError("memory_endpoint must name one measured memory pool")
        if self.launch_authorized:
            raise ValueError(
                "repository configs fail closed; authorize a resolved hash in the "
                "experiment ledger instead of editing launch_authorized"
            )

    @property
    def context_tokens(self) -> int:
        return sum(level.token_count for level in self.levels)

    @property
    def nominal_reach(self) -> int:
        return max(level.nominal_reach for level in self.levels)

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def experiment_config_from_mapping(data: Mapping[str, Any]) -> ExperimentConfig:
    values = dict(data)
    raw_levels = values.get("levels")
    if not isinstance(raw_levels, list):
        raise ValueError("levels must be a JSON array")
    values["levels"] = tuple(ContextLevel(**dict(level)) for level in raw_levels)
    return ExperimentConfig(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"top-level JSON object required: {config_path}")
    return experiment_config_from_mapping(data)
