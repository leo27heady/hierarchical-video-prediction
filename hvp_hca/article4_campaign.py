"""Hash-locked run contracts for the complete Article 4 30k campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from hvp_hca.allocation import (
    AllocationPlan,
    allocation_1792_plan,
    coarse_expansion_plan,
    frontier_plan,
    overlap_1792_plan,
    spatial_temporal_plan,
    three_level_1792_plan,
)


PRIMARY_FAMILIES = {"FRT", "ALC", "CEX", "STS", "LVL", "OVR"}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_plan(kind: str, arguments: dict[str, Any]) -> AllocationPlan:
    args = dict(arguments)
    if kind == "frontier":
        return frontier_plan(
            int(args["budget"]),
            str(args["policy"]),
            geometric_decay=float(args.get("geometric_decay", 0.94)),
        )
    if kind == "allocation_1792":
        return allocation_1792_plan(
            str(args["policy"]),
            geometric_decay=float(args.get("geometric_decay", 0.94)),
        )
    if kind == "coarse_expansion":
        return coarse_expansion_plan(int(args["coarse_times"]))
    if kind == "spatial_temporal":
        return spatial_temporal_plan(
            int(args["budget"]), int(args["quota"]), str(args["temporal_policy"])
        )
    if kind == "three_level_1792":
        return three_level_1792_plan(
            str(args["base_policy"]),
            geometric_decay=float(args.get("geometric_decay", 0.94)),
        )
    if kind == "overlap_1792":
        return overlap_1792_plan(
            str(args["base_policy"]),
            str(args["mode"]),
            geometric_decay=float(args.get("geometric_decay", 0.94)),
        )
    raise ValueError(f"unknown Article 4 allocation kind: {kind}")


@dataclass(frozen=True)
class Article4CampaignRunSpec:
    schema_version: int
    campaign_id: str
    experiment_id: str
    phase: str
    family: str
    policy: str
    overlap: str
    budget_tokens: int
    seed: int
    allocation_kind: str
    allocation_args: dict[str, Any]
    dataset_manifest: str
    dataset_manifest_sha256: str
    ae_path: str
    ae_sha256: str
    top_ae_path: str | None
    top_ae_sha256: str | None
    predictor_mode: str
    width: int
    depth: int
    heads: int
    temporal_embedding_positions: int
    context_anchor: int
    batch_size: int
    train_updates: int
    learning_rate: float
    weight_decay: float
    warmup_updates: int
    precision: str
    mixed_precision_rmsnorm: bool
    ema_decay: float
    validation_every: int
    validation_batches: int
    validation_batch_size: int
    validation_nfe: int
    checkpoint_every: int
    milestone_every: int
    image_every: int
    log_every: int
    output_root: str
    wandb_project: str
    wandb_group: str
    primary_metric: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("Article4CampaignRunSpec schema_version must be 2")
        if self.phase != "primary_30k":
            raise ValueError("Article 4 campaign phase must be primary_30k")
        if self.family not in PRIMARY_FAMILIES:
            raise ValueError(f"unknown Article 4 family: {self.family}")
        if self.seed not in {0, 1}:
            raise ValueError("primary Article 4 seeds are exactly 0 and 1")
        if self.predictor_mode != "diffusion_v_prediction":
            raise ValueError("Article 4 primary predictor is diffusion v-prediction")
        if self.precision != "bf16_autocast_fp32_master":
            raise ValueError("Article 4 precision contract mismatch")
        if self.train_updates != 30_000:
            raise ValueError("the confirmed Article 4 campaign schedule is 30,000 updates")
        if (self.width, self.depth, self.heads, self.batch_size) != (256, 6, 8, 16):
            raise ValueError("campaign instrument must be width256/depth6/heads8/batch16")
        if self.temporal_embedding_positions != 100 or self.context_anchor != 96:
            raise ValueError("campaign uses the shared 100-position table and target anchor 96")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.primary_metric != "horizon4_foreground_mse":
            raise ValueError("the locked primary metric is horizon4_foreground_mse")
        if not self.wandb_project or not self.wandb_group:
            raise ValueError("W&B project and group are required")
        for name in (
            "warmup_updates",
            "validation_every",
            "validation_batches",
            "validation_batch_size",
            "validation_nfe",
            "checkpoint_every",
            "milestone_every",
            "image_every",
            "log_every",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.train_updates % self.validation_every:
            raise ValueError("validation cadence must divide 30k exactly")
        if self.validation_every % self.checkpoint_every:
            raise ValueError("every validation point must follow a resumable checkpoint")
        if self.train_updates % self.milestone_every:
            raise ValueError("milestone cadence must divide 30k exactly")
        if self.image_every % self.validation_every:
            raise ValueError("image cadence must be a validation cadence multiple")
        if len(self.dataset_manifest_sha256) != 64 or len(self.ae_sha256) != 64:
            raise ValueError("dataset and base autoencoder SHA-256 values are required")
        if (self.top_ae_path is None) != (self.top_ae_sha256 is None):
            raise ValueError("top autoencoder path and hash must either both be set or both be null")
        plan = self.plan
        if plan.token_count != self.budget_tokens:
            raise ValueError("resolved allocation does not match budget_tokens")
        if plan.overlap != self.overlap:
            raise ValueError(f"resolved overlap {plan.overlap} != declared {self.overlap}")
        if plan.reach > self.context_anchor:
            raise ValueError("allocation exceeds the common context anchor")
        if plan.top_ages and self.top_ae_path is None:
            raise ValueError("three-level allocation requires the frozen top autoencoder")
        if not plan.top_ages and self.top_ae_path is not None:
            raise ValueError("top autoencoder is only allowed for A4-LVL three-level runs")

    @property
    def plan(self) -> AllocationPlan:
        return resolve_plan(self.allocation_kind, self.allocation_args)

    @property
    def levels(self) -> int:
        plan = self.plan
        return 1 + int(bool(plan.coarse_ages)) + int(bool(plan.top_ages))

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.canonical_dict())

    @property
    def allocation_hash(self) -> str:
        plan = self.plan
        return canonical_sha256(
            {
                "top_ages": plan.top_ages,
                "coarse_ages": plan.coarse_ages,
                "fine_age_cells": plan.fine_age_cells,
                "policy": plan.policy,
                "overlap": plan.overlap,
                "fine_representation": plan.fine_representation,
            }
        )


def load_campaign_run_spec(path: str | Path) -> Article4CampaignRunSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Article 4 campaign run spec must be a JSON object")
    return Article4CampaignRunSpec(**raw)


def validate_campaign_authorization(
    spec: Article4CampaignRunSpec,
    campaign_path: str | Path,
    *,
    resolution_path: str | Path | None = None,
) -> tuple[dict[str, Any], str, str | None]:
    path = Path(campaign_path)
    campaign = json.loads(path.read_text(encoding="utf-8"))
    campaign_hash = file_sha256(path)
    if campaign.get("campaign_id") != spec.campaign_id:
        raise ValueError("campaign id mismatch")
    if campaign.get("status") != "authorized_30k":
        raise ValueError("campaign is not authorized for 30k launch")
    if int(campaign.get("confirmed_train_updates", 0)) != 30_000:
        raise ValueError("campaign update authorization mismatch")
    expected = campaign.get("authorized_runs", {}).get(spec.experiment_id)
    if expected == spec.content_hash:
        return campaign, campaign_hash, None
    if resolution_path is None:
        raise ValueError("run is not in the campaign authorization ledger")
    resolved_path = Path(resolution_path)
    resolution = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolution_hash = file_sha256(resolved_path)
    if resolution.get("campaign_id") != spec.campaign_id:
        raise ValueError("dependency resolution campaign id mismatch")
    if resolution.get("parent_campaign_sha256") != campaign_hash:
        raise ValueError("dependency resolution does not bind the current campaign manifest")
    if resolution.get("status") != "resolved":
        raise ValueError("dependency resolution is not complete")
    if resolution.get("authorized_runs", {}).get(spec.experiment_id) != spec.content_hash:
        raise ValueError("dependent run is not hash-authorized by the resolution ledger")
    return campaign, campaign_hash, resolution_hash
