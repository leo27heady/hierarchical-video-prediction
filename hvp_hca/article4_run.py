"""Resolved, hashable runtime specification for config-driven Article 4 fits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from hvp_hca.allocation import AllocationPlan, frontier_plan


@dataclass(frozen=True)
class Article4RunSpec:
    schema_version: int
    experiment_id: str
    phase: str
    family: str
    policy: str
    overlap: str
    budget_tokens: int
    seed: int
    dataset_manifest: str
    dataset_manifest_sha256: str
    ae_path: str
    ae_sha256: str
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
    log_every: int
    output_root: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Article4RunSpec schema_version must be 1")
        if self.phase != "correctness":
            raise ValueError("current runner is authorized only for A4-COR correctness fits")
        if self.family != "frontier":
            raise ValueError("current A4-COR runner supports frontier calibration only")
        if self.overlap not in {"exclusive", "nested"}:
            raise ValueError("calibration overlap must be exclusive or nested")
        if self.seed != 0:
            raise ValueError("schedule calibration is registered at seed 0")
        if self.predictor_mode != "diffusion_v_prediction":
            raise ValueError("Article 4 primary calibration uses diffusion v-prediction")
        if self.precision != "bf16_autocast_fp32_master":
            raise ValueError("calibration precision contract is bf16 autocast with fp32 master weights")
        if self.train_updates != 30000:
            raise ValueError("schedule calibration must run through 30,000 updates")
        if (self.width, self.depth, self.heads, self.batch_size) != (256, 6, 8, 16):
            raise ValueError("calibration instrument must be width256/depth6/heads8/batch16")
        if self.temporal_embedding_positions != 100:
            raise ValueError("Article 4 uses one shared 100-position temporal embedding table")
        if self.context_anchor != 96:
            raise ValueError("Article 4 uses frame 96 as the invariant first-target anchor")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 < self.learning_rate or not 0 <= self.weight_decay:
            raise ValueError("invalid optimizer parameters")
        if not 0 < self.ema_decay < 1:
            raise ValueError("ema_decay must lie in (0,1)")
        for name in (
            "warmup_updates",
            "validation_every",
            "validation_batches",
            "validation_batch_size",
            "validation_nfe",
            "checkpoint_every",
            "log_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.train_updates % self.validation_every:
            raise ValueError("train_updates must be divisible by validation_every")
        if self.validation_every % self.checkpoint_every:
            raise ValueError("validation milestones must also be resumable checkpoints")
        if len(self.dataset_manifest_sha256) != 64 or len(self.ae_sha256) != 64:
            raise ValueError("manifest and autoencoder SHA-256 values are required")
        if self.plan.token_count != self.budget_tokens:
            raise ValueError("allocation plan does not match budget_tokens")
        expected_overlap = "exclusive" if self.policy == "flat_recent" else "nested"
        if self.overlap != expected_overlap:
            raise ValueError(f"{self.policy} calibration requires {expected_overlap} overlap")

    @property
    def plan(self) -> AllocationPlan:
        return frontier_plan(self.budget_tokens, self.policy)

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_article4_run_spec(path: str | Path) -> Article4RunSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Article 4 run config must be a JSON object")
    return Article4RunSpec(**raw)


def validate_authorization(spec: Article4RunSpec, path: str | Path) -> dict[str, Any]:
    authorization = json.loads(Path(path).read_text(encoding="utf-8"))
    if authorization.get("purpose") != "article4_schedule_calibration":
        raise ValueError("authorization purpose mismatch")
    if authorization.get("status") != "authorized":
        raise ValueError("calibration authorization is not active")
    identities = authorization.get("authorized_config_hashes", {})
    if identities.get(spec.experiment_id) != spec.content_hash:
        raise ValueError("run spec hash is not present in the authorization ledger")
    if spec.train_updates > int(authorization.get("max_updates_per_run", 0)):
        raise ValueError("run exceeds authorization update limit")
    if spec.experiment_id not in authorization.get("authorized_experiment_ids", []):
        raise ValueError("experiment id is not authorized")
    return authorization
