"""Frozen-dataclass config tree (single source of truth for shapes).

`TopologyConfig` defines the encoder downsample schedule; every per-stage spatial
and temporal size is *derived* from it (no re-declaration), so the predictor and
the masks read shapes from one place. YAML loading supports `_base_` inheritance
and dot-path CLI overrides; unknown keys are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, get_type_hints

import yaml


def causal_output_length(t_in: int, kernel_t: int = 3, stride_t: int = 2) -> int:
    """Output length of a causal Conv with left-pad (K-1) and the given stride."""
    return (t_in - 1) // stride_t + 1


# --------------------------------------------------------------------------- #
# Topology — the encoder downsample schedule; all stage shapes derive from it. #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TopologyConfig:
    # default = 64x64 operating resolution: 64 -> BOT 16 -> MID 4 -> TOP 1, T 9/5/3
    image_size: int = 64
    image_c: int = 3
    base_ch: int = 32
    ch_mult: Tuple[int, ...] = (1, 2, 2, 2, 2, 4, 4)         # 7 levels -> 6 downsamples
    temporal_downsample: Tuple[bool, ...] = (False, False, True, False, True, False)
    num_res_blocks: int = 1
    tap_after: Tuple[int, ...] = (1, 3, 5)                   # downsample idx after which to tap (bot..top)
    stage_latent_ch: Tuple[int, ...] = (16, 8, 8)           # bot, mid, top continuous latent channels
    decode_width: int = 64
    sequence_length: int = 9

    # ---- derived ---------------------------------------------------------- #
    def n_stages(self) -> int:
        return len(self.tap_after)

    def n_down(self) -> int:
        return len(self.ch_mult) - 1

    def stage_spatial(self) -> Tuple[int, ...]:
        """H==W at each tapped stage (bot, mid, top order = tap_after order)."""
        return tuple(self.image_size // (2 ** (i + 1)) for i in self.tap_after)

    def stage_temporal(self) -> Tuple[int, ...]:
        """Temporal length at each tapped stage, from the cumulative causal strides."""
        out = []
        for tap in self.tap_after:
            t = self.sequence_length
            for j in range(tap + 1):
                if self.temporal_downsample[j]:
                    t = causal_output_length(t)
            out.append(t)
        return tuple(out)

    def level_width(self, level: int) -> int:
        return self.base_ch * self.ch_mult[level]

    def validate(self) -> None:
        if len(self.temporal_downsample) != self.n_down():
            raise ValueError(
                f"temporal_downsample has {len(self.temporal_downsample)} flags, "
                f"expected n_down={self.n_down()}"
            )
        if len(self.stage_latent_ch) != self.n_stages():
            raise ValueError("stage_latent_ch must have one entry per tap_after stage")
        if any(t >= self.n_down() + 1 for t in self.tap_after):
            raise ValueError("tap_after indices must be < number of encoder levels")
        if self.sequence_length % 4 != 1:
            raise ValueError("sequence_length must be ≡ 1 (mod 4) for the causal encoder")
        if tuple(sorted(self.tap_after)) != tuple(self.tap_after):
            raise ValueError("tap_after must be ascending (bot..top = fine..coarse)")


@dataclass(frozen=True)
class VAEConfig:
    recon_loss: str = "l1"                       # "l1" | "l2"
    recon_weight: float = 1.0
    progressive: bool = True                     # coarse-first curriculum
    progressive_steps_per_depth: int = 400
    depth_loss_weights: Optional[Tuple[float, ...]] = None  # per-depth recon weight
    vicreg_weight: float = 1.0                   # variance-hinge weight
    vicreg_gamma: float = 1.0                    # per-dim std floor
    vicreg_stage_scale: Tuple[float, ...] = (0.1, 1.0, 1.0)  # bot, mid, top
    vicreg_cov_weight: float = 0.04


@dataclass(frozen=True)
class SchedConfig:
    """Anti-drift (P2) grad-enabled self-rollout schedule. All defaults = today's behavior (no-op),
    so configs that omit `predictor.sched` are byte-identical to before."""
    rollout_enabled: bool = False
    rollout_steps: int = 1                        # K feedback steps (seq[i] predicts shift i+2)
    lambda_rollout: float = 0.0                   # weight of the self-rollout term (ramped)
    rollout_warmup_steps: int = 2000              # ramp lambda_rollout 0->target, AFTER pred warmup
    sample_rollout: bool = False                  # feedback via head_mode (False) | head_sample (True)
    detach_feedback: bool = True                  # stop-grad on fed-back latents (grad truncation)

    def validate(self) -> None:
        if self.rollout_steps < 1:
            raise ValueError("predictor.sched.rollout_steps must be >= 1")
        if self.lambda_rollout < 0:
            raise ValueError("predictor.sched.lambda_rollout must be >= 0")


@dataclass(frozen=True)
class PredictorConfig:
    # --- per-stage capacity (scalars broadcast to all stages; tuples override per stage bot..top) ---
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    stage_dims: Optional[Tuple[int, ...]] = None
    stage_heads: Optional[Tuple[int, ...]] = None
    stage_layers: Optional[Tuple[int, ...]] = None
    mlp_ratio: int = 4
    dropout: float = 0.0
    # --- attention / positional ---
    rope_base: float = 10000.0
    qk_norm: bool = True
    # --- head / conditioning ---
    head: str = "l2"                             # "l2" | "gaussian" | "gmm"
    head_components: int = 4                      # GMM mixture components (head="gmm")
    conditioning: str = "cross_attn"             # "cross_attn" | "none"
    use_cascade: bool = True                     # if False, stages predict independently
    detach_parent: bool = True
    # --- per-stage attention windowing (bot, mid, top); 0 = unlimited ---
    # forces fine stages to lean on the parent: motion/context must come via cross-attention.
    stage_t_window: Optional[Tuple[int, ...]] = None   # frames back incl self; 1 = self-frame only
    stage_s_window: Optional[Tuple[int, ...]] = None   # square spatial window side; 0 = global
    # --- predictive loss ---
    lambda_pred: float = 0.25
    pred_warmup_steps: int = 1000                # ramp lambda 0->lambda_pred
    stage_pred_weight: Tuple[float, ...] = (1.0, 2.0, 2.0)  # bot, mid, top
    sched: SchedConfig = field(default_factory=SchedConfig)  # P2 anti-drift self-rollout (default no-op)

    def windows(self, n_stages: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        tw = self.stage_t_window or tuple(0 for _ in range(n_stages))
        sw = self.stage_s_window or tuple(0 for _ in range(n_stages))
        for name, t in (("stage_t_window", tw), ("stage_s_window", sw)):
            if len(t) != n_stages:
                raise ValueError(f"predictor.{name} must have {n_stages} entries, got {t}")
        return tw, sw

    def resolved(self, n_stages: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        dims = self.stage_dims or tuple(self.dim for _ in range(n_stages))
        heads = self.stage_heads or tuple(self.n_heads for _ in range(n_stages))
        layers = self.stage_layers or tuple(self.n_layers for _ in range(n_stages))
        for name, t in (("stage_dims", dims), ("stage_heads", heads), ("stage_layers", layers)):
            if len(t) != n_stages:
                raise ValueError(f"predictor.{name} must have {n_stages} entries, got {t}")
        for d, h in zip(dims, heads):
            if d % h != 0:
                raise ValueError(f"stage dim {d} not divisible by n_heads {h}")
        return dims, heads, layers


@dataclass(frozen=True)
class DataConfig:
    cache_root: str = "C:/Users/leoni/Documents/data/shape_cache"
    train_hash: str = "e075debc82fc"             # 64x64, T=9, 600 clips (independently rendered)
    val_hash: str = "9dffa66fdc0e"               # 64x64, T=9, 200 clips (true cross-split)
    val_long_hash: str = "c5691972cc49"          # 64x64, T=33, 120 clips (long-horizon rollout eval)
    image_size: int = 64
    max_clips: Optional[int] = None
    num_workers: int = 0


@dataclass(frozen=True)
class OptimConfig:
    optimizer: str = "adamw"                     # "adam" | "adamw"
    lr: float = 3e-4
    min_lr: float = 0.0
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    batch_size: int = 16
    grad_clip: float = 1.0
    max_steps: int = 2000
    lr_schedule: str = "constant"                # "constant" | "cosine" | "warmup_cosine"
    warmup_steps: int = 0
    amp: str = "bf16"                            # "bf16" | "off"
    seed: int = 0


@dataclass(frozen=True)
class RunConfig:
    out_dir: str = "logs"
    log_every: int = 50
    val_every: int = 500
    val_max_batches: int = 8
    save_every: int = 0                          # 0 = save only at the end
    max_checkpoints_keep: int = 3
    wandb: bool = False
    wandb_project: str = "hvp-vqfree"
    wandb_name: Optional[str] = None
    wandb_mode: str = "online"                   # "online" | "offline" | "disabled"
    n_log_images: int = 6                        # clips shown in the wandb image table


@dataclass(frozen=True)
class Config:
    phase: str = "fit_oneshot"                   # "fit_oneshot" | "cache_features" | "fit_predictor"
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self) -> "Config":
        self.topology.validate()
        self.predictor.resolved(self.topology.n_stages())
        self.predictor.sched.validate()
        if self.data.image_size != self.topology.image_size:
            raise ValueError("data.image_size must match topology.image_size")
        if len(self.predictor.stage_pred_weight) != self.topology.n_stages():
            raise ValueError("predictor.stage_pred_weight must have one entry per stage")
        return self

    # ---- (de)serialization ------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Config":
        return _build_dataclass(cls, d)

    @classmethod
    def from_yaml(cls, path, overrides: Optional[list[str]] = None) -> "Config":
        raw = load_yaml_config(Path(path))
        if overrides:
            raw = apply_cli_overrides(raw, overrides)
        return cls.from_dict(raw).validate()


# --------------------------------------------------------------------------- #
# YAML + override plumbing                                                     #
# --------------------------------------------------------------------------- #
def load_yaml_config(path: Path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if "_base_" in cfg:
        base = load_yaml_config(path.parent / cfg.pop("_base_"))
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = dict(cfg)
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def _build_dataclass(cls, d: Mapping[str, Any]):
    if not isinstance(d, Mapping):
        return d
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    hints = get_type_hints(cls)  # resolves string annotations (PEP 563) to real types
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in d:
            continue
        val = d[name]
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(val, Mapping):
            kwargs[name] = _build_dataclass(ftype, val)
        elif isinstance(val, list):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)
