"""Fixed-budget adaptive fine-context feasibility study.

This module is intentionally separate from the frozen RHVD experiment registry.  It keeps the
published 48-coarse-frame, 1,152-token operating point and reallocates its 384 fine-context
tokens from six dense recent frames to selected 8x8-latent cells across the 48-frame history.
Several recent-reserved layouts are explicit development gates; none changes the frozen article
evidence.  Regression is used for the cheapest LoopMom gate and diffusion for the Moving-MNIST
transfer gate.  Reversible execution remains a separately gated follow-up stage.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from hvp_rev import hier_convae, rhvd
import hvp_rev.step1_mech as mechanism
from hvp_rev.evaluation.frame_metrics import evaluate_frames
from hvp_rev.step1_mech import ActionHierData, decode01, fg_mse_px_frames


GRID = 8
CELLS = GRID * GRID
CONTEXT_FRAMES = 48
FINE_CELLS_PER_FRAME = 8
CONTEXT_TOKENS = CONTEXT_FRAMES * 16 + CONTEXT_FRAMES * FINE_CELLS_PER_FRAME


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for context-only, fixed-count spatial routing."""

    mode: str = "variance"
    k: int = FINE_CELLS_PER_FRAME
    ema_alpha: float = 0.8
    entropy_bins: int = 16
    seed: int = 0
    layout: str = "recent2"
    dense_recent_frames: int = 2
    min_older_cells: int = 5

    def validate(self) -> None:
        if self.mode not in {"variance", "entropy", "motion", "hybrid", "random"}:
            raise ValueError(f"unknown selection mode: {self.mode}")
        if not 1 <= self.k <= CELLS:
            raise ValueError(f"k must lie in [1,{CELLS}]")
        if not 0.0 <= self.ema_alpha < 1.0:
            raise ValueError("ema_alpha must lie in [0,1)")
        if self.entropy_bins < 2:
            raise ValueError("entropy_bins must be at least 2")
        if self.layout not in {"uniform", "recent2", "recent4"}:
            raise ValueError(f"unknown allocation layout: {self.layout}")
        if self.layout == "recent2":
            if self.dense_recent_frames != 2 or self.min_older_cells != 5:
                raise ValueError("the first recent-reserved gate is frozen at 2 dense + 5 minimum")
        if self.layout == "recent4":
            if self.dense_recent_frames != 4 or self.min_older_cells != 2:
                raise ValueError("the conservative gate is frozen at 4 dense + 2 minimum")


def _as_frame_tensor(frames: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return float frames in [0,1] with shape (B,T,64,64)."""
    if isinstance(frames, np.ndarray) and not frames.flags.writeable:
        frames = np.array(frames, copy=True)
    x = torch.as_tensor(frames)
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise ValueError(f"expected one image channel, got {tuple(x.shape)}")
        x = x[:, :, 0]
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4 or tuple(x.shape[-2:]) != (64, 64):
        raise ValueError(f"frames must have shape (B,T,64,64), got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        x = x.float() / 255.0
    else:
        x = x.float()
        if float(x.detach().max()) > 1.5:
            x = x / 255.0
        elif float(x.detach().min()) < -0.1:
            x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def _patches(frames01: torch.Tensor) -> torch.Tensor:
    """Convert 64x64 frames into row-major 8x8 cells with 8x8 pixels per cell."""
    B, T, H, W = frames01.shape
    patch_h, patch_w = H // GRID, W // GRID
    return (
        frames01.reshape(B, T, GRID, patch_h, GRID, patch_w)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, T, CELLS, patch_h * patch_w)
    )


def _standardize_cells(score: torch.Tensor) -> torch.Tensor:
    mean = score.mean(dim=-1, keepdim=True)
    std = score.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (score - mean) / std


def cell_scores(frames: torch.Tensor | np.ndarray, cfg: SelectionConfig) -> torch.Tensor:
    """Score every 8x8 image cell using context frames only.

    The scores are allocation diagnostics, not a claim that local pixel entropy is identical to
    V-BLT predictive entropy.  ``hybrid`` combines cell variance and temporal change after
    per-frame standardization.  EMA smoothing is causal within the context sequence.
    """
    cfg.validate()
    x = _as_frame_tensor(frames)
    patches = _patches(x)
    variance = patches.var(dim=-1, unbiased=False)
    previous = torch.cat([patches[:, :1], patches[:, :-1]], dim=1)
    motion = (patches - previous).abs().mean(dim=-1)
    if cfg.mode == "variance":
        score = variance
    elif cfg.mode == "motion":
        score = motion
    elif cfg.mode == "hybrid":
        score = _standardize_cells(variance) + _standardize_cells(motion)
    elif cfg.mode == "entropy":
        quantized = (patches * cfg.entropy_bins).long().clamp(0, cfg.entropy_bins - 1)
        hist = torch.nn.functional.one_hot(quantized, cfg.entropy_bins).float().mean(dim=-2)
        score = -(hist * hist.clamp_min(1e-12).log()).sum(dim=-1)
    else:  # random is produced below without looking at pixels
        generator = torch.Generator(device=x.device).manual_seed(cfg.seed)
        score = torch.rand((x.shape[0], x.shape[1], CELLS), generator=generator, device=x.device)
    if cfg.ema_alpha > 0 and score.shape[1] > 1:
        smoothed = torch.empty_like(score)
        smoothed[:, 0] = score[:, 0]
        for t in range(1, score.shape[1]):
            smoothed[:, t] = cfg.ema_alpha * smoothed[:, t - 1] + (1 - cfg.ema_alpha) * score[:, t]
        score = smoothed
    return score


def select_fine_cells(frames: torch.Tensor | np.ndarray, cfg: SelectionConfig) -> torch.Tensor:
    """Return deterministic row-major cell ids with shape (B,T,K)."""
    score = cell_scores(frames, cfg)
    ids = score.topk(cfg.k, dim=-1, largest=True, sorted=False).indices
    return ids.sort(dim=-1).values


def select_spacetime_cells(
    frames: torch.Tensor | np.ndarray,
    cfg: SelectionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate exactly 384 cells under a declared recent-reserved layout.

    ``recent2`` retains two dense recent frames, gives five cells to every older frame and assigns
    the residual globally.  ``recent4`` retains four dense recent frames, gives two cells to every
    older frame and assigns its residual globally.  Both preserve the same fine-token budget.
    """
    cfg.validate()
    if cfg.layout not in {"recent2", "recent4"}:
        raise ValueError("spacetime allocation requires a recent-reserved layout")
    score = cell_scores(frames, cfg)
    B, T, N = score.shape
    if T != CONTEXT_FRAMES or N != CELLS:
        raise ValueError(
            f"recent-reserved allocation expects (B,48,64) scores, got {tuple(score.shape)}"
        )
    old_frames = T - cfg.dense_recent_frames
    base = score[:, :old_frames].topk(cfg.min_older_cells, dim=-1, sorted=False).indices
    selected = torch.zeros((B, old_frames, CELLS), dtype=torch.bool, device=score.device)
    selected.scatter_(-1, base, True)
    remaining = (
        CONTEXT_FRAMES * FINE_CELLS_PER_FRAME
        - cfg.dense_recent_frames * CELLS
        - old_frames * cfg.min_older_cells
    )
    if remaining < 0:
        raise ValueError("recent reservation exceeds the fixed fine-token budget")
    masked = score[:, :old_frames].masked_fill(selected, float("-inf")).reshape(B, -1)
    extra_flat = masked.topk(remaining, dim=-1, sorted=False).indices
    selected.reshape(B, -1).scatter_(1, extra_flat, True)
    recent = torch.ones(
        (B, cfg.dense_recent_frames, CELLS), dtype=torch.bool, device=score.device
    )
    mask = torch.cat([selected, recent], dim=1)
    flat = mask.reshape(B, -1).nonzero(as_tuple=False)
    token_budget = CONTEXT_FRAMES * FINE_CELLS_PER_FRAME
    if flat.shape[0] != B * token_budget:
        raise AssertionError(f"selected {flat.shape[0]} cells, expected {B * token_budget}")
    flat_ids = flat[:, 1].reshape(B, token_budget)
    frame_ids = flat_ids // CELLS
    cell_ids = flat_ids % CELLS
    return frame_ids, cell_ids


def selection_diagnostics(
    frames: torch.Tensor | np.ndarray,
    ids: torch.Tensor,
    frame_ids: torch.Tensor | None = None,
    foreground_threshold: float = 0.5,
) -> dict[str, float]:
    """Measure foreground capture and adjacent-mask stability for a selection."""
    x = _as_frame_tensor(frames).to(ids.device)
    fg = _patches((x > foreground_threshold).float()).sum(dim=-1)
    fg_cells = fg > 0
    selected_mask = torch.zeros_like(fg_cells)
    if frame_ids is None:
        selected_mask.scatter_(-1, ids, True)
    else:
        if frame_ids.shape != ids.shape:
            raise ValueError("frame_ids and ids must have the same shape")
        batch_ids = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(ids)
        selected_mask[batch_ids, frame_ids, ids] = True
    selected_fg = fg[selected_mask].sum()
    total_fg = fg.sum().clamp_min(1.0)
    recall_den = fg_cells.sum().clamp_min(1)
    recall = (selected_mask & fg_cells).sum().float() / recall_den
    if selected_mask.shape[1] > 1:
        left, right = selected_mask[:, :-1], selected_mask[:, 1:]
        inter = (left & right).sum(dim=-1).float()
        union = (left | right).sum(dim=-1).clamp_min(1).float()
        jaccard = (inter / union).mean()
    else:
        jaccard = torch.tensor(1.0, device=ids.device)
    return {
        "foreground_pixel_capture": float((selected_fg / total_fg).cpu()),
        "foreground_cell_recall": float(recall.cpu()),
        "adjacent_mask_jaccard": float(jaccard.cpu()),
    }


def adaptive_window_batch(
    data: ActionHierData,
    batch_size: int,
    cfg: SelectionConfig,
    *,
    train: bool,
    seed: int | None = None,
    frames: bool = False,
) -> dict:
    """Build the 48-coarse + 48xK selected-fine batch at the 1,152-token operating point."""
    batch = rhvd.window_batch(
        data, batch_size, CONTEXT_FRAMES, CONTEXT_FRAMES, train=train, seed=seed,
        frames=frames, return_indices=True,
    )
    raw = np.stack([
        data.clips_u8[int(ci), int(tj) - CONTEXT_FRAMES:int(tj)]
        for ci, tj in zip(batch["sampled_clips"], batch["sampled_starts"])
    ])
    if seed is None and cfg.mode == "random":
        route_seed = int(np.random.randint(0, 2**31 - 1))
    else:
        route_seed = cfg.seed if seed is None else seed
    local_cfg = SelectionConfig(**{**asdict(cfg), "seed": route_seed})
    if cfg.layout == "uniform":
        batch["ctx_f_ids"] = select_fine_cells(raw, local_cfg).to(rhvd.DEV)
    else:
        frame_ids, cell_ids = select_spacetime_cells(raw, local_cfg)
        batch["ctx_f_ids"] = cell_ids.to(rhvd.DEV)
        batch["ctx_f_frame_ids"] = frame_ids.to(rhvd.DEV)
    return batch


def build_model(selected: bool, layout: str = "recent2", mode: str = "reg") -> rhvd.WindowDiT:
    """Construct a state-dict-compatible fixed or selected-context predictor."""
    return rhvd.WindowDiT(
        n_ctx_c=CONTEXT_FRAMES,
        n_ctx_f=CONTEXT_FRAMES if selected else 6,
        mode=mode,
        fine_cells_per_frame=(FINE_CELLS_PER_FRAME if selected and layout == "uniform" else None),
        fine_token_budget=(CONTEXT_FRAMES * FINE_CELLS_PER_FRAME
                           if selected and layout in {"recent2", "recent4"} else None),
    ).to(rhvd.DEV)


def initialize_from_checkpoint(net: rhvd.WindowDiT, checkpoint: str | Path) -> dict:
    """Load the same frozen EMA initialization into either package layout."""
    state = torch.load(checkpoint, map_location=rhvd.DEV, weights_only=False)
    source = state.get("ema", state.get("model"))
    if source is None:
        raise ValueError(f"checkpoint has no model/ema state: {checkpoint}")
    net.load_state_dict(source, strict=True)
    return state


def _batch_for_variant(
    data: ActionHierData,
    batch_size: int,
    variant: str,
    cfg: SelectionConfig,
    *,
    train: bool,
    seed: int | None = None,
    frames: bool = False,
) -> dict:
    if variant == "fixed":
        return rhvd.window_batch(
            data, batch_size, CONTEXT_FRAMES, 6, train=train, seed=seed, frames=frames
        )
    return adaptive_window_batch(
        data, batch_size, cfg, train=train, seed=seed, frames=frames
    )


@torch.no_grad()
def evaluate_one_window(
    net: rhvd.WindowDiT,
    data: ActionHierData,
    ae,
    variant: str,
    cfg: SelectionConfig,
    *,
    batches: int = 8,
    batch_size: int = 8,
    nfe: int = 1,
    metric_threshold: float = 0.5,
) -> dict[str, float]:
    """Development-only four-frame evaluation on deterministic validation batches."""
    net.eval()
    wrapper = (rhvd.WindowDiffusion(net) if net.mode == "diff" else rhvd.WindowRegression(net))
    fg_values: list[float] = []
    full_values: list[float] = []
    union_values: list[float] = []
    for batch_index in range(batches):
        batch = _batch_for_variant(
            data, batch_size, variant, cfg, train=False, seed=1000 + batch_index, frames=True
        )
        torch.manual_seed(200000 + batch_index)
        _, pred_f = wrapper.sample(
            batch["ctx_c"], batch["ctx_f"], batch["actions"],
            ctx_f_ids=batch.get("ctx_f_ids"),
            ctx_f_frame_ids=batch.get("ctx_f_frame_ids"),
            n_steps=nfe,
        )
        for frame_index in range(rhvd.WIN):
            pred01 = decode01(ae, pred_f[:, frame_index])
            gt01 = (batch["gt"][:, frame_index].clamp(-1, 1) + 1) * 0.5
            fg_values.extend(fg_mse_px_frames(pred01, gt01).tolist())
            metrics = evaluate_frames(
                pred01.cpu().numpy(), gt01.cpu().numpy(), threshold=metric_threshold
            )
            full_values.extend(metrics["mse_full"].tolist())
            union_values.extend(metrics["mse_union"].tolist())
    return {
        "n_frame_records": len(fg_values),
        "fg_mse": float(np.mean(fg_values)),
        "mse_full": float(np.mean(full_values)),
        "mse_union": float(np.mean(union_values)),
    }


def train_continuation(
    net: rhvd.WindowDiT,
    data: ActionHierData,
    variant: str,
    cfg: SelectionConfig,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    lr: float,
) -> tuple[rhvd.WindowDiT, rhvd.EMA, list[float]]:
    """Matched continuation from a common frozen EMA initialization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    wrapper = (rhvd.WindowDiffusion(net) if net.mode == "diff" else rhvd.WindowRegression(net))
    ema = rhvd.EMA(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    net.train()
    for step in range(1, steps + 1):
        batch = _batch_for_variant(data, batch_size, variant, cfg, train=True)
        loss = rhvd.training_update(
            wrapper, net, ema, optimizer, batch, step, lr=lr, warmup=min(200, steps)
        )
        losses.append(float(loss.detach()))
        if step == 1 or step % max(1, min(100, steps // 5)) == 0 or step == steps:
            print(f"[{variant} {step}/{steps}] loss={np.mean(losses[-50:]):.6f}", flush=True)
    return net, ema, losses


@torch.no_grad()
def rollout_evaluate(
    net: rhvd.WindowDiT,
    data: ActionHierData,
    ae,
    variant: str,
    cfg: SelectionConfig,
    *,
    n_clips: int,
    warm: int = CONTEXT_FRAMES,
    chunk: int = 8,
    max_rollout_frames: int = 0,
    nfe: int = 1,
    metric_threshold: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Development rollout with a causally updated selector and full latent buffers."""
    if warm < CONTEXT_FRAMES:
        raise ValueError("warm-up must cover the 48-frame context reach")
    available = data.T - warm
    horizon = available if max_rollout_frames <= 0 else min(available, max_rollout_frames)
    horizon -= horizon % rhvd.WIN
    if horizon <= 0:
        raise ValueError(f"dataset T={data.T} has no complete rollout window after warm={warm}")
    clips = data.val_idx[:n_clips]
    wrapper = (rhvd.WindowDiffusion(net.eval()) if net.mode == "diff"
               else rhvd.WindowRegression(net.eval()))
    records: dict[str, list] = {
        "clip": [], "h": [], "fg_mse": [], "mse_full": [], "mse_union": [],
        "fg_recall": [], "fg_precision": [],
    }
    started = time.time()
    for c0 in range(0, len(clips), chunk):
        selected_clips = clips[c0:c0 + chunk]
        B = len(selected_clips)
        buf_c = data.zc[selected_clips, warm - CONTEXT_FRAMES:warm].clone().to(rhvd.DEV)
        buf_f = data.zf[selected_clips, warm - CONTEXT_FRAMES:warm].clone().to(rhvd.DEV)
        buf_frames = torch.from_numpy(
            data.clips_u8[selected_clips, warm - CONTEXT_FRAMES:warm].astype(np.float32) / 255.0
        ).to(rhvd.DEV)
        for offset in range(0, horizon, rhvd.WIN):
            target_start = warm + offset
            actions = torch.from_numpy(
                data.acts[selected_clips, target_start:target_start + rhvd.WIN].astype(np.int64)
            ).to(rhvd.DEV)
            if variant == "fixed":
                ctx_f = buf_f[:, -6:]
                frame_ids = cell_ids = None
            else:
                ctx_f = buf_f
                route_cfg = cfg
                if cfg.mode == "random":
                    route_cfg = SelectionConfig(
                        **{**asdict(cfg), "seed": cfg.seed + 100000 * c0 + offset}
                    )
                frame_ids, cell_ids = select_spacetime_cells(buf_frames, route_cfg)
            torch.manual_seed(rhvd.rollout_crn_seed(0, c0 // chunk, offset // rhvd.WIN))
            pred_c, pred_f = wrapper.sample(
                buf_c, ctx_f, actions,
                ctx_f_ids=cell_ids,
                ctx_f_frame_ids=frame_ids,
                n_steps=nfe,
            )
            pred_frames = decode01(ae, pred_f.reshape(B * rhvd.WIN, *pred_f.shape[2:]))
            pred_frames = pred_frames.reshape(B, rhvd.WIN, 1, 64, 64)
            buf_c = torch.cat([buf_c, pred_c], dim=1)[:, -CONTEXT_FRAMES:]
            buf_f = torch.cat([buf_f, pred_f], dim=1)[:, -CONTEXT_FRAMES:]
            buf_frames = torch.cat([buf_frames, pred_frames[:, :, 0]], dim=1)[:, -CONTEXT_FRAMES:]
            for j in range(rhvd.WIN):
                gt01 = torch.from_numpy(
                    data.clips_u8[selected_clips, target_start + j].astype(np.float32) / 255.0
                ).unsqueeze(1).to(rhvd.DEV)
                pred01 = pred_frames[:, j]
                fg = fg_mse_px_frames(pred01, gt01)
                metrics = evaluate_frames(
                    pred01.cpu().numpy(), gt01.cpu().numpy(), threshold=metric_threshold
                )
                for b, clip_id in enumerate(selected_clips):
                    records["clip"].append(int(clip_id))
                    records["h"].append(offset + j + 1)
                    records["fg_mse"].append(float(fg[b]))
                    for key in ("mse_full", "mse_union", "fg_recall", "fg_precision"):
                        records[key].append(float(metrics[key][b]))
        print(
            f"[rollout {variant}] clips {c0}..{c0 + B - 1}, horizon={horizon}, "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    arrays = {key: np.asarray(value) for key, value in records.items()}
    late = arrays["h"] > max(rhvd.WIN, horizon // 2)
    summary = {
        "n_clips": len(clips),
        "horizon": horizon,
        "n_frame_records": len(arrays["h"]),
        "fg_mse": float(arrays["fg_mse"].mean()),
        "fg_mse_late": float(arrays["fg_mse"][late].mean()),
        "mse_full": float(arrays["mse_full"].mean()),
        "mse_union": float(arrays["mse_union"].mean()),
        "fg_recall": float(arrays["fg_recall"].mean()),
        "fg_precision": float(arrays["fg_precision"].mean()),
        "seconds": time.time() - started,
    }
    return arrays, summary


def save_run(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    cfg: SelectionConfig,
    net: rhvd.WindowDiT,
    ema: rhvd.EMA,
    losses: list[float],
    before: dict,
    after_net: dict,
    after_ema: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "schema_version": 1,
        "study": "adaptive-hierarchy-feasibility",
        "evidence_class": "development-feasibility-not-frozen-article-evidence",
        "variant": args.variant,
        "mode": args.mode,
        "selection": asdict(cfg),
        "model": net.state_dict(),
        "ema": ema.shadow.state_dict(),
        "optimizer_updates": args.steps,
        "source_checkpoint": str(Path(args.init_checkpoint).resolve()),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    report = {
        "schema_version": 1,
        "study": checkpoint["study"],
        "evidence_class": checkpoint["evidence_class"],
        "variant": args.variant,
        "mode": args.mode,
        "selection": asdict(cfg),
        "context_tokens": CONTEXT_TOKENS,
        "context_reach_frames": CONTEXT_FRAMES,
        "before": before,
        "after_net": after_net,
        "after_ema": after_ema,
        "loss_first_50": float(np.mean(losses[:50])) if losses else None,
        "loss_last_50": float(np.mean(losses[-50:])) if losses else None,
        "optimizer_updates": args.steps,
        "batch_size": args.batch,
        "seed": args.seed,
        "bf16": bool(args.bf16),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def diagnose_raw_dataset(args: argparse.Namespace, cfg: SelectionConfig) -> dict:
    raw = np.load(args.train_path, mmap_mode="r")
    n_clips = min(args.diagnostic_clips, raw.shape[0])
    n_frames = min(args.diagnostic_frames, raw.shape[1])
    frames = np.asarray(raw[:n_clips, :n_frames])
    started = time.time()
    if cfg.layout == "uniform":
        frame_ids = None
        ids = select_fine_cells(frames, cfg)
    else:
        frame_ids, ids = select_spacetime_cells(frames, cfg)
    report = {
        "schema_version": 1,
        "dataset": str(Path(args.train_path).resolve()),
        "clips": n_clips,
        "frames_per_clip": n_frames,
        "selection": asdict(cfg),
        "context_tokens": CONTEXT_TOKENS,
        **selection_diagnostics(frames, ids, frame_ids=frame_ids),
        "seconds": time.time() - started,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnose", "train", "rollout"), default="diagnose")
    parser.add_argument("--variant", choices=("fixed", "adaptive", "random"), default="adaptive")
    parser.add_argument("--mode", choices=("reg", "diff"), default="reg")
    parser.add_argument("--selector", choices=("variance", "entropy", "motion", "hybrid", "random"),
                        default="variance")
    parser.add_argument("--layout", choices=("uniform", "recent2", "recent4"), default="recent2")
    parser.add_argument("--k", type=int, default=FINE_CELLS_PER_FRAME)
    parser.add_argument("--ema-alpha", type=float, default=0.8)
    parser.add_argument("--entropy-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-state", choices=("model", "ema"), default="model")
    parser.add_argument("--n-eval-clips", type=int, default=64)
    parser.add_argument("--rollout-chunk", type=int, default=8)
    parser.add_argument("--max-rollout-frames", type=int, default=0)
    parser.add_argument("--world", type=int, default=192)
    parser.add_argument("--wspeed", type=int, default=4)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--train-path", default="data/loop_mom_probe_300x100.npy")
    parser.add_argument("--test-path", default="data/loopmom_test_500x200.npy")
    parser.add_argument("--ae-path", default="logs_rhcp/hier_ae_mom.pt")
    parser.add_argument("--init-checkpoint", default="logs_rhcp/rhvd_reg-long_s0_t30000.pt")
    parser.add_argument("--output-root", default="D:/hierarchical-video-prediction-storage/feasibility")
    parser.add_argument("--diagnostic-clips", type=int, default=200)
    parser.add_argument("--diagnostic-frames", type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--acknowledge-feasibility", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_feasibility:
        raise RuntimeError("pass --acknowledge-feasibility to confirm this is new development evidence")
    if args.variant == "random":
        args.selector = "random"
    dense_recent, min_older = ((4, 2) if args.layout == "recent4" else (2, 5))
    cfg = SelectionConfig(
        mode=args.selector, k=args.k, ema_alpha=args.ema_alpha,
        entropy_bins=args.entropy_bins, seed=args.seed, layout=args.layout,
        dense_recent_frames=dense_recent, min_older_cells=min_older,
    )
    cfg.validate()
    if cfg.k != FINE_CELLS_PER_FRAME:
        raise ValueError("the first matched-budget study is frozen at k=8")
    if args.stage == "diagnose":
        report = diagnose_raw_dataset(args, cfg)
        root = Path(args.output_root)
        root.mkdir(parents=True, exist_ok=True)
        dataset_tag = Path(args.train_path).stem
        name = (
            f"routing_{dataset_tag}_{cfg.layout}_{cfg.mode}_"
            f"k{cfg.k}_a{cfg.ema_alpha:g}_s{cfg.seed}.json"
        )
        output = root / name
        if output.exists():
            raise FileExistsError(f"routing diagnostic already exists: {output}")
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    if rhvd.DEV != "cuda":
        raise RuntimeError("training feasibility gate requires CUDA")
    mechanism._BF16 = bool(args.bf16)
    ae = hier_convae.load(args.ae_path)
    data = ActionHierData(ae, args.train_path, args.train_path.replace(".npy", "_actions.npy"),
                          n_ctx=6, k=rhvd.WIN, W=args.world, speed=args.wspeed,
                          n_val=args.n_val)
    test = ActionHierData(ae, args.test_path, args.test_path.replace(".npy", "_actions.npy"),
                          n_ctx=6, k=rhvd.WIN, W=args.world, speed=args.wspeed,
                          n_val=args.n_val, stats=data.stats)
    selected = args.variant != "fixed"
    net = build_model(selected, cfg.layout, args.mode)
    if args.stage == "rollout":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for rollout")
        saved = torch.load(args.checkpoint, map_location=rhvd.DEV, weights_only=False)
        if args.checkpoint_state not in saved:
            raise ValueError(f"checkpoint has no state named {args.checkpoint_state}")
        net.load_state_dict(saved[args.checkpoint_state], strict=True)
        arrays, summary = rollout_evaluate(
            net, test, ae, args.variant, cfg,
            n_clips=args.n_eval_clips, chunk=args.rollout_chunk,
            max_rollout_frames=args.max_rollout_frames,
            nfe=args.nfe, metric_threshold=args.metric_threshold,
        )
        output_dir = Path(args.checkpoint).resolve().parent
        dataset_tag = Path(args.test_path).stem
        suffix = (
            f"rollout_{args.checkpoint_state}_{args.variant}_{cfg.mode}_"
            f"{dataset_tag}_nfe{args.nfe}_h{summary['horizon']}"
        )
        npz_path, json_path = output_dir / f"{suffix}.npz", output_dir / f"{suffix}.json"
        if npz_path.exists() or json_path.exists():
            raise FileExistsError(f"rollout output already exists: {suffix}")
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps({"output_dir": str(output_dir), "rollout": summary}, indent=2),
              flush=True)
        return
    initialize_from_checkpoint(net, args.init_checkpoint)
    before = evaluate_one_window(
        net, test, ae, args.variant, cfg, batches=args.eval_batches, batch_size=args.batch,
        nfe=args.nfe, metric_threshold=args.metric_threshold,
    )
    net, ema, losses = train_continuation(
        net, data, args.variant, cfg, steps=args.steps, batch_size=args.batch,
        seed=args.seed, lr=args.lr,
    )
    after_net = evaluate_one_window(
        net, test, ae, args.variant, cfg, batches=args.eval_batches, batch_size=args.batch,
        nfe=args.nfe, metric_threshold=args.metric_threshold,
    )
    after_ema = evaluate_one_window(
        ema.shadow, test, ae, args.variant, cfg, batches=args.eval_batches, batch_size=args.batch,
        nfe=args.nfe, metric_threshold=args.metric_threshold,
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / f"{stamp}_{args.variant}_{cfg.mode}_s{args.seed}_t{args.steps}"
    save_run(
        output_dir, args=args, cfg=cfg, net=net, ema=ema, losses=losses,
        before=before, after_net=after_net, after_ema=after_ema,
    )
    print(json.dumps({
        "output_dir": str(output_dir), "before": before,
        "after_net": after_net, "after_ema": after_ema,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
