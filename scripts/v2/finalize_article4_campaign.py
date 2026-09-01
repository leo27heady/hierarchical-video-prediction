"""Finalize Article 4 after all 86 training runs without test-driven reselection.

Stages are restart-safe: aggregate final development metrics, write a hash-bound selection ledger,
generate/verify the sealed TRB holdout once, then evaluate the union of A4-MEC and confirmation
roles. Local JSON/JSONL/NPZ/PNG evidence is canonical; W&B receives a best-effort summary mirror.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.article4_campaign import file_sha256, load_campaign_run_spec
from hvp_hca.data import PackedWindowBatcher, load_trb_confirmation, load_trb_development
from hvp_hca.performance import enable_mixed_precision_rmsnorm
from hvp_hca.postcampaign import pair_results, select_postcampaign
from hvp_hca.tracking import WandbMirror
from hvp_rev import hier3, hier_convae
from hvp_rev import rhvd as legacy
from hvp_rev.step1_mech import decode01
from scripts.v2.train_article4_campaign import forward_kwargs, save_panel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "experiments/v2/campaigns/article4_primary_30k_2026-09-01.json"
ARTIFACT_ROOT = ROOT / "artifacts/v2/article4_primary_30k/post_training"
RESOLUTION_PATH = ROOT / "artifacts/v2/article4_primary_30k/dependency_resolution.json"
CONFIRMATION_ROOT = ROOT / "artifacts/v2/article4_primary_30k/confirmation"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_hash(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_directory(spec: Any) -> Path:
    return ROOT / spec.output_root / f"{spec.experiment_id}__{spec.content_hash[:12]}"


def final_validation(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in records if int(row["step"]) == 30_000]
    if len(selected) != 1:
        raise ValueError(f"expected one final 30k validation record in {path}")
    return selected[0]


def all_entries(campaign: dict[str, Any], campaign_path: Path) -> list[dict[str, Any]]:
    if not RESOLUTION_PATH.exists():
        raise FileNotFoundError("dependent A4-LVL/A4-OVR resolution is missing")
    resolution = json.loads(RESOLUTION_PATH.read_text(encoding="utf-8"))
    if resolution.get("campaign_id") != campaign["campaign_id"]:
        raise ValueError("dependency resolution campaign mismatch")
    if resolution.get("parent_campaign_sha256") != file_sha256(campaign_path):
        raise ValueError("dependency resolution is not bound to the current campaign")
    if resolution.get("status") != "resolved":
        raise ValueError("dependency resolution is not locked")
    entries = list(campaign["run_order"]) + list(resolution["run_order"])
    if len(entries) != 86:
        raise ValueError("post-training finalizer requires all 86 authorized runs")
    return entries


def collect_rows(
    campaign: dict[str, Any], campaign_path: Path, source_commit: str
) -> list[dict[str, Any]]:
    rows = []
    for entry in all_entries(campaign, campaign_path):
        spec = load_campaign_run_spec(ROOT / entry["config"])
        run_dir = run_directory(spec)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete" or int(status.get("step", 0)) != 30_000:
            raise RuntimeError(f"run is not complete at 30k: {spec.experiment_id}")
        if status.get("config_hash") != spec.content_hash:
            raise ValueError(f"status/config hash mismatch: {spec.experiment_id}")
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("git_commit") != source_commit or manifest.get("git_dirty_at_launch"):
            raise ValueError(f"run source provenance mismatch: {spec.experiment_id}")
        metrics = final_validation(run_dir / "validation_metrics.jsonl")
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "config": entry["config"],
                "config_hash": spec.content_hash,
                "spec": spec.canonical_dict(),
                "step": 30_000,
                "metrics": metrics,
                "levels": spec.levels,
                "context_tokens": spec.plan.token_count,
                "state_bytes_per_sample_bf16": manifest["state_bytes_per_sample_bf16"]["total"],
                "peak_allocated_gib": status["peak_allocated_gib"],
                "runtime_seconds": status["runtime_seconds_this_attempt"],
                "checkpoint": str(run_dir / "milestones/milestone_step030000.pt"),
                "checkpoint_sha256": status["final_milestone_sha256"],
            }
        )
    return rows


def selection_payload(campaign_path: Path, campaign: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selections = select_postcampaign(pair_results(rows))
    roles: dict[str, Any] = {}
    for role, result in selections.items():
        roles[role] = {
            "configuration_key": result.key,
            "policy": result.policy,
            "budget_tokens": result.budget_tokens,
            "levels": result.levels,
            "overlap": result.overlap,
            "mean_horizon4_foreground_mse": result.mean_primary,
            "seed_values": list(result.seed_values),
            "context_tokens": result.context_tokens,
            "state_bytes_per_sample_bf16": result.state_bytes_per_sample_bf16,
            "runs": [
                {
                    "experiment_id": entry["experiment_id"],
                    "config": entry["config"],
                    "config_hash": entry["config_hash"],
                    "checkpoint": entry["checkpoint"],
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                }
                for entry in result.run_entries
            ],
        }
    return {
        "schema_version": 1,
        "status": "locked_from_development_validation",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": file_sha256(campaign_path),
        "dependency_resolution_sha256": file_sha256(RESOLUTION_PATH),
        "selection_split": "development validation only",
        "checkpoint_step": 30_000,
        "primary_metric": "horizon4_foreground_mse",
        "seed_aggregation": "arithmetic mean across seeds 0 and 1",
        "memory_pareto_tolerance_relative": 0.05,
        "roles": roles,
        "created_utc": utc_now(),
    }


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # Timestamps do not participate when proving a restarted selection is identical.
        old = {key: value for key, value in existing.items() if key != "created_utc"}
        new = {key: value for key, value in payload.items() if key != "created_utc"}
        if old != new:
            raise RuntimeError(f"immutable ledger would change on resume: {path}")
        return
    atomic_json(path, payload)


def confirmation_paths() -> tuple[Path, Path]:
    clips = CONFIRMATION_ROOT / "trb_article4_confirmation_500x200.npy"
    manifest = clips.with_name(clips.stem + ".manifest.json")
    return clips, manifest


def ensure_confirmation(campaign: dict[str, Any], source_commit: str) -> Path:
    config = campaign["post_training"]["confirmation"]
    clips, manifest = confirmation_paths()
    if manifest.exists():
        loaded = json.loads(manifest.read_text(encoding="utf-8"))
        if loaded["generator"]["seed"] != config["seed"]:
            raise ValueError("existing confirmation seed differs from campaign contract")
        for artifact in loaded["artifacts"].values():
            if file_sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError("existing confirmation artifact hash mismatch")
        return manifest

    CONFIRMATION_ROOT.mkdir(parents=True, exist_ok=True)
    # A killed generator may leave only a strict subset of its own known files. They are not a
    # valid dataset and are safely regenerated; no unrelated path or glob is touched.
    generated = (
        clips,
        clips.with_name(clips.stem + "_actions.npy"),
        clips.with_name(clips.stem + "_positions.npy"),
        clips.with_name(clips.stem + "_objects.npz"),
        manifest,
    )
    for path in generated:
        if path.exists():
            path.unlink()
    mnist = Path(config["mnist_path"])
    if file_sha256(mnist) != config["mnist_sha256"]:
        raise ValueError("registered MNIST glyph pool hash mismatch")
    command = [
        sys.executable,
        "-u",
        "scripts/generate_loopmom_confirmation.py",
        "--out", str(clips),
        "--seed", str(config["seed"]),
        "--generator-commit", source_commit,
        "--dirty-hash", "CLEAN",
        "--purpose", "confirmation_holdout",
        "--mnist", str(mnist),
        "--n", str(config["clips"]),
        "--frames", str(config["frames"]),
        "--world", str(config["world"]),
        "--digits", str(config["digits"]),
        "--speed", str(config["speed"]),
        "--p-cont", str(config["p_cont"]),
        "--p-hold", str(config["p_hold"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if not manifest.exists():
        raise RuntimeError("confirmation generator completed without a manifest")
    return manifest


def _clip_hashes(array: np.ndarray) -> set[str]:
    return {
        hashlib.sha256(np.ascontiguousarray(array[index]).tobytes()).hexdigest()
        for index in range(len(array))
    }


def audit_confirmation(campaign: dict[str, Any], manifest_path: Path) -> Path:
    """Verify schema, dynamics, uniqueness, and zero collision with development clips."""

    out = CONFIRMATION_ROOT / "confirmation_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = campaign["post_training"]["confirmation"]
    artifacts = manifest["artifacts"]
    clips = np.load(artifacts["clips"]["path"], mmap_mode="r")
    actions = np.load(artifacts["actions"]["path"])
    positions = np.load(artifacts["positions"]["path"])
    failures = []
    expected = (config["clips"], config["frames"], config["viewport"], config["viewport"])
    if clips.shape != expected or clips.dtype != np.uint8:
        failures.append(f"clip schema mismatch: {clips.shape}/{clips.dtype}")
    if actions.shape != expected[:2] or positions.shape != expected[:2]:
        failures.append("action/position shape mismatch")
    if not set(np.unique(actions)).issubset({-1, 0, 1}):
        failures.append("action vocabulary mismatch")
    expected_delta = (actions[:, 1:].astype(np.int64) * int(config["speed"])) % int(config["world"])
    actual_delta = (positions[:, 1:].astype(np.int64) - positions[:, :-1]) % int(config["world"])
    if not np.array_equal(expected_delta, actual_delta):
        failures.append("positions are inconsistent with actions")
    if positions.min() < 0 or positions.max() >= int(config["world"]):
        failures.append("viewport position outside the registered torus")
    holdout_hashes = _clip_hashes(clips)
    if len(holdout_hashes) != int(config["clips"]):
        failures.append("duplicate clips inside the sealed confirmation set")
    development_manifest = json.loads((ROOT / campaign["dataset_manifest"]).read_text(encoding="utf-8"))
    development = np.load(development_manifest["artifacts"]["clips"]["path"], mmap_mode="r")
    collisions = len(holdout_hashes & _clip_hashes(development))
    if collisions:
        failures.append(f"{collisions} exact clip collisions with development data")
    p_hold = float((actions == 0).mean())
    continuation = []
    for row in actions:
        moving = row[row != 0]
        if len(moving) > 1:
            continuation.extend((moving[1:] == moving[:-1]).tolist())
    p_continue = float(np.mean(continuation))
    if not 0.08 <= p_hold <= 0.12:
        failures.append(f"empirical p_hold outside tolerance: {p_hold:.4f}")
    if not 0.87 <= p_continue <= 0.93:
        failures.append(f"empirical p_cont outside tolerance: {p_continue:.4f}")
    report = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "manifest_sha256": file_sha256(manifest_path),
        "shape": list(clips.shape),
        "dtype": str(clips.dtype),
        "unique_clips": len(holdout_hashes),
        "development_collisions": collisions,
        "position_action_consistency": np.array_equal(expected_delta, actual_delta),
        "p_hold": p_hold,
        "p_continue_nonzero": p_continue,
        "failures": failures,
        "completed_utc": utc_now(),
    }
    atomic_json(out, report)
    if failures:
        raise RuntimeError("sealed confirmation audit failed: " + "; ".join(failures))
    return out


def tensor_stats(run_manifest: dict[str, Any]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
    values = run_manifest["latent_standardization"]
    base = tuple(
        torch.tensor(values[name], dtype=torch.float32).view(16, 1, 1)
        for name in ("coarse_mean", "coarse_std", "fine_mean", "fine_std")
    )
    top = None
    if values["top_mean"] is not None:
        top = tuple(
            torch.tensor(values[name], dtype=torch.float32).view(16, 1, 1)
            for name in ("top_mean", "top_std")
        )
    return base, top


def build_model(spec: Any, checkpoint: Path) -> tuple[torch.nn.Module, Any]:
    net = legacy.WindowDiT(
        n_ctx_t=len(spec.plan.top_ages),
        n_ctx_c=len(spec.plan.coarse_ages),
        n_ctx_f=spec.plan.reach,
        fine_token_budget=spec.plan.fine_tokens,
        mode="diff",
        d=spec.width,
        depth=spec.depth,
        heads=spec.heads,
        temporal_embedding_positions=spec.temporal_embedding_positions,
        temporal_context_span=spec.context_anchor,
        top_context_only=bool(spec.plan.top_ages),
    ).to(legacy.DEV)
    if spec.mixed_precision_rmsnorm:
        enable_mixed_precision_rmsnorm(net)
    saved = torch.load(checkpoint, map_location=legacy.DEV, weights_only=False)
    if saved["config_hash"] != spec.content_hash or int(saved["step"]) != 30_000:
        raise ValueError("final checkpoint identity mismatch")
    net.load_state_dict(saved["ema"])
    net.eval()
    return net, legacy.WindowDiffusion(net)


def visibility_labels(manifest: dict[str, Any], start: int) -> tuple[np.ndarray, np.ndarray]:
    artifacts = manifest["artifacts"]
    positions = np.load(artifacts["positions"]["path"])
    with np.load(artifacts["objects"]["path"]) as objects:
        world_x = np.asarray(objects["world_x"], dtype=np.int64)
        glyph_size = int(objects["glyph_size"])
    config = manifest["configuration"]
    world, viewport = int(config["world"]), int(config["viewport"])
    long_context = 96
    recent_reference = 28
    n, _ = positions.shape
    reentry = np.zeros((n, 4), dtype=bool)
    last_seen_gap = np.full((n, 4), -1, dtype=np.int16)

    def visible(clip: int, frame: int, obj: int) -> bool:
        sprite = {(int(world_x[clip, obj]) + offset) % world for offset in range(glyph_size)}
        view = {(int(positions[clip, frame]) + offset) % world for offset in range(viewport)}
        return bool(sprite & view)

    for clip in range(n):
        for horizon in range(4):
            target = start + horizon
            for obj in range(world_x.shape[1]):
                if not visible(clip, target, obj):
                    continue
                seen = [
                    frame
                    for frame in range(start - long_context, start)
                    if visible(clip, frame, obj)
                ]
                if seen:
                    gap = target - max(seen)
                    last_seen_gap[clip, horizon] = max(last_seen_gap[clip, horizon], gap)
                    if max(seen) < start - recent_reference:
                        reentry[clip, horizon] = True
    return reentry, last_seen_gap


@contextmanager
def attention_diagnostics(net: torch.nn.Module, spans: dict[str, tuple[int, int]]):
    """Collect target-query attention entropy/mass using hooks; production attention stays fused."""

    records: list[dict[str, Any]] = []
    hooks = []
    for layer_index, block in enumerate(net.blocks):
        module = block.attn

        def hook(attn_module: Any, inputs: tuple[torch.Tensor, ...], index: int = layer_index) -> None:
            x = inputs[0]
            qkv = attn_module.qkv(x).reshape(
                x.shape[0], x.shape[1], 3, attn_module.num_heads, attn_module.head_dim
            )
            q, k, _ = qkv.unbind(dim=2)
            q = attn_module.q_norm(q.transpose(1, 2))
            k = attn_module.k_norm(k.transpose(1, 2))
            target_start = spans["target_coarse"][0]
            layer = {"layer": index, "normalized_entropy": 0.0, "source_mass": {}}
            entropies = []
            masses: dict[str, list[float]] = {name: [] for name in spans}
            for head in range(attn_module.num_heads):
                scores = (q[:, head] @ k[:, head].transpose(-2, -1)) * (
                    attn_module.head_dim ** -0.5
                )
                weights = scores.float().softmax(dim=-1)
                target_weights = weights[:, target_start:]
                entropy = -(target_weights * target_weights.clamp_min(1e-12).log()).sum(dim=-1)
                entropies.append(float((entropy / math.log(weights.shape[-1])).mean()))
                for name, (left, right) in spans.items():
                    if right > left:
                        masses[name].append(float(target_weights[..., left:right].sum(dim=-1).mean()))
                del scores, weights, target_weights
            layer["normalized_entropy"] = float(np.mean(entropies))
            layer["source_mass"] = {
                name: float(np.mean(values)) if values else 0.0 for name, values in masses.items()
            }
            records.append(layer)

        hooks.append(module.register_forward_pre_hook(hook))
    try:
        yield records
    finally:
        for hook in hooks:
            hook.remove()


def token_spans(spec: Any) -> dict[str, tuple[int, int]]:
    lengths = {
        "context_top": len(spec.plan.top_ages) * 4,
        "context_coarse": len(spec.plan.coarse_ages) * 16,
        "context_fine": spec.plan.fine_tokens,
        "target_coarse": 4 * 16,
        "target_fine": 4 * 64,
    }
    spans = {}
    cursor = 0
    for name, length in lengths.items():
        spans[name] = (cursor, cursor + length)
        cursor += length
    return spans


@torch.no_grad()
def evaluate_selected_run(
    *, spec: Any, checkpoint: Path, confirmation_manifest: Path, role_names: list[str]
) -> dict[str, Any]:
    run_dir = run_directory(spec)
    output = ARTIFACT_ROOT / "evaluations" / f"{spec.experiment_id}__{spec.content_hash[:12]}"
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("checkpoint_sha256") == file_sha256(checkpoint):
            return summary
        raise RuntimeError("existing post-training evaluation has a different checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    base_stats, top_stats = tensor_stats(run_manifest)
    ae = (
        hier3.load_top(spec.ae_path, spec.top_ae_path)
        if spec.top_ae_path
        else hier_convae.load(spec.ae_path)
    )
    loaded = load_trb_confirmation(
        ae, confirmation_manifest, train_stats=base_stats, top_stats=top_stats
    )
    net, wrapper = build_model(spec, checkpoint)
    legacy.M._BF16 = True
    confirmation = loaded.manifest["configuration"]
    start = int(json.loads(DEFAULT_CAMPAIGN.read_text(encoding="utf-8"))["post_training"]["confirmation"]["evaluation_start"])
    reentry, last_seen_gap = visibility_labels(loaded.manifest, start)
    arrays: dict[str, list[np.ndarray]] = {
        "clip": [], "horizon": [], "pixel_mse": [], "foreground_mse": [],
        "reentry": [], "last_seen_gap": [],
    }
    first_panel = None
    batch_size = spec.validation_batch_size
    for offset in range(0, int(confirmation["n"]), batch_size):
        indices = np.arange(offset, min(offset + batch_size, int(confirmation["n"])), dtype=np.int64)
        # Pad only the final compute batch, then discard duplicates from scientific records.
        valid = len(indices)
        if valid < batch_size:
            indices = np.pad(indices, (0, batch_size - valid), mode="edge")
        batcher = PackedWindowBatcher(
            loaded.data, spec.plan, batch_size=batch_size, seed=0, device=legacy.DEV,
            context_anchor=spec.context_anchor,
        )
        batch = batcher.sample(
            split="validation", include_frames=True, clip_indices=indices,
            starts=np.full(batch_size, start, dtype=np.int64),
        )
        sampled = wrapper.sample(
            batch["ctx_c"], batch["ctx_f"], batch["actions"],
            n_steps=spec.validation_nfe, **forward_kwargs(batch),
        )[1]
        decoded = decode01(ae, sampled.reshape(-1, *sampled.shape[2:])).reshape(
            batch_size, 4, 1, 64, 64
        )
        target = batch["gt_frames_01"]
        squared = (decoded - target).square()
        foreground = target > 0.5
        pixel = squared.mean(dim=(2, 3, 4)).detach().cpu().numpy()[:valid]
        fg = (
            (squared * foreground).sum(dim=(2, 3, 4))
            / foreground.sum(dim=(2, 3, 4)).clamp_min(1)
        ).detach().cpu().numpy()[:valid]
        arrays["clip"].append(np.repeat(indices[:valid, None], 4, axis=1))
        arrays["horizon"].append(np.broadcast_to(np.arange(1, 5), (valid, 4)))
        arrays["pixel_mse"].append(pixel)
        arrays["foreground_mse"].append(fg)
        arrays["reentry"].append(reentry[indices[:valid]])
        arrays["last_seen_gap"].append(last_seen_gap[indices[:valid]])
        if first_panel is None:
            first_panel = {
                "sampled_clips": indices[:valid],
                "sampled_starts": np.full(valid, start),
                "last": batch["last_context_frame_01"][:valid].cpu(),
                "prediction": decoded[:valid].cpu(),
                "target": target[:valid].cpu(),
                "absolute_error": (decoded - target).abs()[:valid].cpu(),
            }
    joined = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    np.savez_compressed(output / "per_clip_horizon_metrics.npz", **joined)
    if first_panel is not None:
        save_panel(output, 30_000, first_panel)

    # One fixed, direct denoiser forward gives bounded attention mechanism evidence. It does not
    # alter prediction metrics or select a checkpoint.
    probe_batcher = PackedWindowBatcher(
        loaded.data, spec.plan, batch_size=1, seed=0, device=legacy.DEV,
        context_anchor=spec.context_anchor,
    )
    probe = probe_batcher.sample(
        split="validation", clip_indices=np.asarray([0]), starts=np.asarray([start])
    )
    spans = token_spans(spec)
    with attention_diagnostics(net, spans) as attention:
        net(
            probe["tgt_c"], probe["tgt_f"], probe["ctx_c"], probe["ctx_f"],
            torch.full((1,), 0.5, device=legacy.DEV), probe["actions"],
            **forward_kwargs(probe),
        )
    mask = joined["reentry"].astype(bool)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": spec.experiment_id,
        "config_hash": spec.content_hash,
        "checkpoint_sha256": file_sha256(checkpoint),
        "confirmation_manifest_sha256": file_sha256(confirmation_manifest),
        "roles": sorted(role_names),
        "evaluation_windows": int(confirmation["n"]),
        "evaluation_horizons": int(joined["pixel_mse"].size),
        "mean_pixel_mse": float(joined["pixel_mse"].mean()),
        "mean_foreground_mse": float(joined["foreground_mse"].mean()),
        "reentry_records": int(mask.sum()),
        "reentry_foreground_mse": (
            float(joined["foreground_mse"][mask].mean()) if mask.any() else None
        ),
        "non_reentry_foreground_mse": float(joined["foreground_mse"][~mask].mean()),
        "temporal_coverage": {
            "reach": spec.plan.reach,
            "top_time_positions": len(spec.plan.top_ages),
            "coarse_time_positions": len(spec.plan.coarse_ages),
            "fine_distinct_time_positions": len({age for age, _ in spec.plan.fine_age_cells}),
        },
        "level_token_utilization": {
            "top": len(spec.plan.top_ages) * 4,
            "coarse": len(spec.plan.coarse_ages) * 16,
            "fine": spec.plan.fine_tokens,
        },
        "attention": {
            "probe_clip": 0,
            "probe_start": start,
            "probe_diffusion_time": 0.5,
            "token_spans": spans,
            "layers": attention,
        },
        "completed_utc": utc_now(),
    }
    atomic_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    campaign_path = args.campaign.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if current_commit != args.source_commit or dirty:
        raise RuntimeError("post-training finalization requires the pinned clean source commit")
    rows = collect_rows(campaign, campaign_path, args.source_commit)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate_path = ARTIFACT_ROOT / "all_final_development_metrics.json"
    atomic_json(
        aggregate_path,
        {
            "schema_version": 1,
            "campaign_sha256": file_sha256(campaign_path),
            "run_count": len(rows),
            "rows": rows,
        },
    )
    ledger = selection_payload(campaign_path, campaign, rows)
    selection_path = ARTIFACT_ROOT / "selection_ledger.json"
    write_immutable_json(selection_path, ledger)
    selection_hash = file_sha256(selection_path)
    confirmation_manifest = ensure_confirmation(campaign, args.source_commit)
    confirmation_audit = audit_confirmation(campaign, confirmation_manifest)

    role_to_kind = {
        "best_flat_1792": {"mechanism", "confirmation"},
        "best_hierarchy_1792": {"mechanism", "confirmation"},
        "failure_hierarchy_1792": {"mechanism"},
        "memory_pareto_within_5pct": {"confirmation"},
    }
    selected_runs: dict[str, dict[str, Any]] = {}
    for role, kinds in role_to_kind.items():
        for run in ledger["roles"][role]["runs"]:
            record = selected_runs.setdefault(run["config_hash"], {**run, "roles": []})
            record["roles"].extend(f"{kind}:{role}" for kind in sorted(kinds))
    evaluations = []
    for record in selected_runs.values():
        spec = load_campaign_run_spec(ROOT / record["config"])
        evaluations.append(
            evaluate_selected_run(
                spec=spec,
                checkpoint=Path(record["checkpoint"]),
                confirmation_manifest=confirmation_manifest,
                role_names=sorted(set(record["roles"])),
            )
        )

    report = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": file_sha256(campaign_path),
        "selection_ledger": str(selection_path.relative_to(ROOT)).replace("\\", "/"),
        "selection_ledger_sha256": selection_hash,
        "confirmation_manifest": str(confirmation_manifest.relative_to(ROOT)).replace("\\", "/"),
        "confirmation_manifest_sha256": file_sha256(confirmation_manifest),
        "confirmation_audit": str(confirmation_audit.relative_to(ROOT)).replace("\\", "/"),
        "confirmation_audit_sha256": file_sha256(confirmation_audit),
        "unique_selected_seed_runs": len(evaluations),
        "evaluations": evaluations,
        "completed_utc": utc_now(),
    }
    report_path = ARTIFACT_ROOT / "final_report.json"
    atomic_json(report_path, report)
    tracker = WandbMirror(
        run_dir=ARTIFACT_ROOT,
        project=campaign["wandb"]["project"],
        group=campaign["wandb"]["group"],
        experiment_id="A4_POST_TRAINING_FINALIZATION",
        config_hash=selection_hash,
        config={
            "campaign_sha256": file_sha256(campaign_path),
            "selection_ledger_sha256": selection_hash,
            "confirmation_manifest_sha256": file_sha256(confirmation_manifest),
        },
        tags=["article4", "post-training", "mechanism", "sealed-confirmation"],
        job_type="article4-post-training-finalization",
        disabled=args.no_wandb,
    )
    tracker.log(
        {
            "post_training/unique_selected_seed_runs": len(evaluations),
            "post_training/mean_confirmation_foreground_mse": float(
                np.mean([row["mean_foreground_mse"] for row in evaluations])
            ),
        },
        step=30_000,
    )
    tracker.finish(
        summary={"status": "complete", "config_hash": selection_hash, **report},
        artifact_paths=(
            aggregate_path, selection_path, report_path, confirmation_manifest, confirmation_audit
        ),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
