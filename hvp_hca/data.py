"""Manifest-locked TRB loading and CPU-packed Article 4 window batches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hvp_hca.allocation import AllocationPlan
from hvp_rev.faithful_nets import LAT
from hvp_rev.step1_mech import ActionHierData


TARGET_HORIZON = 4


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("purpose") != "development_only":
        raise ValueError("Article 4 development loader requires purpose=development_only")
    if manifest.get("test_policy", {}).get("included"):
        raise ValueError("development manifest must not include a test split")
    return manifest, hashlib.sha256(raw).hexdigest()


def _channel_stats(tensor: Tensor, indices: np.ndarray, chunk_size: int = 64) -> tuple[Tensor, Tensor]:
    total = 0
    sums = torch.zeros(tensor.shape[2], dtype=torch.float64)
    squares = torch.zeros_like(sums)
    for start in range(0, len(indices), chunk_size):
        chunk_indices = torch.as_tensor(indices[start:start + chunk_size], dtype=torch.long)
        chunk = tensor.index_select(0, chunk_indices).to(torch.float64)
        sums += chunk.sum(dim=(0, 1, 3, 4))
        squares += chunk.square().sum(dim=(0, 1, 3, 4))
        total += chunk.shape[0] * chunk.shape[1] * chunk.shape[3] * chunk.shape[4]
    mean = sums / total
    variance = (squares - sums.square() / total) / max(total - 1, 1)
    std = variance.clamp_min(1e-8).sqrt().clamp_min(1e-4)
    return mean.float().view(LAT, 1, 1), std.float().view(LAT, 1, 1)


@dataclass(frozen=True)
class LoadedTRB:
    data: ActionHierData
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def load_trb_development(
    ae: torch.nn.Module,
    manifest_path: str | Path,
    *,
    verify_source_hashes: bool = True,
) -> LoadedTRB:
    """Encode TRB once, apply the manifest split, and standardize from train clips only."""

    manifest, manifest_hash = load_manifest(manifest_path)
    artifacts = manifest["artifacts"]
    clips_path = Path(artifacts["clips"]["path"])
    actions_path = Path(artifacts["actions"]["path"])
    if verify_source_hashes:
        for name, path in (("clips", clips_path), ("actions", actions_path)):
            actual = sha256_file(path)
            expected = artifacts[name]["sha256"]
            if actual != expected:
                raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")

    configuration = manifest["configuration"]
    data = ActionHierData(
        ae,
        str(clips_path),
        str(actions_path),
        n_ctx=6,
        k=TARGET_HORIZON,
        W=int(configuration["world_width"]),
        speed=int(configuration["speed"]),
        standardize=False,
    )
    split = manifest["split"]
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(split["validation_indices"], dtype=np.int64)
    if len(train_indices) != split["train_count"] or len(validation_indices) != split["validation_count"]:
        raise ValueError("manifest split counts do not match index arrays")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("manifest train and validation clips overlap")
    combined = np.concatenate((train_indices, validation_indices))
    if set(combined.tolist()) != set(range(data.N)):
        raise ValueError("manifest split must partition every development clip exactly once")
    data.train_idx = train_indices
    data.val_idx = validation_indices

    mu_c, sd_c = _channel_stats(data.zc, train_indices)
    mu_f, sd_f = _channel_stats(data.zf, train_indices)
    data.zc.sub_(mu_c[None, None]).div_(sd_c[None, None])
    data.zf.sub_(mu_f[None, None]).div_(sd_f[None, None])
    data.stats = (mu_c, sd_c, mu_f, sd_f)
    ae._lat_stats = data.stats
    return LoadedTRB(
        data=data,
        manifest=manifest,
        manifest_path=Path(manifest_path).resolve(),
        manifest_sha256=manifest_hash,
    )


class PackedWindowBatcher:
    """Sample invariant four-frame targets and transfer only the selected fine state."""

    def __init__(
        self,
        data: ActionHierData,
        plan: AllocationPlan,
        *,
        batch_size: int,
        seed: int,
        device: str | torch.device,
        context_anchor: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if plan.reach <= 0:
            raise ValueError("allocation plan must have positive temporal reach")
        self.context_anchor = plan.reach if context_anchor is None else int(context_anchor)
        if self.context_anchor < plan.reach or data.T < self.context_anchor + TARGET_HORIZON:
            raise ValueError(
                f"TRB clip length {data.T} cannot support reach {plan.reach}, "
                f"shared context anchor {self.context_anchor}, "
                f"and target {TARGET_HORIZON}"
            )
        self.data = data
        self.plan = plan
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        # WindowDiT consumes coarse slots oldest to newest.
        self.coarse_ages = torch.tensor(sorted(plan.coarse_ages, reverse=True), dtype=torch.long)
        self.fine_ages = torch.tensor([age for age, _ in plan.fine_age_cells], dtype=torch.long)
        self.fine_cells = torch.tensor([cell for _, cell in plan.fine_age_cells], dtype=torch.long)
        self.fine_rows = self.fine_cells // 8
        self.fine_columns = self.fine_cells % 8
        self.coarse_frame_ids = self.context_anchor - self.coarse_ages
        self.fine_frame_ids = plan.reach - self.fine_ages

    def sample(
        self,
        *,
        split: str,
        seed: int | None = None,
        include_frames: bool = False,
    ) -> dict[str, Tensor | np.ndarray]:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        pool = self.data.train_idx if split == "train" else self.data.val_idx
        clips_np = rng.choice(pool, size=self.batch_size, replace=True).astype(np.int64)
        # numpy integers uses an exclusive upper bound; +1 admits the last valid start.
        starts_np = rng.integers(
            self.context_anchor,
            self.data.T - TARGET_HORIZON + 1,
            size=self.batch_size,
            dtype=np.int64,
        )
        clips = torch.from_numpy(clips_np)
        starts = torch.from_numpy(starts_np)
        target_offsets = torch.arange(TARGET_HORIZON, dtype=torch.long)
        target_frames = starts[:, None] + target_offsets[None, :]

        if self.coarse_ages.numel():
            coarse_frames = starts[:, None] - self.coarse_ages[None, :]
            ctx_c = self.data.zc[clips[:, None], coarse_frames]
            coarse_ids = self.coarse_frame_ids[None, :].expand(self.batch_size, -1)
        else:
            ctx_c = torch.empty(self.batch_size, 0, LAT, 4, 4)
            coarse_ids = torch.empty(self.batch_size, 0, dtype=torch.long)

        fine_frames = starts[:, None] - self.fine_ages[None, :]
        # Multiple advanced indices gather exactly one physical cell while preserving C as the
        # trailing basic-index dimension. No dense BxKxCx8x8 candidate tensor is materialized.
        packed_fine = self.data.zf[
            clips[:, None],
            fine_frames,
            :,
            self.fine_rows[None, :],
            self.fine_columns[None, :],
        ]
        if packed_fine.shape != (self.batch_size, self.plan.fine_tokens, LAT):
            raise AssertionError(f"unexpected packed fine shape: {tuple(packed_fine.shape)}")
        tgt_c = self.data.zc[clips[:, None], target_frames]
        tgt_f = self.data.zf[clips[:, None], target_frames]
        actions_np = self.data.acts[clips_np[:, None], target_frames.numpy()]
        actions = torch.from_numpy(np.asarray(actions_np, dtype=np.int64))

        output: dict[str, Tensor | np.ndarray] = {
            "ctx_c": ctx_c.to(self.device),
            "ctx_f": torch.empty(self.batch_size, 0, LAT, 8, 8, device=self.device),
            "ctx_c_frame_ids": coarse_ids.to(self.device),
            "ctx_f_packed": packed_fine.to(self.device),
            "ctx_f_ids": self.fine_cells[None, :].expand(self.batch_size, -1).to(self.device),
            "ctx_f_frame_ids": self.fine_frame_ids[None, :].expand(self.batch_size, -1).to(self.device),
            "tgt_c": tgt_c.to(self.device),
            "tgt_f": tgt_f.to(self.device),
            "actions": actions.to(self.device),
            "sampled_clips": clips_np,
            "sampled_starts": starts_np,
            "context_coarse_frames": (
                starts_np[:, None] - self.coarse_ages.numpy()[None, :]
            ),
            "context_fine_frames": (
                starts_np[:, None] - self.fine_ages.numpy()[None, :]
            ),
            "target_frames": target_frames.numpy(),
        }
        if include_frames:
            frames = self.data.clips_u8[clips_np[:, None], target_frames.numpy()]
            output["gt_frames_01"] = torch.from_numpy(
                np.asarray(frames, dtype=np.float32) / 255.0
            ).unsqueeze(2).to(self.device)
        return output

    def state_bytes_per_sample(self, *, latent_dtype: torch.dtype = torch.bfloat16) -> dict[str, int]:
        latent_bytes = torch.empty((), dtype=latent_dtype).element_size()
        index_bytes = torch.empty((), dtype=torch.int64).element_size()
        coarse_values = len(self.plan.coarse_ages) * 16 * LAT
        fine_values = self.plan.fine_tokens * LAT
        payload = (coarse_values + fine_values) * latent_bytes
        metadata = (len(self.plan.coarse_ages) + 2 * self.plan.fine_tokens) * index_bytes
        return {
            "latent_payload": payload,
            "frame_and_cell_metadata": metadata,
            "total": payload + metadata,
        }
