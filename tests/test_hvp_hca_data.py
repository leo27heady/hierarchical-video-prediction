from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hvp_hca.allocation import frontier_plan
from hvp_hca.article4_run import load_article4_run_spec, validate_authorization
from hvp_hca.data import PackedWindowBatcher


ROOT = Path(__file__).resolve().parents[1]


class DummyTRB:
    def __init__(self) -> None:
        self.N, self.T = 5, 80
        self.train_idx = np.array([0, 1, 2], dtype=np.int64)
        self.val_idx = np.array([3, 4], dtype=np.int64)
        base = torch.arange(self.N * self.T * 16 * 8 * 8, dtype=torch.float32)
        self.zf = base.reshape(self.N, self.T, 16, 8, 8)
        self.zc = base[: self.N * self.T * 16 * 4 * 4].reshape(
            self.N, self.T, 16, 4, 4
        )
        self.acts = np.tile(np.array([-1, 0, 1, 0], dtype=np.int64), 20)[None].repeat(
            self.N, axis=0
        )
        self.clips_u8 = np.zeros((self.N, self.T, 64, 64), dtype=np.uint8)


@pytest.mark.parametrize("policy", ["flat_recent", "balanced_recent"])
def test_packed_batch_gathers_exact_selected_cells_without_target_leak(policy: str) -> None:
    data = DummyTRB()
    plan = frontier_plan(1792, policy)
    batcher = PackedWindowBatcher(data, plan, batch_size=2, seed=9, device="cpu")
    batch = batcher.sample(split="train", seed=72, include_frames=True)

    assert batch["ctx_f_packed"].shape == (2, plan.fine_tokens, 16)
    assert batch["ctx_c"].shape == (2, len(plan.coarse_ages), 16, 4, 4)
    assert batch["tgt_f"].shape == (2, 4, 16, 8, 8)
    assert batch["actions"].shape == (2, 4)
    assert set(batch["sampled_clips"].tolist()) <= set(data.train_idx.tolist())
    assert np.all(batch["context_fine_frames"] < batch["target_frames"][:, :1])
    if len(plan.coarse_ages):
        assert np.all(batch["context_coarse_frames"] < batch["target_frames"][:, :1])

    for row, (clip, start) in enumerate(
        zip(batch["sampled_clips"], batch["sampled_starts"])
    ):
        for column, (age, cell) in enumerate(plan.fine_age_cells):
            expected = data.zf[clip, start - age, :, cell // 8, cell % 8]
            torch.testing.assert_close(batch["ctx_f_packed"][row, column], expected)


def test_target_sampling_is_allocation_invariant_for_same_seed() -> None:
    data = DummyTRB()
    flat = PackedWindowBatcher(
        data, frontier_plan(1792, "flat_recent"), batch_size=3, seed=0, device="cpu",
        context_anchor=76,
    ).sample(split="validation", seed=88)
    hierarchy = PackedWindowBatcher(
        data, frontier_plan(1792, "balanced_recent"), batch_size=3, seed=0, device="cpu",
        context_anchor=76,
    ).sample(split="validation", seed=88)
    np.testing.assert_array_equal(flat["sampled_clips"], hierarchy["sampled_clips"])
    np.testing.assert_array_equal(flat["sampled_starts"], hierarchy["sampled_starts"])
    np.testing.assert_array_equal(flat["target_frames"], hierarchy["target_frames"])
    assert np.all(np.diff(flat["target_frames"], axis=1) == 1)
    assert np.all(np.diff(hierarchy["target_frames"], axis=1) == 1)
    assert flat["target_frames"].shape == hierarchy["target_frames"].shape == (3, 4)
    assert set(hierarchy["sampled_clips"].tolist()) <= set(data.val_idx.tolist())


def test_packed_state_accounting_matches_actual_latent_and_index_payload() -> None:
    data = DummyTRB()
    plan = frontier_plan(1792, "balanced_recent")
    batcher = PackedWindowBatcher(data, plan, batch_size=2, seed=0, device="cpu")
    accounting = batcher.state_bytes_per_sample(latent_dtype=torch.float32)
    expected_values = len(plan.coarse_ages) * 16 * 16 + plan.fine_tokens * 16
    expected_indices = len(plan.coarse_ages) + 2 * plan.fine_tokens
    assert accounting["latent_payload"] == expected_values * 4
    assert accounting["frame_and_cell_metadata"] == expected_indices * 8
    assert accounting["total"] == accounting["latent_payload"] + accounting["frame_and_cell_metadata"]


def test_article4_calibration_specs_are_hash_authorized_and_exact() -> None:
    specs = sorted((ROOT / "experiments/v2/runspecs/article4").glob("*.json"))
    authorization = (
        ROOT / "experiments/v2/authorizations/article4_schedule_calibration_2026-09-01.json"
    )
    assert len(specs) == 2
    loaded = [load_article4_run_spec(path) for path in specs]
    for spec in loaded:
        validate_authorization(spec, authorization)
        assert spec.train_updates == 30000
        assert spec.validation_every == 2000
        assert spec.plan.token_count == 1792
    assert {spec.policy for spec in loaded} == {"flat_recent", "balanced_recent"}
    assert len({spec.content_hash for spec in loaded}) == 2
