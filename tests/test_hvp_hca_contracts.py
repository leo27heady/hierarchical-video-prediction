from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hvp_hca.config import experiment_config_from_mapping, load_experiment_config
from hvp_hca.performance import MixedPrecisionRMSNorm, enable_mixed_precision_rmsnorm
from hvp_hca.state import CoordinateFourierEncoding, TokenMetadata, pack_fixed_quota
from hvp_rev.rhvd import WindowDiT
from hvp_rev.reversible_rollout import ReversibleSecondOrderRollout


ROOT = Path(__file__).resolve().parents[1]


def _metadata(batch: int, tokens: int, level: int) -> TokenMetadata:
    xy = torch.linspace(0.0, 1.0, tokens).repeat(batch, 2, 1).transpose(1, 2)
    return TokenMetadata(
        xy=xy,
        time=torch.arange(tokens, dtype=torch.float32).repeat(batch, 1),
        scale=torch.full((batch, tokens, 2), 1.0 / max(tokens, 1)),
        level=torch.full((batch, tokens), level, dtype=torch.int64),
        valid=torch.ones((batch, tokens), dtype=torch.bool),
    )


def test_all_v2_config_templates_validate_and_fail_closed() -> None:
    paths = sorted((ROOT / "experiments" / "v2" / "configs").rglob("*.json"))
    assert paths
    configs = [load_experiment_config(path) for path in paths]
    assert len({config.experiment_id for config in configs}) == len(configs)
    assert len({config.content_hash for config in configs}) == len(configs)
    assert all(not config.launch_authorized for config in configs)
    assert all(config.checkpoint_selection_split == "validation" for config in configs)


def test_config_rejects_test_based_checkpoint_selection() -> None:
    path = next((ROOT / "experiments" / "v2" / "configs").rglob("*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checkpoint_selection_split"] = "test"
    with pytest.raises(ValueError, match="checkpoint selection"):
        experiment_config_from_mapping(data)


def test_article4_confirmation_rejects_unregistered_third_seed() -> None:
    path = ROOT / "experiments" / "v2" / "configs" / "article4" / "A4_FRT_B1152_L1_FLAT_EXCL_S00.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["phase"] = "confirmation"
    data["seed"] = 2
    with pytest.raises(ValueError, match="seeds are registered as 0 and 1"):
        experiment_config_from_mapping(data)


def test_config_rejects_static_positions_for_moving_routes() -> None:
    path = ROOT / "experiments" / "v2" / "configs" / "article5" / "A5_B4_HIER_DYNAMIC_S00.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["positional_encoding"] = "static_slot"
    with pytest.raises(ValueError, match="coordinate-aware"):
        experiment_config_from_mapping(data)


def test_fixed_quota_changes_real_state_payload() -> None:
    batch, width = 2, 8
    coarse = torch.randn(batch, 4, width)
    fine = torch.randn(batch, 12, width)
    scores = torch.arange(12, dtype=torch.float32).repeat(batch, 1)
    coarse_meta = _metadata(batch, 4, level=0)
    fine_meta = _metadata(batch, 12, level=1)

    state_k8 = pack_fixed_quota(
        coarse_tokens=coarse,
        coarse_metadata=coarse_meta,
        fine_tokens=fine,
        fine_metadata=fine_meta,
        fine_scores=scores,
        detail_quota=8,
    )
    state_k4 = pack_fixed_quota(
        coarse_tokens=coarse,
        coarse_metadata=coarse_meta,
        fine_tokens=fine,
        fine_metadata=fine_meta,
        fine_scores=scores,
        detail_quota=4,
    )
    assert state_k8.detail_quota == 8
    assert state_k4.detail_quota == 4
    assert state_k4.payload_bytes() < state_k8.payload_bytes()


def test_coordinate_encoding_is_equivariant_to_packed_slot_permutation() -> None:
    torch.manual_seed(0)
    metadata = _metadata(batch=2, tokens=7, level=1)
    order = torch.tensor([[6, 0, 3, 2, 1, 5, 4], [2, 1, 6, 5, 4, 0, 3]])
    permuted = metadata.gather(order)
    encoding = CoordinateFourierEncoding(d_model=16, max_levels=3)
    original = encoding(metadata)
    moved = encoding(permuted)
    gather_index = order.unsqueeze(-1).expand(-1, -1, original.shape[-1])
    torch.testing.assert_close(moved, original.gather(1, gather_index))


def test_global_fine_selection_separates_gather_and_context_frame_ids() -> None:
    """Selected fine frames remain gatherable inside a longer coarse context."""

    batch = 2
    model = WindowDiT(
        n_ctx_c=8,
        n_ctx_f=4,
        fine_token_budget=6,
        d=32,
        depth=1,
        heads=4,
    )
    coarse = torch.randn(batch, 8, 16, 4, 4)
    fine = torch.randn(batch, 4, 16, 8, 8)
    local_frames = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 3]])
    cells = torch.tensor([[0, 7, 9, 17, 32, 63], [1, 8, 10, 18, 33, 62]])

    tokens, _, _, frame_ids = model._ctx_tokens(
        coarse,
        fine,
        ctx_f_ids=cells,
        ctx_f_frame_ids=local_frames,
    )
    assert tokens[-1].shape == (batch, 6, 32)
    # Fine context occupies absolute frames 4..7 because the coarse reach is eight.
    torch.testing.assert_close(frame_ids[-1], local_frames + 4)

    flattened = fine.permute(0, 1, 3, 4, 2).reshape(batch, 4 * 64, 16)
    packed_indices = local_frames * 64 + cells
    packed = flattened.gather(1, packed_indices.unsqueeze(-1).expand(-1, -1, 16))
    packed_tokens, packed_levels, packed_roles, packed_frames = model._ctx_tokens(
        coarse,
        fine,
        ctx_f_ids=cells,
        ctx_f_frame_ids=local_frames,
        ctx_f_packed=packed,
    )
    torch.testing.assert_close(packed_tokens[-1], tokens[-1])
    torch.testing.assert_close(packed_levels[-1], torch.ones_like(cells))
    torch.testing.assert_close(packed_roles[-1], torch.zeros_like(cells))
    torch.testing.assert_close(packed_frames[-1], local_frames + 4)


def test_explicit_coarse_frame_ids_support_exclusive_gapped_history() -> None:
    model = WindowDiT(
        n_ctx_c=3,
        n_ctx_f=8,
        fine_token_budget=4,
        d=32,
        depth=1,
        heads=4,
    )
    batch = 2
    coarse = torch.randn(batch, 3, 16, 4, 4)
    fine_placeholder = torch.empty(batch, 0, 16, 8, 8)
    coarse_frames = torch.tensor([[0, 2, 4], [0, 2, 4]])
    fine_frames = torch.tensor([[5, 6, 7, 7], [5, 6, 7, 7]])
    fine_cells = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    packed = torch.randn(batch, 4, 16)
    _, _, _, frame_ids = model._ctx_tokens(
        coarse,
        fine_placeholder,
        ctx_c_frame_ids=coarse_frames,
        ctx_f_ids=fine_cells,
        ctx_f_frame_ids=fine_frames,
        ctx_f_packed=packed,
    )
    for token_frames, expected in zip(frame_ids[:3], (0, 2, 4)):
        torch.testing.assert_close(token_frames, torch.full_like(token_frames, expected))


def test_mixed_precision_rmsnorm_preserves_fp32_outputs_and_gradients() -> None:
    torch.manual_seed(4)
    reference = torch.nn.RMSNorm(8)
    optimized = MixedPrecisionRMSNorm.from_module(reference)
    first = torch.randn(3, 5, 8, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    reference(first).sum().backward()
    optimized(second).sum().backward()
    torch.testing.assert_close(optimized(second), reference(first.detach()))
    torch.testing.assert_close(second.grad, first.grad)


def test_shared_temporal_embedding_capacity_keeps_reach_parameter_neutral() -> None:
    flat = WindowDiT(
        n_ctx_c=0, n_ctx_f=28, fine_token_budget=1792, d=32, depth=1, heads=4,
        temporal_embedding_positions=100,
        temporal_context_span=96,
    )
    hierarchy = WindowDiT(
        n_ctx_c=56, n_ctx_f=56, fine_token_budget=896, d=32, depth=1, heads=4,
        temporal_embedding_positions=100,
        temporal_context_span=96,
    )
    assert flat.tok.frame_embedding.num_embeddings == 100
    assert hierarchy.tok.frame_embedding.num_embeddings == 100
    assert sum(parameter.numel() for parameter in flat.parameters()) == sum(
        parameter.numel() for parameter in hierarchy.parameters()
    )


def test_mixed_precision_rmsnorm_replaces_only_attention_norms() -> None:
    model = WindowDiT(n_ctx_c=2, n_ctx_f=2, d=32, depth=2, heads=4)
    assert enable_mixed_precision_rmsnorm(model) == 4
    assert all(
        isinstance(block.attn.q_norm, MixedPrecisionRMSNorm)
        and isinstance(block.attn.k_norm, MixedPrecisionRMSNorm)
        for block in model.blocks
    )


def test_legacy_second_order_wrapper_is_not_ordinary_multistep_rollout() -> None:
    class Double(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return 2.0 * value

    transition = Double()
    initial = torch.tensor([1.0])
    ordinary = transition(transition(initial))
    _, second_order = ReversibleSecondOrderRollout(transition)(
        initial, steps=2, reversible=False
    )
    torch.testing.assert_close(ordinary, torch.tensor([4.0]))
    torch.testing.assert_close(second_order, torch.tensor([5.0]))
    assert not torch.equal(ordinary, second_order)


def test_top_context_only_keeps_the_two_level_prediction_target() -> None:
    model = WindowDiT(
        n_ctx_t=3,
        n_ctx_c=2,
        n_ctx_f=2,
        fine_token_budget=4,
        d=32,
        depth=1,
        heads=4,
        temporal_embedding_positions=20,
        temporal_context_span=16,
        top_context_only=True,
    )
    batch = 2
    result = model(
        torch.randn(batch, 4, 16, 4, 4),
        torch.randn(batch, 4, 16, 8, 8),
        torch.randn(batch, 2, 16, 4, 4),
        torch.empty(batch, 0, 16, 8, 8),
        torch.zeros(batch),
        torch.zeros(batch, 4, dtype=torch.long),
        ctx_t=torch.randn(batch, 3, 16, 2, 2),
        ctx_f_ids=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        ctx_f_frame_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        ctx_f_packed=torch.randn(batch, 4, 16),
    )
    assert isinstance(result, tuple) and len(result) == 2
    assert result[0].shape == (batch, 4, 16, 4, 4)
    assert result[1].shape == (batch, 4, 16, 8, 8)


def test_packed_residual_subtracts_the_projected_coarse_reference() -> None:
    model = WindowDiT(
        n_ctx_c=1,
        n_ctx_f=1,
        fine_token_budget=2,
        d=32,
        depth=1,
        heads=4,
    )
    packed = torch.randn(2, 2, 16)
    reference = torch.randn_like(packed)
    frames = torch.zeros(2, 2, dtype=torch.long)
    cells = torch.tensor([[0, 1], [2, 3]])
    absolute, *_ = model.tok.tokenize_packed_sequence(packed, frames, cells)
    residual, *_ = model.tok.tokenize_packed_sequence(
        packed, frames, cells, reference=reference
    )
    torch.testing.assert_close(absolute - residual, model.tok.patchify(reference))
