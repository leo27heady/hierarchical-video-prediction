from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hvp_hca.config import experiment_config_from_mapping, load_experiment_config
from hvp_hca.state import CoordinateFourierEncoding, TokenMetadata, pack_fixed_quota
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
