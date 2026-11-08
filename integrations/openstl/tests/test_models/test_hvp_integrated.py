from __future__ import annotations

import copy

import torch
import pytest

from openstl.models.hvp_integrated_model import (
    CoordinateFourierEncoding,
    HVPIntegrated_Model,
    PackedContext,
    PhysicalTokenMetadata,
    ReversibleTransformerCore,
    SpaceTimeRouter,
    _grid_metadata,
)


def _tiny_model(**overrides):
    values = dict(
        in_shape=(2, 1, 16, 16),
        aft_seq_length=2,
        d_model=16,
        stem_width=8,
        core_depth=2,
        decoder_depth=1,
        num_heads=2,
        detail_quota=4,
        min_detail_per_frame=1,
        routing_mode="hybrid",
        hierarchy_mode="hierarchical",
        reversible_core=True,
        position_frequencies=2,
    )
    values.update(overrides)
    return HVPIntegrated_Model(**values)


def _nonzero_head(model):
    torch.nn.init.normal_(model.decoder.output.weight, std=0.03)
    return model


def test_coordinate_encoding_follows_metadata_not_packed_slot():
    torch.manual_seed(1)
    metadata = _grid_metadata(
        2, 2, 2, 3, level=1, future=False, time_normalizer=2,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    order = torch.tensor(
        [[5, 0, 7, 1, 3, 11, 2, 9, 4, 8, 10, 6],
         [8, 4, 3, 9, 0, 10, 1, 7, 6, 2, 11, 5]]
    )
    encoding = CoordinateFourierEncoding(16, num_frequencies=3)
    original = encoding(metadata)
    moved = encoding(metadata.gather(order))
    gather = order.unsqueeze(-1).expand_as(original)
    torch.testing.assert_close(moved, original.gather(1, gather))


def test_density_router_selects_real_fixed_quota_and_temporal_coverage():
    torch.manual_seed(2)
    batch, frames, width, fine_h, fine_w = 1, 2, 8, 4, 4
    fine = torch.zeros(batch, frames, width, fine_h, fine_w)
    fine[:, 1, :, 3, 2] = 10.0
    coarse = torch.zeros(batch, frames, width, 2, 2)
    metadata = _grid_metadata(
        batch, frames, fine_h, fine_w, level=1, future=False,
        time_normalizer=2, device=torch.device("cpu"), dtype=torch.float32,
    )
    router = SpaceTimeRouter(
        width, learned_weight=0.0, min_detail_per_frame=1
    )
    selected, selected_meta, indices, _, diagnostics, resolved = router(
        fine, coarse, metadata, quota=4, mode="density"
    )
    changed_index = fine_h * fine_w + 3 * fine_w + 2
    assert resolved == "density"
    assert selected.shape == (batch, 4, width)
    assert selected_meta.token_count == 4
    assert indices.unique().numel() == 4
    assert changed_index in indices[0].tolist()
    assert diagnostics["route_temporal_coverage"] == 1


def test_mixed_mode_uses_dynamic_routing_during_evaluation():
    model = _tiny_model(routing_mode="mixed")
    model.eval()
    with torch.no_grad():
        _, auxiliary = model(torch.rand(1, 2, 1, 16, 16), return_aux=True)
    assert auxiliary["resolved_routing_mode"] == "hybrid"


def test_memory_contract_has_expected_three_point_two_token_ratio():
    moving_mnist = HVPIntegrated_Model(
        in_shape=(10, 1, 64, 64), aft_seq_length=10,
        d_model=32, stem_width=16, core_depth=1, decoder_depth=1,
        num_heads=4, detail_quota=160,
    )
    taxibj = HVPIntegrated_Model(
        in_shape=(4, 2, 32, 32), aft_seq_length=4,
        d_model=32, stem_width=16, core_depth=1, decoder_depth=1,
        num_heads=4, detail_quota=16,
    )
    assert moving_mnist.memory_contract()["packed_tokens"] == 800
    assert taxibj.memory_contract()["packed_tokens"] == 80
    assert moving_mnist.memory_contract()["flat_to_packed_token_ratio"] == 3.2
    assert taxibj.memory_contract()["flat_to_packed_token_ratio"] == 3.2


def test_reversible_core_matches_vanilla_outputs_and_gradients():
    torch.manual_seed(3)
    reversible = ReversibleTransformerCore(16, depth=3, heads=2)
    vanilla = copy.deepcopy(reversible)
    x_reversible = torch.randn(2, 7, 16, requires_grad=True)
    x_vanilla = x_reversible.detach().clone().requires_grad_(True)
    y_reversible = reversible(x_reversible, reversible=True)
    y_vanilla = vanilla(x_vanilla, reversible=False)
    loss_reversible = y_reversible.square().mean()
    loss_vanilla = y_vanilla.square().mean()
    loss_reversible.backward()
    loss_vanilla.backward()
    torch.testing.assert_close(y_reversible, y_vanilla, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(x_reversible.grad, x_vanilla.grad, atol=2e-5, rtol=2e-4)
    for (name_rev, parameter_rev), (name_std, parameter_std) in zip(
        reversible.named_parameters(), vanilla.named_parameters(), strict=True
    ):
        assert name_rev == name_std
        assert parameter_rev.grad is not None, name_rev
        assert parameter_std.grad is not None, name_std
        torch.testing.assert_close(
            parameter_rev.grad, parameter_std.grad, atol=3e-5, rtol=3e-4
        )


def test_reversible_blocks_reconstruct_their_input():
    torch.manual_seed(4)
    core = ReversibleTransformerCore(16, depth=3, heads=2).double()
    original = torch.randn(2, 5, 16, dtype=torch.float64)
    transformed = original
    for block in core.blocks:
        transformed = block(transformed)
    recovered = core.inverse_blocks(transformed)
    torch.testing.assert_close(recovered, original, atol=1e-10, rtol=1e-10)


def test_reversible_core_preserves_cpu_bfloat16_autocast_parity():
    torch.manual_seed(41)
    reversible = ReversibleTransformerCore(16, depth=2, heads=2)
    vanilla = copy.deepcopy(reversible)
    first = torch.randn(1, 5, 16, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output_reversible = reversible(first, reversible=True)
        loss_reversible = output_reversible.float().square().mean()
    loss_reversible.backward()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output_vanilla = vanilla(second, reversible=False)
        loss_vanilla = output_vanilla.float().square().mean()
    loss_vanilla.backward()
    torch.testing.assert_close(output_reversible, output_vanilla, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(first.grad, second.grad, atol=5e-3, rtol=5e-2)
    for parameter_rev, parameter_std in zip(
        reversible.parameters(), vanilla.parameters(), strict=True
    ):
        assert parameter_rev.grad is not None and parameter_std.grad is not None
        torch.testing.assert_close(parameter_rev.grad, parameter_std.grad, atol=1e-2, rtol=8e-2)


def test_whole_tiny_model_matches_forward_and_gradient_across_core_modes():
    torch.manual_seed(5)
    reversible = _nonzero_head(_tiny_model(routing_mode="static"))
    vanilla = copy.deepcopy(reversible)
    x_reversible = torch.rand(1, 2, 1, 16, 16, requires_grad=True)
    x_vanilla = x_reversible.detach().clone().requires_grad_(True)
    y_reversible = reversible(x_reversible, reversible_core=True)
    y_vanilla = vanilla(x_vanilla, reversible_core=False)
    y_reversible.square().mean().backward()
    y_vanilla.square().mean().backward()
    torch.testing.assert_close(y_reversible, y_vanilla, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(x_reversible.grad, x_vanilla.grad, atol=3e-5, rtol=4e-4)
    for (name, parameter), (other_name, other) in zip(
        reversible.named_parameters(), vanilla.named_parameters(), strict=True
    ):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, atol=3e-5, rtol=4e-4)
    for module in (reversible.encoder, reversible.router, reversible.core):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 1e-7


def test_hybrid_route_connects_learned_scores_to_the_prediction_graph():
    torch.manual_seed(51)
    model = _nonzero_head(_tiny_model(routing_mode="hybrid", reversible_core=False))
    prediction = model(torch.rand(1, 2, 1, 16, 16))
    prediction.square().mean().backward()
    for name, parameter in model.router.named_parameters():
        assert parameter.grad is not None, name
    assert sum(float(p.grad.abs().sum()) for p in model.router.learned_score.parameters()) > 1e-7


def test_decoder_is_invariant_to_packed_slot_permutation():
    torch.manual_seed(6)
    model = _nonzero_head(_tiny_model(routing_mode="density", reversible_core=False)).eval()
    frames = torch.rand(1, 2, 1, 16, 16)
    with torch.no_grad():
        packed = model.encode_and_pack(frames, "density")
        reference = model.decode_packed(packed, frames, reversible=False)
        order = torch.randperm(packed.tokens.shape[1]).unsqueeze(0)
        token_order = order.unsqueeze(-1).expand_as(packed.tokens)
        moved = PackedContext(
            tokens=packed.tokens.gather(1, token_order),
            metadata=packed.metadata.gather(order),
            route_indices=packed.route_indices,
            route_scores=packed.route_scores,
            diagnostics=packed.diagnostics,
            coarse_grid=packed.coarse_grid,
        )
        result = model.decode_packed(moved, frames, reversible=False)
        moved.tokens = torch.randn_like(moved.tokens)
        corrupted = model.decode_packed(moved, frames, reversible=False)
    torch.testing.assert_close(result, reference, atol=3e-6, rtol=3e-6)
    assert float((corrupted - reference).abs().max()) > 1e-5


def test_persistence_initialization_is_separate_from_dependency_tests():
    model = _tiny_model().eval()
    frames = torch.rand(1, 2, 1, 16, 16)
    torch.testing.assert_close(model(frames), frames[:, -1:].expand(-1, 2, -1, -1, -1))


def test_all_membership_modes_share_features_at_full_quota():
    model = _tiny_model(detail_quota=32).eval()
    frames = torch.rand(1, 2, 1, 16, 16)
    with torch.no_grad():
        reference = model.encode_and_pack(frames, "static")
        for mode in ("density", "hybrid"):
            other = model.encode_and_pack(frames, mode)
            torch.testing.assert_close(reference.tokens, other.tokens)


def test_infeasible_quota_fails_and_ties_have_physical_order():
    model = _tiny_model(detail_quota=1, min_detail_per_frame=1)
    with pytest.raises(ValueError, match="infeasible"):
        model(torch.rand(1, 2, 1, 16, 16))
    router = SpaceTimeRouter(8, min_detail_per_frame=1)
    selected = router._dynamic_indices(torch.zeros(1, 8), frames=2, cells=4, quota=4)
    assert selected.tolist() == [[0, 1, 2, 4]]


def test_checkpointed_core_matches_eager_parameter_gradients():
    torch.manual_seed(61)
    eager = _nonzero_head(_tiny_model(reversible_core=False))
    checked = copy.deepcopy(eager)
    checked.core.activation_checkpointing = True
    inputs = torch.rand(1, 2, 1, 16, 16)
    target = torch.rand_like(inputs)
    first, second = eager(inputs), checked(inputs)
    (first - target).square().mean().backward()
    (second - target).square().mean().backward()
    torch.testing.assert_close(first, second)
    for (name, p), (_, q) in zip(eager.named_parameters(), checked.named_parameters(), strict=True):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=4e-4)


def test_both_openstl_dataset_shapes_and_checkpoint_round_trip(tmp_path):
    specifications = [
        ((10, 1, 64, 64), 10, 40),
        ((4, 2, 32, 32), 4, 8),
    ]
    for index, (shape, output_frames, quota) in enumerate(specifications):
        model = HVPIntegrated_Model(
            in_shape=shape, aft_seq_length=output_frames,
            d_model=16, stem_width=8, core_depth=1, decoder_depth=1,
            num_heads=2, detail_quota=quota, min_detail_per_frame=1,
            routing_mode="static", reversible_core=False, position_frequencies=2,
        )
        inputs = torch.rand(1, *shape)
        prediction = model(inputs)
        assert prediction.shape == (1, output_frames, shape[1], shape[2], shape[3])
        loss = prediction.square().mean()
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.decoder.parameters())
        path = tmp_path / f"model_{index}.pt"
        torch.save(model.state_dict(), path)
        restored = copy.deepcopy(model)
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        restored.eval()
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), model(inputs))


def test_invalid_inference_route_fails_closed():
    model = _tiny_model()
    try:
        model.set_routing_mode("unknown")
    except ValueError as error:
        assert "unknown routing mode" in str(error)
    else:
        raise AssertionError("invalid routing mode was accepted")


def test_openstl_registry_training_validation_and_optimizer_smoke(tmp_path):
    from openstl.methods import method_maps
    from openstl.utils import create_parser

    parsed = create_parser().parse_args(["--method", "HVPIntegrated"])
    assert parsed.method == "HVPIntegrated"
    assert method_maps["hvpintegrated"].__name__ == "HVPIntegrated"
    method = method_maps["hvpintegrated"](
        dataname="mmnist",
        metrics=["mse"],
        in_shape=(2, 1, 16, 16),
        pre_seq_length=2,
        aft_seq_length=2,
        d_model=16,
        stem_width=8,
        core_depth=1,
        decoder_depth=1,
        num_heads=2,
        detail_quota=4,
        min_detail_per_frame=1,
        routing_mode="static",
        reversible_core=False,
        position_frequencies=2,
        steps_per_epoch=2,
        epoch=2,
        test_mean=0,
        test_std=1,
        save_dir=str(tmp_path),
        opt="adamw",
        weight_decay=1e-4,
        filter_bias_and_bn=False,
        lr=1e-3,
        sched="onecycle",
        gpus=[0],
        final_div_factor=1e4,
    )
    batch = (torch.rand(1, 2, 1, 16, 16), torch.rand(1, 2, 1, 16, 16))
    assert torch.isfinite(method.training_step(batch, 0))
    assert torch.isfinite(method.validation_step(batch, 0))
    configured = method.configure_optimizers()
    assert configured["optimizer"].__class__.__name__ == "AdamW"
