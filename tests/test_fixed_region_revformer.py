import copy

import torch

from hvp_rev.faithful_nets import LAT
from hvp_rev.fixed_region_revformer import (
    FixedRegionLayout,
    FixedRegionReconstructor,
    FixedRegionRevPredictor,
    count_context_tokens,
    fine_to_regions,
    openstl_mse,
    regions_to_fine,
)


def _latents(batch=2, frames=3):
    generator = torch.Generator().manual_seed(17)
    zc = torch.randn(batch, frames, LAT, 4, 4, generator=generator)
    zf = torch.randn(batch, frames, LAT, 8, 8, generator=generator)
    return zc, zf


def test_fixed_region_geometry_is_exact_and_budget_is_1152():
    layout = FixedRegionLayout()
    _, zf = _latents()
    cells = fine_to_regions(zf, layout)
    assert cells.shape == (2, 3, 8, 8, LAT)
    assert torch.equal(regions_to_fine(cells, layout), zf)
    assert layout.tokens_per_frame == 24
    assert count_context_tokens(48, layout) == 1152


def test_density_pooling_has_fixed_membership_and_normalized_weights():
    model = FixedRegionRevPredictor(dim=16, heads=4, depth=1, max_frames=4, max_steps=3)
    zc, zf = _latents(frames=3)
    tokens, weights = model.tokenizer(zc, zf)
    assert tokens.shape == (2, 72, 16)
    assert weights.shape == (2, 3, 8, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 8), atol=1e-6)
    assert torch.isfinite(tokens).all()


def test_compact_reconstructor_has_exact_24_by_16_persistent_values():
    reconstructor = FixedRegionReconstructor()
    zc, zf = _latents(frames=3)
    pred_f, regions, weights = reconstructor(zc, zf)
    assert pred_f.shape == zf.shape
    assert regions.shape == (2, 3, 8, LAT)
    assert weights.shape == (2, 3, 8, 8)
    assert reconstructor.values_per_frame == 24 * LAT
    (pred_f.square().mean() + reconstructor.pixel_residual(regions).mean()).backward()
    assert all(
        parameter.grad is not None
        for name, parameter in reconstructor.named_parameters()
        if not name.endswith("blend_logit")
    )


def test_predictor_shapes_and_backward():
    torch.manual_seed(5)
    model = FixedRegionRevPredictor(dim=16, heads=4, depth=1, max_frames=4, max_steps=3)
    zc, zf = _latents(frames=3)
    zc.requires_grad_(True)
    zf.requires_grad_(True)
    pred_c, pred_f, _ = model(zc, zf, steps=2, reversible=True)
    assert pred_c.shape == zc.shape
    assert pred_f.shape == zf.shape
    (pred_c.square().mean() + pred_f.square().mean()).backward()
    assert zc.grad is not None and torch.isfinite(zc.grad).all()
    assert zf.grad is not None and torch.isfinite(zf.grad).all()
    assert all(p.grad is not None for p in model.core.blocks.parameters())


def test_reversible_and_vanilla_gradients_match():
    torch.manual_seed(23)
    reversible = FixedRegionRevPredictor(dim=8, heads=2, depth=2, max_frames=3, max_steps=3)
    vanilla = copy.deepcopy(reversible)
    zc, zf = _latents(batch=1, frames=2)

    out_rev = reversible(zc, zf, steps=2, reversible=True)
    loss_rev = out_rev[0].square().mean() + out_rev[1].square().mean()
    loss_rev.backward()

    out_std = vanilla(zc, zf, steps=2, reversible=False)
    loss_std = out_std[0].square().mean() + out_std[1].square().mean()
    loss_std.backward()

    assert torch.allclose(out_rev[0], out_std[0], atol=2e-6, rtol=2e-6)
    assert torch.allclose(out_rev[1], out_std[1], atol=2e-6, rtol=2e-6)
    for (name_rev, parameter_rev), (name_std, parameter_std) in zip(
        reversible.named_parameters(), vanilla.named_parameters(), strict=True
    ):
        assert name_rev == name_std
        if parameter_rev.requires_grad:
            assert parameter_rev.grad is not None, name_rev
            assert parameter_std.grad is not None, name_std
            assert torch.allclose(parameter_rev.grad, parameter_std.grad, atol=2e-5, rtol=2e-4), name_rev


def test_openstl_metric_matches_definition():
    pred = torch.zeros(2, 10, 1, 64, 64)
    true = torch.ones_like(pred) * 0.5
    assert torch.allclose(openstl_mse(pred, true), torch.tensor(1024.0))
