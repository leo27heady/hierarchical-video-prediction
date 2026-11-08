import torch
import torch.nn as nn

from hvp_rev.faithful_nets import LAT
from hvp_rev.fixed_region_revformer import FixedRegionReconstructor
from hvp_rev.hier_convae import HierConvAE
from hvp_rev.packed_simvp import HierarchicalPackedTransition


class IdentityPredictor(nn.Module):
    def forward(self, x):
        return x


def test_packed_transition_shapes_and_gradient_path():
    torch.manual_seed(0)
    ae = HierConvAE(w=8)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    adapter = FixedRegionReconstructor()
    transition = HierarchicalPackedTransition(ae, adapter, IdentityPredictor())
    frames = torch.rand(1, 2, 1, 64, 64)
    state = transition.pack_frames(frames).detach().requires_grad_(True)
    assert state.shape == (1, 2, 24, LAT)
    prediction = transition.predict_pixels(state)
    assert prediction.shape == frames.shape
    future = transition(state)
    assert future.shape == state.shape
    future.square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert transition.values_per_frame == 384
