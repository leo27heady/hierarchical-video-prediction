import copy

import torch
import torch.nn as nn

from hvp_rev.reversible_rollout import ReversibleSecondOrderRollout


class TinyTransition(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 16), nn.Tanh(), nn.Linear(16, dim))

    def forward(self, x):
        return self.net(x)


def test_first_step_is_exactly_the_transition():
    torch.manual_seed(1)
    transition = TinyTransition()
    rollout = ReversibleSecondOrderRollout(transition)
    current = torch.randn(2, 3, 8)
    _, prediction = rollout(current, steps=1, reversible=True)
    assert torch.allclose(prediction, transition(current), atol=1e-7, rtol=1e-7)


def test_rollout_inverse_recovers_both_initial_states():
    torch.manual_seed(2)
    rollout = ReversibleSecondOrderRollout(TinyTransition()).double()
    previous = torch.randn(2, 3, 8, dtype=torch.float64)
    current = torch.randn_like(previous)
    final = rollout(current, steps=7, previous=previous, reversible=False)
    recovered = rollout.inverse(*final, steps=7)
    assert torch.allclose(recovered[0], previous, atol=1e-11, rtol=1e-11)
    assert torch.allclose(recovered[1], current, atol=1e-11, rtol=1e-11)


def test_reversible_and_vanilla_parameter_gradients_match():
    torch.manual_seed(3)
    transition_rev = TinyTransition()
    transition_std = copy.deepcopy(transition_rev)
    rev = ReversibleSecondOrderRollout(transition_rev)
    std = ReversibleSecondOrderRollout(transition_std)
    current_rev = torch.randn(2, 3, 8, requires_grad=True)
    current_std = current_rev.detach().clone().requires_grad_(True)
    outputs_rev = rev(current_rev, steps=5, reversible=True)
    outputs_std = std(current_std, steps=5, reversible=False)
    loss_rev = outputs_rev[0].square().mean() + outputs_rev[1].square().mean()
    loss_std = outputs_std[0].square().mean() + outputs_std[1].square().mean()
    loss_rev.backward()
    loss_std.backward()
    assert torch.allclose(outputs_rev[0], outputs_std[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs_rev[1], outputs_std[1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(current_rev.grad, current_std.grad, atol=2e-5, rtol=2e-4)
    for rev_parameter, std_parameter in zip(
        transition_rev.parameters(), transition_std.parameters(), strict=True
    ):
        assert torch.allclose(rev_parameter.grad, std_parameter.grad, atol=2e-5, rtol=2e-4)
