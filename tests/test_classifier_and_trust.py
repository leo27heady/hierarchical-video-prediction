import numpy as np
import torch

from hvp_rev.classifier_validation import ValidationDomain, validate_classifier
from hvp_rev.trust_diagnostics import corrupt_latent, refresh_mask, trust_penalty


class _Classifier(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(len(x), 10)
        logits[:, 0] = 10
        return logits


def test_classifier_domains_keep_shifted_accuracy_visible():
    x = np.zeros((4, 1, 28, 28), np.float32)
    report = validate_classifier(_Classifier(), {
        "clean": ValidationDomain("clean", x, np.zeros(4, np.int64), 0.9),
        "shift": ValidationDomain("shift", x, np.ones(4, np.int64), 0.9),
        "overlap": ValidationDomain("overlap", x, np.ones(4, np.int64), None),
    })
    assert report["domains"]["clean"]["passed"]
    assert not report["domains"]["shift"]["passed"]
    assert report["domains"]["overlap"]["passed"]  # diagnostic, not hidden
    assert not report["passed"]


def test_trust_interventions_are_seeded_and_aligned():
    z = torch.zeros(2, 3)
    assert torch.equal(corrupt_latent(z, 0.2, seed=4), corrupt_latent(z, 0.2, seed=4))
    assert refresh_mask(8, 3).tolist() == [False, False, True, False, False, True, False, False]
    assert trust_penalty([0.1, 0.2], [0.15, 0.5]).tolist() == [0.04999999999999999, 0.3]
