import numpy as np
import pytest

from scripts.analyze_e4_factorial import _aligned


def _run(*, with_labels: bool = True) -> dict[str, np.ndarray]:
    run = {
        "clip": np.array([0, 0, 1, 1]),
        "h": np.array([1, 2, 1, 2]),
        "hard": np.array([0.0, 1.0, 0.0, 1.0]),
        "fg": np.array([0.1, 0.2, 0.3, 0.4]),
    }
    if with_labels:
        run.update({"dt": np.array([1, 2, 1, 2]), "seen": np.ones(4)})
    return run


def test_e4_alignment_accepts_legacy_cells_without_optional_labels():
    _aligned(_run(), _run(with_labels=False), _run(), _run(with_labels=False))


def test_e4_alignment_checks_optional_labels_where_available():
    mismatched = _run()
    mismatched["dt"] = np.array([1, 99, 1, 2])
    with pytest.raises(ValueError, match="not paired on dt"):
        _aligned(_run(), _run(with_labels=False), mismatched)


def test_e4_alignment_rejects_required_geometry_mismatch():
    mismatched = _run(with_labels=False)
    mismatched["clip"] = np.array([0, 0, 0, 1])
    with pytest.raises(ValueError, match="not paired on clip"):
        _aligned(_run(), mismatched)
