import json
import random

import numpy as np
import pytest
import torch

from hvp_rev.provenance import (
    ArtifactStore,
    RunSpec,
    capture_rng_state,
    restore_rng_state,
)


def _spec(seed=3, arm="fixture"):
    return RunSpec(
        study="cpu-fixture",
        arm=arm,
        seed=seed,
        dataset_hash="d" * 64,
        ae_hash="a" * 64,
        arm_config={"context_tokens": 512, "target_tokens": 320},
        train_examples=32,
        optimizer_updates=4,
        batch_size=8,
        precision="fp32",
        nfe=1,
        warmup_frames=8,
        rollout_frames=4,
        evaluation_clips=2,
        source_commit="deadbeef",
        environment={"fixture": True},
    )


def test_run_identity_changes_with_scientific_configuration():
    assert _spec().identity == _spec().identity
    assert _spec(seed=4).identity != _spec(seed=3).identity
    assert _spec(arm="other").identity != _spec().identity


def test_artifact_store_refuses_completed_overwrite(tmp_path):
    spec = _spec()
    store = ArtifactStore(tmp_path, spec)
    store.save_npz("metrics/records.npz", x=np.arange(3))
    store.finalize()
    manifest = json.loads((store.path / "manifest.json").read_text())
    assert manifest["run_spec_hash"] == spec.identity
    with pytest.raises(FileExistsError):
        ArtifactStore(tmp_path, spec)


def test_artifact_store_reopens_partial_with_json_equivalent_sequences(tmp_path):
    base = _spec()
    spec = RunSpec(**{
        **base.resolved(),
        "arm_config": {"context_tokens": 512, "loss_weights": (0.0, 1.0, 1.0)},
    })
    first = ArtifactStore(tmp_path, spec)
    second = ArtifactStore(tmp_path, spec)
    assert second.path == first.path


def test_artifact_store_still_rejects_real_partial_spec_collision(tmp_path):
    spec = _spec()
    store = ArtifactStore(tmp_path, spec)
    path = store.path / "run_spec.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["batch_size"] += 1
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(FileExistsError, match="collision"):
        ArtifactStore(tmp_path, spec)


def test_rng_capture_restores_python_numpy_and_torch():
    random.seed(11); np.random.seed(12); torch.manual_seed(13)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    _ = (random.random(), np.random.rand(), torch.rand(()))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_incompatible_specs_fail_closed():
    with pytest.raises(ValueError, match="seed"):
        _spec(seed=1).assert_compatible(_spec(seed=2))


def test_optimizer_ema_and_sampler_changes_are_resume_incompatible():
    base = _spec()
    for field, value in (
        ("optimizer", {"lr": 3e-4}),
        ("ema", {"decay": 0.999}),
        ("sampler", {"kind": "ddim", "eta": 0.0}),
    ):
        changed = RunSpec(**{**base.resolved(), field: value})
        with pytest.raises(ValueError, match=field):
            base.assert_compatible(changed)
