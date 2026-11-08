import copy
import random

import numpy as np
import torch

from hvp_rev import arm_registry, rhvd
from hvp_rev.provenance import RunSpec, load_training_checkpoint, save_training_checkpoint
from hvp_rev.rhvd_train_v2 import _data_artifacts


class _FixtureData:
    def __init__(self):
        generator = torch.Generator().manual_seed(91)
        self.T = 12
        self.train_idx = np.arange(4)
        self.val_idx = np.arange(4, 6)
        self.zc = torch.randn(6, self.T, 16, 4, 4, generator=generator)
        self.zf = torch.randn(6, self.T, 16, 8, 8, generator=generator)
        self.acts = np.arange(6 * self.T, dtype=np.int64).reshape(6, self.T) % 3 - 1


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        elif isinstance(left[key], dict):
            _assert_state_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def _one_update(shared, monkeypatch):
    monkeypatch.setattr(rhvd, "DEV", "cpu")
    torch.manual_seed(7)
    np.random.seed(8)
    data = _FixtureData()
    net = rhvd.WindowDiT(
        n_ctx_c=2, n_ctx_f=2, mode="reg", d=48, depth=1, heads=3,
    )
    wrap = rhvd.WindowRegression(net)
    ema = rhvd.EMA(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    batch = rhvd.window_batch(data, 2, 2, 2, return_indices=True)
    if shared:
        loss = rhvd.training_update(wrap, net, ema, optimizer, batch, 1)
    else:
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 * min(1.0, 1 / 2000)
        loss = wrap.loss(
            batch["ctx_c"], batch["ctx_f"], batch["tgt_c"], batch["tgt_f"],
            batch["actions"], ctx_t=batch.get("ctx_t"), tgt_t=batch.get("tgt_t"),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        ema.update(net)
    return {
        "clips": batch["sampled_clips"],
        "starts": batch["sampled_starts"],
        "batch": {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "loss": loss.detach(),
        "model": copy.deepcopy(net.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "ema": copy.deepcopy(ema.shadow.state_dict()),
    }


def test_h3b_shared_update_is_numerically_identical_to_legacy_inline(monkeypatch):
    legacy = _one_update(False, monkeypatch)
    v2 = _one_update(True, monkeypatch)
    assert np.array_equal(legacy["clips"], v2["clips"])
    assert np.array_equal(legacy["starts"], v2["starts"])
    for key in legacy["batch"]:
        assert torch.equal(legacy["batch"][key], v2["batch"][key]), key
    assert torch.equal(legacy["loss"], v2["loss"])
    _assert_state_equal(legacy["model"], v2["model"])
    _assert_state_equal(legacy["optimizer"], v2["optimizer"])
    _assert_state_equal(legacy["ema"], v2["ema"])


def test_arm_registry_is_exact_legacy_adapter():
    adapted = arm_registry.from_legacy_mapping(rhvd.ARMS)
    for name in rhvd.ARMS:
        n_c, n_f, head, stride_c, n_hist, n_top, stride_top = rhvd.arm_cfg(name)
        spec = adapted[name]
        assert (
            spec.coarse_context_slots, spec.fine_recent_slots, spec.head,
            spec.coarse_context_stride, spec.fine_history_slots,
            spec.top_context_slots, spec.top_context_stride,
        ) == (n_c, n_f, head, stride_c, n_hist, n_top, stride_top)


def test_v2_dataset_identity_includes_optional_geometric_labels(tmp_path):
    train = tmp_path / "fixture_train.npy"
    test = tmp_path / "fixture_test.npy"
    for path in (train, test):
        np.save(path, np.zeros((2, 3), dtype=np.uint8))
        np.save(path.with_name(path.stem + "_actions.npy"), np.zeros((2, 3), dtype=np.int8))
        np.save(path.with_name(path.stem + "_positions.npy"), np.zeros((2, 3), dtype=np.int32))
    artifacts = _data_artifacts(str(train), str(test))
    assert set(artifacts) == {
        "train_clips", "train_actions", "train_positions",
        "test_clips", "test_actions", "test_positions",
    }
    assert all(len(value) == 64 for value in artifacts.values())


def _resume_spec():
    return RunSpec(
        study="resume-fixture", arm="toy", seed=3, dataset_hash="d" * 64,
        ae_hash="a" * 64, arm_config={"toy": True}, train_examples=6,
        optimizer_updates=2, batch_size=3, precision="fp32", nfe=1,
        warmup_frames=1, rollout_frames=1, evaluation_clips=1,
        source_commit="fixture", environment={"fixture": True},
        optimizer={"name": "AdamW"}, ema={"decay": 0.9},
        sampler={"numpy": "global"},
    )


def _toy_update(model, optimizer):
    x = torch.rand(3, 4) + float(np.random.rand())
    target = torch.rand(3, 2) + random.random()
    loss = torch.nn.functional.mse_loss(model(x), target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_rng_complete_checkpoint_resume_matches_uninterrupted(tmp_path):
    random.seed(1); np.random.seed(2); torch.manual_seed(3)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _toy_update(model, optimizer)
    checkpoint = tmp_path / "resume.pt"
    save_training_checkpoint(
        checkpoint, spec=_resume_spec(), step=1, model=model.state_dict(),
        optimizer=optimizer.state_dict(), ema=model.state_dict(), sampler_state={"next": 2},
    )
    _toy_update(model, optimizer)
    uninterrupted = copy.deepcopy(model.state_dict())

    resumed_model = torch.nn.Linear(4, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    state = load_training_checkpoint(checkpoint, _resume_spec(), restore_rng=True)
    resumed_model.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    _toy_update(resumed_model, resumed_optimizer)
    _assert_state_equal(uninterrupted, resumed_model.state_dict())
