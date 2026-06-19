from pathlib import Path

import pytest

from hvp.config import Config


def _present(cfg, h) -> bool:
    return (Path(cfg.data.cache_root) / h).exists()


def test_dataset_returns_normalized_64_clip():
    cfg = Config()
    if not _present(cfg, cfg.data.train_hash):
        pytest.skip("64x64 shape cache not present on this machine")
    from hvp.data import dataset_from_cfg

    ds = dataset_from_cfg(cfg, "train", max_clips=8)
    assert len(ds) > 0
    sample = ds[0]["video"]
    assert sample.shape == (3, 9, 64, 64)
    assert float(sample.min()) >= -1.0001 and float(sample.max()) <= 1.0001


def test_train_val_are_independent_caches():
    cfg = Config()
    if not (_present(cfg, cfg.data.train_hash) and _present(cfg, cfg.data.val_hash)):
        pytest.skip("64x64 caches not present")
    assert cfg.data.train_hash != cfg.data.val_hash  # true cross-split
    from hvp.data import dataset_from_cfg

    assert len(dataset_from_cfg(cfg, "val", max_clips=20)) > 0
