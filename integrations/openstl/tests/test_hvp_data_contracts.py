from __future__ import annotations

import gzip
import numpy as np
from pathlib import Path
import pytest
import subprocess
import sys
import torch

from openstl.datasets.dataloader_moving_mnist import (
    MovingMNIST, load_data as load_moving_mnist, load_fixed_set,
)
from openstl.datasets.dataloader_taxibj import TaxibjDataset, load_data as load_taxibj


def test_prepare_data_cli_bootstraps_openstl_import():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "prepare_data" /
                             "prepare_hvp_validation_and_cache.py"), "--help"],
        cwd=root, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--prepare" in result.stdout


def test_moving_mnist_fixed_set_is_memory_mapped(tmp_path):
    folder = tmp_path / "moving_mnist"
    folder.mkdir()
    path = folder / "validation.npy"
    source = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(4, 3, 8, 8))
    source[:] = 17
    source.flush()
    del source
    loaded = load_fixed_set(str(tmp_path), fixed_file="moving_mnist/validation.npy")
    assert loaded.shape == (4, 3, 8, 8, 1)
    assert isinstance(loaded, np.memmap)


def test_moving_mnist_fixed_dataset_keeps_test_shape_contract(tmp_path):
    folder = tmp_path / "moving_mnist"
    folder.mkdir()
    path = folder / "validation.npy"
    sequence = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(4, 3, 16, 16))
    sequence[:] = 128
    sequence.flush()
    del sequence
    dataset = MovingMNIST(
        str(tmp_path), is_train=False, data_name="mnist",
        n_frames_input=2, n_frames_output=2, image_size=16,
        fixed_file="moving_mnist/validation.npy",
    )
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == (2, 1, 16, 16)
    assert inputs.dtype == torch.float32
    torch.testing.assert_close(inputs.mean(), torch.tensor(128 / 255, dtype=torch.float32))


def test_moving_mnist_loader_accepts_zero_workers(tmp_path):
    folder = tmp_path / "moving_mnist"
    folder.mkdir()
    with gzip.open(folder / "train-images-idx3-ubyte.gz", "wb") as handle:
        handle.write(bytes(16) + bytes(28 * 28))
    for name in ("validation.npy", "mnist_test_seq.npy"):
        array = np.lib.format.open_memmap(
            folder / name, mode="w+", dtype=np.uint8, shape=(4, 3, 8, 8)
        )
        array[:] = 0
        array.flush()
        del array
    train, validation, test = load_moving_mnist(
        batch_size=2, val_batch_size=2, data_root=str(tmp_path), num_workers=0,
        pre_seq_length=2, aft_seq_length=2, in_shape=[2, 1, 8, 8],
        validation_file="moving_mnist/validation.npy", strict_validation=True,
    )
    assert train.persistent_workers is False
    assert validation.persistent_workers is False
    assert test.persistent_workers is False


def test_taxibj_dataset_normalizes_lazily_without_eager_copy():
    source_x = np.full((3, 4, 2, 8, 8), -1.0, dtype=np.float32)
    source_y = np.full_like(source_x, 1.0)
    dataset = TaxibjDataset(source_x, source_y)
    assert dataset.X is source_x and dataset.Y is source_y
    inputs, targets = dataset[1]
    assert inputs.min() == inputs.max() == 0
    assert targets.min() == targets.max() == 1


def _write_taxibj_cache(root, train=20, test=5):
    folder = root / "taxibj" / "memmap_v1"
    folder.mkdir(parents=True)
    for name, size in (("X_train", train), ("Y_train", train), ("X_test", test), ("Y_test", test)):
        array = np.lib.format.open_memmap(
            folder / f"{name}.npy", mode="w+", dtype=np.float32,
            shape=(size, 4, 2, 8, 8),
        )
        array[:] = 0
        array.flush()
        del array


def test_taxibj_loader_uses_separate_chronological_validation(tmp_path):
    _write_taxibj_cache(tmp_path)
    train, validation, test = load_taxibj(
        batch_size=2, val_batch_size=2, data_root=str(tmp_path), num_workers=0,
        pre_seq_length=4, aft_seq_length=4, validation_fraction=0.2,
        validation_gap=2, strict_memmap=True,
    )
    assert len(train.dataset) == 14
    assert len(validation.dataset) == 4
    assert len(test.dataset) == 5
    assert validation.dataset is not test.dataset
    batch_x, batch_y = next(iter(validation))
    assert batch_x.shape[1:] == batch_y.shape[1:] == (4, 2, 8, 8)


def test_taxibj_strict_cache_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="memmap cache"):
        load_taxibj(
            batch_size=2, val_batch_size=2, data_root=str(tmp_path), num_workers=0,
            pre_seq_length=4, aft_seq_length=4, strict_memmap=True,
        )
