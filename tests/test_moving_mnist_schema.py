import numpy as np
import pytest

from hvp_rev.data.moving_mnist import generate_moving_mnist, save_moving_mnist


def _source():
    images = np.zeros((3, 28, 28), np.uint8)
    images[0, 5:23, 10:18] = 255
    images[1, 8:20, 4:24] = 200
    images[2, 4:24, 4:24] = np.eye(20, dtype=np.uint8) * 255
    labels = np.asarray([1, 7, 9], np.uint8)
    return images, labels


def test_generator_saves_persistent_ids_classes_and_trajectories(tmp_path):
    images, labels = _source()
    bundle = generate_moving_mnist(
        images, labels, n_clips=2, frames=5, objects=2, seed=7,
        image_size=64, source_split="synthetic-fixture",
    )
    bundle.validate()
    assert bundle.object_id.shape == (2, 2)
    assert len(np.unique(bundle.object_id)) == 4
    assert bundle.trajectory_yx.shape == (2, 2, 5, 2)
    assert bundle.bbox_yxyx.shape == (2, 2, 5, 4)
    out = tmp_path / "fixture.npy"
    manifest = save_moving_mnist(
        bundle, out, purpose="development", generator_commit="fixture",
    )
    assert manifest["source_glyph_pool"]["split"] == "synthetic-fixture"
    assert out.exists() and out.with_suffix(".metadata.npz").exists()
    actions = out.with_name(out.stem + "_actions.npy")
    assert actions.exists()
    assert not np.load(actions).any()
    assert manifest["data_schema"] == "moving-mnist-persistent-v1"
    assert manifest["artifacts"]["clips"]["path"].endswith("fixture.npy")
    with pytest.raises(FileExistsError):
        save_moving_mnist(bundle, out, purpose="development", generator_commit="fixture")
