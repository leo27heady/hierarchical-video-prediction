import numpy as np

from scripts.gen_looping import build_world, pan_trajectory, render_clip


def test_metadata_path_is_pixel_and_rng_identical_to_legacy_path():
    digits = np.arange(16 * 28 * 28, dtype=np.float32).reshape(16, 28, 28) % 256
    legacy_rng = np.random.RandomState(123)
    metadata_rng = np.random.RandomState(123)

    legacy_world = build_world(192, 64, digits, legacy_rng, 3)
    world, metadata = build_world(192, 64, digits, metadata_rng, 3, return_metadata=True)
    assert np.array_equal(world, legacy_world)
    assert len(metadata) == 3

    legacy_pos, legacy_actions = pan_trajectory(
        40, 192, legacy_rng, p_hold=0.1, speed=4, p_cont=0.9
    )
    pos, actions = pan_trajectory(40, 192, metadata_rng, p_hold=0.1, speed=4, p_cont=0.9)
    assert np.array_equal(pos, legacy_pos)
    assert np.array_equal(actions, legacy_actions)
    assert np.array_equal(render_clip(world, pos, 64), render_clip(legacy_world, legacy_pos, 64))
