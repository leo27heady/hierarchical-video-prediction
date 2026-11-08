import numpy as np
import pytest
import json
from pathlib import Path

from scripts.evaluate_e8a_pose_mosaic import (
    _validate_manifest_artifacts,
    _validate_manifest_header,
    build_pose_mosaic,
    geometric_labels,
    query_pose_mosaic,
)
from scripts.analyze_e8a_pose_mosaic import (
    HIGHER_BETTER,
    _load_canonicalized,
    population_mask,
    registered_verdict,
    simultaneous_family_allowing_degenerate,
)
from scripts.validate_e8a_design import _is_manifest_mirror


def _static_clip(world, positions):
    columns = np.arange(64)
    return np.stack([world[:, (int(pos) + columns) % world.shape[1]] for pos in positions])


def test_pose_mosaic_reconstructs_observed_static_world_exactly():
    world = np.arange(64 * 192, dtype=np.uint16).reshape(64, 192).astype(np.uint8)
    positions = np.arange(48, dtype=np.int32) * 4 % 192
    frames = _static_clip(world, positions)
    mosaic, observed = build_pose_mosaic(
        frames, positions, world=192, warm=48, span=48
    )
    predicted, coverage = query_pose_mosaic(
        mosaic, observed, positions, row_prior=np.zeros(64, dtype=np.float32)
    )
    assert observed.all()
    assert coverage.all()
    np.testing.assert_array_equal(predicted, frames.astype(np.float32) / 255.0)


def test_pose_mosaic_unknown_columns_use_only_frozen_row_prior():
    world = np.zeros((64, 192), dtype=np.uint8)
    world[:, :64] = 255
    positions = np.zeros(48, dtype=np.int32)
    frames = _static_clip(world, positions)
    mosaic, observed = build_pose_mosaic(
        frames, positions, world=192, warm=48, span=6
    )
    row_prior = np.linspace(0.1, 0.4, 64, dtype=np.float32)
    predicted, coverage = query_pose_mosaic(
        mosaic, observed, np.asarray([64], dtype=np.int32), row_prior=row_prior
    )
    assert not coverage.any()
    np.testing.assert_allclose(
        predicted[0], np.broadcast_to(row_prior[:, None], predicted[0].shape)
    )


def test_pose_mosaic_rejects_nonstatic_repeated_world_pixels():
    frames = np.zeros((48, 64, 64), dtype=np.uint8)
    positions = np.zeros(48, dtype=np.int32)
    frames[47, 0, 0] = 1
    with pytest.raises(ValueError, match="static-world consistency failure"):
        build_pose_mosaic(frames, positions, world=192, warm=48, span=6)


def test_geometric_labels_mark_beyond_window_reentry():
    world = np.zeros((64, 192), dtype=np.uint8)
    world[0, 0] = 255
    positions = np.concatenate([
        np.zeros(48, dtype=np.int32),
        np.full(55, 64, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
    ])
    frames = _static_clip(world, positions)
    hard, delay, seen = geometric_labels(
        frames, positions, world=192, warm=48, threshold=0.5
    )
    assert hard[-1] == 1.0
    assert delay[-1] > 48
    assert seen[-1] == 1.0


def test_e8a_registered_verdict_categories():
    ids = {"n1", "n2", "n3", "n4"}
    clean = [
        {"effect": "raw14", "mean": 0.1, "lo": 0.01},
        {"effect": "raw38", "mean": 0.0, "lo": 0.0},
        *[{"effect": effect, "mean": 0.1, "lo": 0.01} for effect in sorted(ids)],
    ]
    assert registered_verdict(clean, ids) == "RAW_ACCESS_CEILING_CI_CLEAN"
    uncertain = [dict(row) for row in clean]
    uncertain[-1]["lo"] = -0.01
    assert registered_verdict(uncertain, ids) == "RAW_ACCESS_DIRECTIONAL_REFERENCE"
    failed = [dict(row) for row in clean]
    failed[-1]["mean"] = -0.01
    assert registered_verdict(failed, ids) == "RAW_ACCESS_CEILING_NOT_ESTABLISHED"


def test_e8a_answerable_population_is_inside_k48_reach():
    reference = {
        "dt": np.asarray([6.0, 7.0, 14.0, 38.0, 48.0, 49.0]),
        "hard": np.ones(6),
        "seen": np.ones(6),
    }
    selected = population_mask(reference, {
        "dt_lower_exclusive": 6,
        "dt_upper_inclusive": 48,
        "hard_lower_exclusive": 0.1,
        "seen_lower_inclusive": 0.5,
    })
    np.testing.assert_array_equal(
        selected, np.asarray([False, True, True, True, True, False])
    )


def test_e8a_simultaneous_family_accepts_exact_saturation_control():
    clusters = np.repeat(np.arange(5), 4)
    contrasts = {
        "raw-reach-38-to-48": np.zeros(20),
        "neural-to-raw48": np.repeat(np.linspace(0.05, 0.09, 5), 4),
    }
    family = simultaneous_family_allowing_degenerate(
        contrasts, clusters, n_boot=500, seed=0, confidence=0.95
    )
    saturated = family["contrasts"]["raw-reach-38-to-48"]
    assert saturated["mean"] == saturated["lo"] == saturated["hi"] == 0.0
    assert saturated["degenerate"] is True


def test_e8a_manifest_header_check_does_not_open_protected_artifacts(tmp_path):
    manifest_path = tmp_path / "reference.manifest.json"
    manifest_path.write_text(json.dumps({
        "purpose": "retrieval_reference_holdout",
        "configuration": {
            "n": 500, "frames": 200, "world": 192, "digits": 3,
            "speed": 4, "p_cont": 0.9, "p_hold": 0.1,
        },
        "artifacts": {
            "clips": {"path": (tmp_path / "missing.npy").as_posix(), "sha256": "0" * 64}
        },
    }), encoding="utf-8")
    config = {
        "dataset": {
            "purpose": "retrieval_reference_holdout",
            "clips": 500, "frames": 200, "world": 192, "digits": 3,
            "speed": 4, "p_cont": 0.9, "p_hold": 0.1,
        }
    }
    manifest = _validate_manifest_header(manifest_path, config)
    assert manifest["purpose"] == "retrieval_reference_holdout"
    with pytest.raises(ValueError, match="dataset artifact hash mismatch"):
        _validate_manifest_artifacts(manifest)


def test_e8a_coverage_diagnostic_uses_higher_is_better_convention():
    assert "current_crop_observed_fraction" in HIGHER_BETTER


def test_e8a_neural_path_has_a_distinct_purpose_lock():
    source = Path("hvp_rev/rhvd.py").read_text(encoding="utf-8")
    for marker in (
        "--retrieval_manifest",
        "retrieval_reference_holdout",
        'locked_prefix, locked_study = "e8a", "reference-e8a"',
        'locked_job_type = "retrieval-reference-eval"',
    ):
        assert marker in source


def test_e8a_byte_identical_manifest_copy_is_a_provenance_mirror(tmp_path):
    protected = tmp_path / "protected.manifest.json"
    tracked = tmp_path / "tracked.manifest.json"
    payload = '{"dataset_id":"same","artifacts":{}}\n'
    protected.write_text(payload, encoding="utf-8")
    tracked.write_text(payload, encoding="utf-8")
    assert protected.resolve() != tracked.resolve()
    assert _is_manifest_mirror(protected, tracked)
    tracked.write_text(payload.replace("same", "different"), encoding="utf-8")
    assert not _is_manifest_mirror(protected, tracked)


def test_e8a_pairing_is_independent_of_serialization_order(tmp_path):
    clip_major = tmp_path / "clip-major.npz"
    horizon_major = tmp_path / "horizon-major.npz"
    clips = np.asarray([250, 250, 251, 251])
    horizons = np.asarray([1, 2, 1, 2])
    metric = clips * 1000 + horizons
    np.savez(clip_major, clip=clips, h=horizons, fg=metric)
    permutation = np.asarray([0, 2, 1, 3])
    np.savez(
        horizon_major,
        clip=clips[permutation],
        h=horizons[permutation],
        fg=metric[permutation],
    )
    left = _load_canonicalized(clip_major)
    right = _load_canonicalized(horizon_major)
    for key in ("clip", "h", "fg"):
        np.testing.assert_array_equal(left[key], right[key])


def test_e8a_pairing_rejects_duplicate_record_keys(tmp_path):
    duplicate = tmp_path / "duplicate.npz"
    np.savez(
        duplicate,
        clip=np.asarray([250, 250]),
        h=np.asarray([1, 1]),
        fg=np.asarray([0.1, 0.2]),
    )
    with pytest.raises(ValueError, match="duplicate paired record key"):
        _load_canonicalized(duplicate)
