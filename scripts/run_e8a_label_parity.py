"""CPU parity gate for E8a geometric labels against the legacy RHVD implementation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file
from hvp_rev.rhvd import DtLabeler, hardness_at
from hvp_rev.step1_mech import rel_positions
from scripts.evaluate_e8a_pose_mosaic import geometric_labels


def main() -> None:
    clips_path = Path("data/dev/e2_test_fixture.npy")
    actions_path = Path("data/dev/e2_test_fixture_actions.npy")
    positions_path = Path("data/dev/e2_test_fixture_positions.npy")
    clips = np.load(clips_path, mmap_mode="r")
    actions = np.load(actions_path, mmap_mode="r")
    positions = np.load(positions_path, mmap_mode="r")
    data = SimpleNamespace(clips_u8=clips, acts=actions, W=192, speed=4)
    checked = min(5, len(clips))
    records = 0
    for clip in range(checked):
        relative = rel_positions(actions[clip], 4)
        if np.unique((positions[clip] - relative) % 192).size != 1:
            raise AssertionError("saved positions differ from action-derived positions beyond offset")
        hard, delay, seen = geometric_labels(
            clips[clip], positions[clip], world=192, warm=48, threshold=0.5
        )
        labeler = DtLabeler(data, 48)
        expected_hard, expected_delay, expected_seen = [], [], []
        for frame in range(48, clips.shape[1]):
            expected_hard.append(hardness_at(data, clip, frame, n_short=6, thr=0.5))
            dt_value, seen_value = labeler.label(clip, frame)
            expected_delay.append(dt_value)
            expected_seen.append(seen_value)
        np.testing.assert_array_equal(hard, np.asarray(expected_hard))
        np.testing.assert_array_equal(delay, np.asarray(expected_delay))
        np.testing.assert_array_equal(seen, np.asarray(expected_seen))
        records += len(hard)
    report = {
        "schema_version": 1,
        "status": "E8A_GEOMETRIC_LABEL_PARITY_PASS",
        "clips": checked,
        "records": records,
        "fixtures": {
            path.as_posix(): sha256_file(path)
            for path in (clips_path, actions_path, positions_path)
        },
    }
    atomic_write_json("gates/e8a_label_parity_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
