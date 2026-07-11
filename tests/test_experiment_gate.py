import json

import pytest

from hvp_rev.experiment_gate import check_experiment


def test_gate_fails_closed_for_blocked_and_unknown(tmp_path):
    ledger = tmp_path / "authorization.json"
    ledger.write_text(json.dumps({
        "schema_version": 1,
        "experiments": {"blocked": {"status": "BLOCKED", "required_files": []}},
    }))
    assert not check_experiment("blocked", ledger, root=tmp_path).launchable
    assert not check_experiment("missing", ledger, root=tmp_path).launchable


def test_gate_requires_files_and_markers(tmp_path):
    ledger = tmp_path / "authorization.json"
    ledger.write_text(json.dumps({
        "schema_version": 1,
        "experiments": {
            "ready": {
                "status": "AUTHORIZED",
                "required_files": ["protocol.md"],
                "required_markers": [{"path": "gate.txt", "contains": "PASS"}],
            }
        },
    }))
    first = check_experiment("ready", ledger, root=tmp_path)
    assert not first.launchable and len(first.failures) == 2
    (tmp_path / "protocol.md").write_text("protocol")
    (tmp_path / "gate.txt").write_text("PASS")
    assert check_experiment("ready", ledger, root=tmp_path).launchable
