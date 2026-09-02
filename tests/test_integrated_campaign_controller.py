from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.v2.run_integrated_openstl_campaign import (
    controller_lock,
    load_manifest,
    run_status,
    sha256,
    validate_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments" / "v2" / "campaigns" / "integrated_openstl_confirmation_2026-09-02.json"


def test_final_manifest_is_exact_two_by_two_and_prelock():
    manifest = load_manifest(CAMPAIGN)
    assert manifest["launch_authorized"] is False
    assert [(run["dataset"], run["seed"]) for run in manifest["runs"]] == [
        ("mmnist", 0), ("mmnist", 1), ("taxibj", 0), ("taxibj", 1)
    ]
    assert len({run["run_id"] for run in manifest["runs"]}) == 4


def test_status_is_pending_without_creating_run_directories(tmp_path):
    manifest = load_manifest(CAMPAIGN)
    manifest["output_root"] = str(tmp_path / "outputs")
    rows = run_status(manifest)
    assert [row["status"] for row in rows] == ["pending"] * 4
    assert not (tmp_path / "outputs").exists()


def test_controller_lock_rejects_duplicate_owner(tmp_path):
    with controller_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another controller"):
            with controller_lock(tmp_path):
                pass
    assert not (tmp_path / "controller.lock").exists()


def test_authorization_is_bound_to_manifest_and_lock_fingerprint(tmp_path):
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({
        "approved": True,
        "manifest_sha256": sha256(CAMPAIGN),
        "lock_fingerprint": "1" * 64,
        "approved_by": "test",
    }), encoding="utf-8")
    assert validate_authorization(CAMPAIGN, authorization, "1" * 64)["approved"] is True
    with pytest.raises(PermissionError, match="locked code/config/data"):
        validate_authorization(CAMPAIGN, authorization, "2" * 64)
    authorization.write_text(json.dumps({
        "approved": True,
        "manifest_sha256": "0" * 64,
    }), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not match"):
        validate_authorization(CAMPAIGN, authorization)
