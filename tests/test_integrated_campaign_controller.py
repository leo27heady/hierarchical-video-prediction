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


def test_gpu_idle_allows_desktop_but_rejects_load():
    from scripts.v2.run_integrated_openstl_campaign import gpu_idle
    gpu = dict(available=True, utilization_percent=15, memory_used_mib=1719, memory_total_mib=16380)
    assert gpu_idle(gpu)
    assert not gpu_idle({**gpu, 'utilization_percent': 60})
    assert not gpu_idle({**gpu, 'memory_used_mib': 6500})


def test_idle_wait_rejects_python_cuda_even_at_low_utilization(monkeypatch):
    from scripts.v2 import run_integrated_openstl_campaign as module
    monkeypatch.setattr(module, 'gpu_state', lambda: dict(available=True, utilization_percent=0, memory_used_mib=1000, memory_total_mib=16380))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: '123, python.exe')
    with pytest.raises(RuntimeError, match='Python CUDA'):
        module.wait_for_idle_gpu(timeout=0)


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


def test_dead_controller_lock_is_archived_for_restart(tmp_path):
    (tmp_path / 'controller.lock').write_text(json.dumps({'pid': 2147483647}))
    with controller_lock(tmp_path):
        assert list(tmp_path.glob('controller.lock.stale.*.json'))


def test_successful_exit_without_matching_result_stops_queue(tmp_path, monkeypatch):
    import argparse
    from scripts.v2 import run_integrated_openstl_campaign as module
    manifest = load_manifest(CAMPAIGN)
    manifest.update(launch_authorized=True, output_root=str(tmp_path))
    monkeypatch.setattr(module, 'preflight', lambda *a, **k: ({'lock_fingerprint': 'fixture'}, True))
    monkeypatch.setattr(module, 'validate_authorization', lambda *a: {'approved': True})
    launched = []
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            launched.append(command)
        def wait(self): return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    args = argparse.Namespace(allow_busy_gpu=False, check_wandb=False, authorization=tmp_path / 'unused',
        wandb_mode='disabled', num_workers=0, log_every=10, checkpoint_every=50,
        no_wandb=True, continue_on_failure=False)
    assert module.execute(CAMPAIGN, manifest, args) == 3
    assert len(launched) == 1
