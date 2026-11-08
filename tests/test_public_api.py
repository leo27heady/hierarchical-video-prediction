"""Release-facing API and CLI smoke tests."""
from __future__ import annotations

import json

import numpy as np

import hvp
from hvp.cli import main
from hvp.context import ArmSpec, ContextGeometry


def test_public_import_is_cpu_safe_and_exposes_core_contract() -> None:
    assert hvp.__version__
    package = ArmSpec(name="example", coarse_context_slots=48, fine_recent_slots=2)
    assert package.context_tokens == 896
    assert package.context_reach == 48
    assert ContextGeometry(name="flat", fine_recent=14).reach == 14

    values = np.array([1.0, 2.0, 3.0, 4.0])
    clusters = np.array([0, 0, 1, 1])
    result = hvp.cluster_bootstrap_mean(values, clusters, n_boot=100, seed=7)
    assert result["mean"] == 2.5
    assert result["clusters"] == 2


def test_cli_unknown_experiment_fails_closed(tmp_path, capsys) -> None:
    ledger = tmp_path / "authorization.json"
    ledger.write_text(
        json.dumps({"schema_version": 1, "experiments": {}}),
        encoding="utf-8",
    )
    status = main(["gate", "not-registered", "--ledger", str(ledger), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert status == 2
    assert payload["status"] == "UNREGISTERED"
    assert payload["launchable"] is False
