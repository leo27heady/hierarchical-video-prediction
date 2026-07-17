"""CPU-only semantic validation for registries and experiment configuration files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.arm_registry import FIXED_TARGET_CONTROLS, validate_fixed_target_pair
from hvp_rev.experiment_gate import load_authorizations


def main() -> None:
    validate_fixed_target_pair()
    config_dir = Path("experiments/configs")
    configs = {}
    for path in sorted(config_dir.glob("*.json")):
        configs[path.name] = json.loads(path.read_text(encoding="utf-8"))
    if len(configs) < 4:
        raise AssertionError("expected all prepared experiment configurations")
    ledger = load_authorizations("experiments/authorization.json")
    if ledger["experiments"]["phase-c-legacy"]["status"] != "PROHIBITED_INVALID_ENDPOINTS":
        raise AssertionError("legacy Phase C must remain prohibited")
    if ledger["experiments"]["phase-c-v2"]["status"] in {"AUTHORIZED", "AUTHORIZED_RUN_ALONE"}:
        phasec = configs["phase_c_v2.json"]
        if phasec.get("status") != "FROZEN":
            raise AssertionError("Phase C v2 authorization requires a frozen scientific config")
        for marker in (
            "gates/phase_c_protocol_frozen.json",
            "gates/phase_c_datasets_final_validated.json",
            "gates/phase_c_classifier_validated.json",
            "gates/phase_c_evaluator_validated.json",
        ):
            if not Path(marker).is_file():
                raise AssertionError(f"Phase C v2 authorization missing {marker}")
    hier = FIXED_TARGET_CONTROLS["hier-s2-fixed-target"]
    flat = FIXED_TARGET_CONTROLS["flat-short-fixed-target"]
    print(f"B1 fixed-target: {hier.context_tokens} context tokens, "
          f"{hier.target_tokens} target tokens, reaches {hier.context_reach}/{flat.context_reach}")
    print(f"validated {len(configs)} experiment configs and fail-closed authorization ledger")


if __name__ == "__main__":
    main()
