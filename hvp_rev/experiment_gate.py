"""Outcome-independent experiment authorization and prerequisite checks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LAUNCHABLE = {"AUTHORIZED", "AUTHORIZED_FINISH", "AUTHORIZED_RUN_ALONE"}


@dataclass(frozen=True)
class GateResult:
    experiment: str
    status: str
    launchable: bool
    failures: tuple[str, ...]

    def require(self) -> None:
        if not self.launchable:
            detail = "; ".join(self.failures) or f"status is {self.status}"
            raise PermissionError(f"experiment {self.experiment!r} is not launchable: {detail}")


def load_authorizations(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("experiments"), dict):
        raise ValueError("invalid experiment authorization schema")
    return value


def check_experiment(name: str, path: str | Path, *, root: str | Path = ".") -> GateResult:
    ledger = load_authorizations(path)
    if name not in ledger["experiments"]:
        return GateResult(name, "UNREGISTERED", False, ("experiment is absent from the ledger",))
    spec = ledger["experiments"][name]
    status = spec.get("status", "UNSPECIFIED")
    failures: list[str] = []
    if status not in LAUNCHABLE:
        failures.append(f"status is {status}")
    root = Path(root)
    for relative in spec.get("required_files", []):
        if not (root / relative).is_file():
            failures.append(f"missing required file: {relative}")
    for marker in spec.get("required_markers", []):
        marker_path = root / marker["path"]
        if not marker_path.is_file():
            failures.append(f"missing marker: {marker['path']}")
        elif marker.get("contains") and marker["contains"] not in marker_path.read_text(encoding="utf-8"):
            failures.append(f"marker {marker['path']} lacks {marker['contains']!r}")
    return GateResult(name, status, not failures, tuple(failures))
