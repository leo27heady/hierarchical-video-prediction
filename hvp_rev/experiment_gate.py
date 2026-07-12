"""Outcome-independent experiment authorization and prerequisite checks."""
from __future__ import annotations

import json
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _attestation_failures(marker_path: Path, expected: dict, root: Path) -> list[str]:
    try:
        value = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return [f"marker {marker_path} is not valid JSON attestation: {error}"]
    failures = []
    if value.get("schema_version") != 1:
        failures.append(f"marker {marker_path} has invalid schema")
    if value.get("status") != expected.get("status"):
        failures.append(
            f"marker {marker_path} status {value.get('status')!r} != {expected.get('status')!r}"
        )
    subject = expected.get("subject")
    if subject:
        if value.get("subject") != subject:
            failures.append(f"marker {marker_path} attests wrong subject")
        subject_path = (root / subject).resolve()
        try:
            subject_path.relative_to(root.resolve())
        except ValueError:
            failures.append(f"attested subject escapes project root: {subject}")
        else:
            if not subject_path.is_file():
                failures.append(f"attested subject missing: {subject}")
            elif value.get("subject_sha256") != _sha256(subject_path):
                failures.append(f"attested subject hash mismatch: {subject}")
    if not value.get("source_commit"):
        failures.append(f"marker {marker_path} lacks source_commit")
    return failures


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
        elif marker.get("attests"):
            failures.extend(_attestation_failures(marker_path, marker["attests"], root))
        elif marker.get("contains") and marker["contains"] not in marker_path.read_text(encoding="utf-8"):
            failures.append(f"marker {marker['path']} lacks {marker['contains']!r}")
    return GateResult(name, status, not failures, tuple(failures))
