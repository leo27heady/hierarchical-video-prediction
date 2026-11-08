"""Best-effort W&B mirror for locally authoritative scientific run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WandbMirror:
    def __init__(
        self,
        *,
        run_dir: Path,
        project: str,
        group: str,
        experiment_id: str,
        config_hash: str,
        config: dict[str, Any],
        tags: Iterable[str],
        job_type: str = "article4-primary-training",
        disabled: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.status_path = run_dir / "wandb_status.json"
        self.run = None
        self.last_error: str | None = None
        self.run_id = f"{experiment_id}-{config_hash[:12]}".replace("/", "-")
        if disabled:
            self._status("disabled", reason="--no-wandb")
            return
        try:
            import wandb

            os.environ.setdefault("WANDB_SILENT", "true")
            self.run = wandb.init(
                project=project,
                group=group,
                name=experiment_id,
                id=self.run_id,
                resume="allow",
                job_type=job_type,
                tags=list(tags),
                config=config,
                dir=str(run_dir),
                # Keep W&B system monitoring enabled: GPU utilization/temperature, host CPU,
                # memory, and network traces complement the explicit PyTorch memory metrics.
                settings=wandb.Settings(init_timeout=180),
            )
            if self.run is None:
                raise RuntimeError("wandb.init returned no run")
            self._status("online", url=self.run.url)
        except Exception as error:  # noqa: BLE001 - tracking cannot kill training
            self.last_error = f"{type(error).__name__}: {error}"
            self.run = None
            self._status("unsynced", error=self.last_error)

    def _status(self, state: str, **extra: Any) -> None:
        payload = {
            "schema_version": 1,
            "state": state,
            "run_id": self.run_id,
            "updated_utc": _utc_now(),
            **extra,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.status_path)

    @property
    def url(self) -> str | None:
        return None if self.run is None else str(self.run.url)

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(dict(values), step=step)
        except Exception as error:  # noqa: BLE001
            self.last_error = f"{type(error).__name__}: {error}"
            self._status("runtime_error_local_continuation", error=self.last_error, url=self.url)

    def image(self, key: str, path: Path, *, step: int, caption: str) -> None:
        if self.run is None or not path.exists():
            return
        try:
            import wandb

            self.run.log({key: wandb.Image(str(path), caption=caption)}, step=step)
        except Exception as error:  # noqa: BLE001
            self.last_error = f"{type(error).__name__}: {error}"
            self._status("runtime_error_local_continuation", error=self.last_error, url=self.url)

    def finish(
        self,
        *,
        summary: dict[str, Any],
        artifact_paths: Iterable[Path] = (),
    ) -> None:
        if self.run is None:
            return
        try:
            import wandb

            for key, value in summary.items():
                self.run.summary[key] = value
            paths = [path for path in artifact_paths if path.exists()]
            if paths:
                artifact = wandb.Artifact(
                    name=f"{self.run_id}-evidence",
                    type="article4-run-evidence",
                    metadata={"config_hash": summary.get("config_hash")},
                )
                for path in paths:
                    artifact.add_file(str(path), name=path.name)
                self.run.log_artifact(artifact, aliases=["latest", "complete"])
            url = self.url
            self.run.finish()
            self._status("synced_complete", url=url)
        except Exception as error:  # noqa: BLE001
            self.last_error = f"{type(error).__name__}: {error}"
            self._status("finish_error_local_complete", error=self.last_error, url=self.url)
        finally:
            self.run = None
