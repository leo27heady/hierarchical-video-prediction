"""Two-hour-safe health audit for the currently active project experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, canonical_json


PROGRESS = re.compile(
    r"\[rollout (?P<arm>[^]]+)\] clips (?P<first>\d+)\.\.(?P<last>\d+) done .*"
    r"elapsed (?P<hours>[0-9.]+)h; running fg (?P<fg>[0-9.]+)"
)


def _process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"if(Get-Process -Id {pid} -ErrorAction SilentlyContinue){{exit 0}}else{{exit 1}}"],
        capture_output=True,
    )
    return result.returncode == 0


def _gpu_snapshot() -> dict:
    try:
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=15).strip().split(",")
        return {
            "utilization_percent": float(text[0]),
            "memory_used_mib": float(text[1]),
            "memory_total_mib": float(text[2]),
            "temperature_c": float(text[3]),
        }
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def _wandb_state(url: str | None) -> dict:
    if not url:
        return {"error": "missing W&B URL"}
    try:
        # https://wandb.ai/entity/project/runs/id
        parts = url.rstrip("/").split("/")
        run_path = "/".join((parts[-4], parts[-3], parts[-1]))
        import wandb
        run = wandb.Api(timeout=30).run(run_path)
        summary = dict(run.summary)
        return {
            "path": run_path,
            "state": run.state,
            "progress": summary.get("health/progress"),
            "running_fg_mse": summary.get("health/running_fg_mse"),
            "fg_mse_all": summary.get("eval/fg_mse_all"),
        }
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", default="artifacts/active_experiment.json")
    parser.add_argument("--out-dir", default="artifacts/monitoring")
    parser.add_argument("--max-log-age-minutes", type=float, default=150.0)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    active_path = Path(args.active)
    warnings, failures = [], []
    if not active_path.is_file():
        result = {
            "schema_version": 1,
            "timestamp_utc": now.isoformat(),
            "status": "IDLE",
            "message": "No active experiment registry exists.",
        }
    else:
        active = json.loads(active_path.read_text(encoding="utf-8"))
        status = active.get("status")
        pid = int(active.get("pid", -1))
        process_alive = _process_exists(pid)
        log_path = Path(active.get("run_log") or "")
        last_progress = None
        log_age_minutes = None
        if log_path.is_file():
            log_age_minutes = (now.timestamp() - log_path.stat().st_mtime) / 60
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]
            for line in reversed(lines):
                match = PROGRESS.search(line)
                if match:
                    last_progress = {
                        "arm": match.group("arm"),
                        "clips_complete": int(match.group("last")) + 1,
                        "elapsed_hours": float(match.group("hours")),
                        "running_fg_mse": float(match.group("fg")),
                    }
                    break
        elif status == "running":
            failures.append("active run log is missing")

        if status == "running" and not process_alive:
            failures.append("active registry says running but PID is absent")
        if status == "running" and log_age_minutes is not None and log_age_minutes > args.max_log_age_minutes:
            failures.append(f"run log stale for {log_age_minutes:.1f} minutes")
        if last_progress:
            fg = last_progress["running_fg_mse"]
            if not math.isfinite(fg) or not 0 <= fg <= 0.75:
                failures.append(f"running foreground MSE outside broad validity range: {fg}")
            legacy_tag = "_60k" if int(active.get("optimizer_updates", 0)) == 60000 else ""
            legacy_path = Path(
                f"logs_rhcp/rhvd_eval_{active['arm']}_s{active['seed']}{legacy_tag}.npz"
            )
            if legacy_path.is_file():
                with np.load(legacy_path) as legacy:
                    legacy_mean = float(legacy["fg"].mean())
                last_progress["legacy_development_fg_mse"] = legacy_mean
                if abs(fg - legacy_mean) > 0.15:
                    warnings.append(
                        f"confirmation running mean differs from development by {fg-legacy_mean:+.3f}"
                    )

        started = active.get("started_utc")
        elapsed = None
        if started:
            elapsed = (now - datetime.fromisoformat(started)).total_seconds() / 3600
            maximum = float(active.get("authorized_gpu_hours", 0) or 0)
            if status == "running" and maximum and elapsed > maximum:
                failures.append(f"run elapsed {elapsed:.3f}h exceeds authorized {maximum:.3f}h")

        wandb_state = _wandb_state(active.get("wandb_url"))
        if status == "running" and "error" in wandb_state:
            failures.append(f"W&B health unavailable: {wandb_state['error']}")
        result = {
            "schema_version": 1,
            "timestamp_utc": now.isoformat(),
            "status": "FAIL" if failures else ("WARN" if warnings else status.upper()),
            "active": active,
            "process_alive": process_alive,
            "elapsed_hours": elapsed,
            "log_age_minutes": log_age_minutes,
            "last_progress": last_progress,
            "gpu": _gpu_snapshot(),
            "wandb": wandb_state,
            "warnings": warnings,
            "failures": failures,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "latest.json", result)
    with (out_dir / "e2_health.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(result) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
