"""Single-command preflight, launch, resume, and status controller for Article 4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import numpy as np
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.article4_campaign import (
    file_sha256,
    load_campaign_run_spec,
    validate_campaign_authorization,
)
from hvp_hca.data import load_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "experiments/v2/campaigns/article4_primary_30k_2026-09-01.json"
STATE_ROOT = ROOT / "artifacts/v2/article4_primary_30k/campaign"
STATE_PATH = STATE_ROOT / "state.json"
SMOKE_STATE_PATH = STATE_ROOT / "smoke_state.json"
EVENTS_PATH = STATE_ROOT / "events.jsonl"
PREFLIGHT_PATH = STATE_ROOT / "preflight.json"
RESOLUTION_PATH = ROOT / "artifacts/v2/article4_primary_30k/dependency_resolution.json"
POST_REPORT_PATH = ROOT / "artifacts/v2/article4_primary_30k/post_training/final_report.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    return commit, dirty


def validate_entries(campaign_path: Path, entries: list[dict[str, Any]]) -> None:
    ids = set()
    hashes = set()
    for entry in entries:
        spec = load_campaign_run_spec(ROOT / entry["config"])
        validate_campaign_authorization(spec, campaign_path)
        if entry["experiment_id"] != spec.experiment_id or entry["config_hash"] != spec.content_hash:
            raise ValueError(f"campaign entry drift: {entry['experiment_id']}")
        if spec.experiment_id in ids or spec.content_hash in hashes:
            raise ValueError("campaign run ids and config hashes must be unique")
        ids.add(spec.experiment_id)
        hashes.add(spec.content_hash)


def wandb_gate(campaign: dict[str, Any], campaign_hash: str) -> str:
    import wandb

    run_id = f"article4-30k-preflight-{campaign_hash[:12]}"
    run = wandb.init(
        project=campaign["wandb"]["project"],
        group="launch-gates",
        name=run_id,
        id=run_id,
        resume="allow",
        job_type="connectivity-gate",
        tags=["article4", "preflight", "cpu-gate"],
        config={"campaign_sha256": campaign_hash, "gate": "online-write"},
        settings=wandb.Settings(init_timeout=60, _disable_stats=True),
    )
    if run is None or not run.url:
        raise RuntimeError("W&B online run creation unavailable")
    run.log({"gate/pass": 1, "gate/unix_time": time.time()})
    run.summary["gate_status"] = "PASS"
    url = str(run.url)
    run.finish()
    return url


def preflight(campaign_path: Path, *, online_wandb: bool, allow_dirty: bool = False) -> dict[str, Any]:
    campaign_path = campaign_path.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_hash = file_sha256(campaign_path)
    if campaign.get("status") != "authorized_30k" or campaign.get("confirmed_train_updates") != 30_000:
        raise ValueError("campaign is not authorized at 30k")
    entries = list(campaign["run_order"])
    if len(entries) != 80 or len(campaign["authorized_runs"]) != 80:
        raise ValueError("campaign must contain exactly 80 wave-1 runs")
    validate_entries(campaign_path, entries)
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise RuntimeError("scientific campaign requires a clean worktree")
    manifest, manifest_hash = load_manifest(ROOT / campaign["dataset_manifest"])
    if manifest_hash != campaign["dataset_manifest_sha256"]:
        raise ValueError("campaign dataset manifest hash mismatch")
    for name in ("clips", "actions"):
        artifact = manifest["artifacts"][name]
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"TRB {name} source hash mismatch")
    clips = np.load(manifest["artifacts"]["clips"]["path"], mmap_mode="r")
    actions = np.load(manifest["artifacts"]["actions"]["path"], mmap_mode="r")
    configuration = manifest["configuration"]
    expected_shape = (
        int(configuration["clips"]),
        int(configuration["frames_per_clip"]),
        int(configuration["viewport"]),
        int(configuration["viewport"]),
    )
    if clips.shape != expected_shape or clips.dtype != np.uint8:
        raise ValueError(f"TRB clip schema mismatch: {clips.shape}/{clips.dtype}")
    if actions.shape != expected_shape[:2] or actions.dtype != np.int8:
        raise ValueError(f"TRB action schema mismatch: {actions.shape}/{actions.dtype}")
    if set(np.unique(actions).tolist()) != {-1, 0, 1}:
        raise ValueError("TRB action vocabulary mismatch")
    split = manifest["split"]
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(split["validation_indices"], dtype=np.int64)
    if len(train_indices) != int(split["train_count"]) or len(validation_indices) != int(
        split["validation_count"]
    ):
        raise ValueError("TRB split count mismatch")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("TRB train/validation split overlap")
    if set(np.concatenate((train_indices, validation_indices)).tolist()) != set(
        range(expected_shape[0])
    ):
        raise ValueError("TRB split is not an exhaustive clip partition")
    base = load_campaign_run_spec(ROOT / entries[0]["config"])
    if file_sha256(base.ae_path) != base.ae_sha256:
        raise ValueError("base autoencoder hash mismatch")
    top_path = Path(
        "C:/Users/leoni/Documents/projects/hierarchical-video-prediction/logs_rhcp/hier_ae_mom_top.pt"
    )
    if not top_path.exists():
        raise FileNotFoundError("A4-LVL frozen top autoencoder is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Article 4 campaign")
    free_disk = shutil.disk_usage(ROOT).free
    if free_disk < 25 * 2**30:
        raise RuntimeError(f"at least 25 GiB free disk required; found {free_disk / 2**30:.1f}")
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    if total_gpu < 12 * 2**30:
        raise RuntimeError(f"campaign requires a >=12 GiB CUDA device; found {total_gpu / 2**30:.1f}")
    wandb_url = wandb_gate(campaign, campaign_hash) if online_wandb else None
    result = {
        "schema_version": 1,
        "status": "PASS",
        "campaign_id": campaign["campaign_id"],
        "campaign_file": str(campaign_path.relative_to(ROOT)).replace("\\", "/"),
        "campaign_sha256": campaign_hash,
        "source_commit": commit,
        "git_dirty": dirty,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_free_gib": free_gpu / 2**30,
        "gpu_total_gib": total_gpu / 2**30,
        "disk_free_gib": free_disk / 2**30,
        "wave1_runs": 80,
        "dependent_runs": 6,
        "total_runs": 86,
        "train_updates_per_run": 30_000,
        "expected_direct_gpu_hours": [187, 205],
        "recommended_reserved_gpu_hours": [205, 234],
        "wandb_project": campaign["wandb"]["project"],
        "wandb_gate_url": wandb_url,
        "dataset_shape": list(clips.shape),
        "dataset_action_vocabulary": [-1, 0, 1],
        "train_clips": len(train_indices),
        "validation_clips": len(validation_indices),
        "host": socket.gethostname(),
        "completed_utc": utc_now(),
    }
    atomic_json(PREFLIGHT_PATH, result)
    append_jsonl(EVENTS_PATH, {"event": "preflight_pass", **result})
    print(json.dumps(result, indent=2))
    return result


def run_dir_for(spec: Any) -> Path:
    return ROOT / spec.output_root / f"{spec.experiment_id}__{spec.content_hash[:12]}"


def completed(spec: Any) -> bool:
    status_path = run_dir_for(spec) / "status.json"
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return (
        status.get("status") == "complete"
        and int(status.get("step", 0)) == 30_000
        and status.get("config_hash") == spec.content_hash
    )


def campaign_status(campaign_path: Path) -> dict[str, Any]:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    stages: dict[str, list[dict[str, Any]]] = {"wave1": list(campaign["run_order"]), "wave2": []}
    if RESOLUTION_PATH.exists():
        resolution = json.loads(RESOLUTION_PATH.read_text(encoding="utf-8"))
        stages["wave2"] = list(resolution.get("run_order", []))
    result: dict[str, Any] = {"campaign_id": campaign["campaign_id"], "stages": {}}
    for stage, entries in stages.items():
        counts = {"complete": 0, "running": 0, "failed": 0, "pending": 0}
        current = []
        for entry in entries:
            spec = load_campaign_run_spec(ROOT / entry["config"])
            path = run_dir_for(spec) / "status.json"
            state = "pending"
            step = 0
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                state = str(payload.get("status", "pending"))
                step = int(payload.get("step", 0))
                if state == "interrupted":
                    state = "failed"
            counts[state if state in counts else "failed"] += 1
            if state != "complete":
                current.append({"experiment_id": spec.experiment_id, "status": state, "step": step})
        result["stages"][stage] = {"counts": counts, "outstanding": current[:20]}
    if STATE_PATH.exists():
        result["controller"] = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if SMOKE_STATE_PATH.exists():
        result["last_smoke"] = json.loads(SMOKE_STATE_PATH.read_text(encoding="utf-8"))
    result["post_training"] = (
        json.loads(POST_REPORT_PATH.read_text(encoding="utf-8"))
        if POST_REPORT_PATH.exists()
        else {"status": "pending"}
    )
    return result


def tee_process(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] COMMAND {subprocess.list2cmdline(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(2)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
            raise


def launch_entries(
    *,
    campaign_path: Path,
    entries: list[dict[str, Any]],
    source_commit: str,
    resolution_path: Path | None,
    no_wandb: bool,
    smoke: bool,
    max_attempts: int,
) -> list[str]:
    failures = []
    for index, entry in enumerate(entries, start=1):
        spec_path = ROOT / entry["config"]
        spec = load_campaign_run_spec(spec_path)
        if not smoke and completed(spec):
            append_jsonl(EVENTS_PATH, {"event": "skip_complete", "experiment_id": spec.experiment_id, "utc": utc_now()})
            continue
        success = False
        for invocation_attempt in range(1, max_attempts + 1):
            state = {
                "status": "running",
                "stage": "wave2" if resolution_path else "wave1",
                "stage_index": index,
                "stage_total": len(entries),
                "experiment_id": spec.experiment_id,
                "invocation_attempt": invocation_attempt,
                "source_commit": source_commit,
                "updated_utc": utc_now(),
            }
            atomic_json(STATE_PATH, state)
            append_jsonl(EVENTS_PATH, {"event": "dispatch", **state})
            command = [
                sys.executable,
                "-u",
                "scripts/v2/train_article4_campaign.py",
                "--config",
                str(spec_path),
                "--campaign",
                str(campaign_path),
                "--source-commit",
                source_commit,
            ]
            if resolution_path:
                command += ["--resolution", str(resolution_path)]
            if no_wandb:
                command.append("--no-wandb")
            if smoke:
                command += ["--smoke", "--skip-source-hash", "--no-wandb"]
            log_path = STATE_ROOT / "logs" / f"{spec.experiment_id}.attempt{invocation_attempt}.log"
            exit_code = tee_process(command, log_path)
            if exit_code == 0:
                success = True
                break
            append_jsonl(
                EVENTS_PATH,
                {"event": "worker_nonzero", "experiment_id": spec.experiment_id, "exit_code": exit_code, "utc": utc_now()},
            )
        if not success:
            failures.append(spec.experiment_id)
    return failures


def resolve_wave2(campaign_path: Path) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-u",
        "scripts/v2/resolve_article4_dependencies.py",
        "--campaign",
        str(campaign_path),
        "--output",
        str(RESOLUTION_PATH),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    resolution = json.loads(RESOLUTION_PATH.read_text(encoding="utf-8"))
    if len(resolution.get("run_order", [])) != 6:
        raise ValueError("dependency resolution did not produce six runs")
    return list(resolution["run_order"])


def finalize_campaign(campaign_path: Path, source_commit: str, no_wandb: bool) -> None:
    command = [
        sys.executable,
        "-u",
        "scripts/v2/finalize_article4_campaign.py",
        "--campaign",
        str(campaign_path),
        "--source-commit",
        source_commit,
    ]
    if no_wandb:
        command.append("--no-wandb")
    exit_code = tee_process(command, STATE_ROOT / "logs/post_training_finalization.log")
    if exit_code:
        raise RuntimeError(f"post-training finalization failed with exit code {exit_code}")


def representative_smoke_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cover every wave-1 builder and both sequence-length extremes without 80 data reloads."""

    wanted = (
        "A4_FRT_B0320_L1_FLAT_RECENT_EXCL_S00_T30K",
        "A4_FRT_B3072_L2_GEOM_AGE_NEST_S00_T30K",
        "A4_ALC_B1792_L2_BAL_RECENT_NEST_S00_T30K",
        "A4_CEX_B1408_L2_C64_EXCL_S00_T30K",
        "A4_STS_B1792_L2_K08_GEOMETRIC_NEST_S00_T30K",
    )
    by_id = {entry["experiment_id"]: entry for entry in entries}
    if any(experiment_id not in by_id for experiment_id in wanted):
        raise ValueError("campaign is missing a representative smoke configuration")
    return [by_id[experiment_id] for experiment_id in wanted]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "status"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--execute", action="store_true", help="required safety latch for run")
    parser.add_argument("--no-wandb", action="store_true", help="debug only; normal campaign uses W&B")
    parser.add_argument("--smoke", action="store_true", help="run two updates per wave-1 config")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--allow-dirty", action="store_true", help="preflight/smoke development only")
    args = parser.parse_args()
    campaign_path = args.campaign.resolve()
    if args.command == "status":
        print(json.dumps(campaign_status(campaign_path), indent=2))
        return 0
    if args.command == "preflight":
        preflight(campaign_path, online_wandb=not args.no_wandb, allow_dirty=args.allow_dirty)
        return 0
    if not args.execute:
        parser.error("run requires --execute")
    gate = preflight(
        campaign_path,
        online_wandb=not args.no_wandb and not args.smoke,
        allow_dirty=args.allow_dirty and args.smoke,
    )
    source_commit = gate["source_commit"]
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    wave1_entries = list(campaign["run_order"])
    if args.smoke:
        wave1_entries = representative_smoke_entries(wave1_entries)
    wave1_failures = launch_entries(
        campaign_path=campaign_path,
        entries=wave1_entries,
        source_commit=source_commit,
        resolution_path=None,
        no_wandb=args.no_wandb,
        smoke=args.smoke,
        max_attempts=args.max_attempts,
    )
    if args.smoke:
        result = {"status": "smoke_complete", "failures": wave1_failures, "updated_utc": utc_now()}
        atomic_json(SMOKE_STATE_PATH, result)
        print(json.dumps(result, indent=2))
        return int(bool(wave1_failures))
    if wave1_failures:
        result = {"status": "wave1_incomplete", "failures": wave1_failures, "updated_utc": utc_now()}
        atomic_json(STATE_PATH, result)
        print(json.dumps(result, indent=2))
        return 1
    wave2 = resolve_wave2(campaign_path)
    wave2_failures = launch_entries(
        campaign_path=campaign_path,
        entries=wave2,
        source_commit=source_commit,
        resolution_path=RESOLUTION_PATH,
        no_wandb=args.no_wandb,
        smoke=False,
        max_attempts=args.max_attempts,
    )
    if not wave2_failures:
        finalize_campaign(campaign_path, source_commit, args.no_wandb)
    result = {
        "status": "complete" if not wave2_failures and POST_REPORT_PATH.exists() else "wave2_incomplete",
        "failures": wave2_failures,
        "source_commit": source_commit,
        "campaign_sha256": file_sha256(campaign_path),
        "dependency_resolution_sha256": file_sha256(RESOLUTION_PATH),
        "completed_utc": utc_now(),
    }
    atomic_json(STATE_PATH, result)
    append_jsonl(EVENTS_PATH, {"event": "campaign_terminal", **result})
    print(json.dumps(result, indent=2))
    return int(bool(wave2_failures))


if __name__ == "__main__":
    raise SystemExit(main())
