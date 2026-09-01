"""Replay locally authoritative Article 4 evidence into W&B after a tracking outage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.article4_campaign import load_campaign_run_spec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "experiments/v2/campaigns/article4_primary_30k_2026-09-01.json"
RESOLUTION = ROOT / "artifacts/v2/article4_primary_30k/dependency_resolution.json"


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def all_entries(campaign: dict) -> list[dict]:
    entries = list(campaign["run_order"])
    if RESOLUTION.exists():
        entries.extend(json.loads(RESOLUTION.read_text(encoding="utf-8"))["run_order"])
    return entries


def sync(spec: object, *, force: bool) -> str:
    import wandb

    run_dir = ROOT / spec.output_root / f"{spec.experiment_id}__{spec.content_hash[:12]}"
    status_path = run_dir / "status.json"
    manifest_path = run_dir / "run_manifest.json"
    wandb_status_path = run_dir / "wandb_status.json"
    if not status_path.exists() or not manifest_path.exists():
        return "not_started"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    mirror = json.loads(wandb_status_path.read_text(encoding="utf-8")) if wandb_status_path.exists() else {}
    if mirror.get("state") == "synced_complete" and not force:
        return "already_synced"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = f"{spec.experiment_id}-{spec.content_hash[:12]}".replace("/", "-")
    run = wandb.init(
        project=spec.wandb_project,
        group=spec.wandb_group,
        name=spec.experiment_id,
        id=run_id,
        resume="allow",
        job_type="article4-primary-training",
        tags=["article4", spec.family.lower(), spec.policy, spec.overlap, f"seed-{spec.seed}", "local-replay"],
        config={**spec.canonical_dict(), "config_hash": spec.content_hash, "replayed_from_local": True},
        settings=wandb.Settings(init_timeout=60),
    )
    if run is None:
        raise RuntimeError(f"W&B initialization failed for {spec.experiment_id}")
    for record in records(run_dir / "training_metrics.jsonl"):
        step = int(record["step"])
        run.log(
            {f"train/{key}": value for key, value in record.items() if isinstance(value, (int, float))},
            step=step,
        )
    for record in records(run_dir / "validation_metrics.jsonl"):
        step = int(record["step"])
        run.log(
            {f"validation/{key}": value for key, value in record.items() if isinstance(value, (int, float))},
            step=step,
        )
        panel = run_dir / "panels" / f"panel_step{step:06d}.png"
        if panel.exists():
            run.log(
                {"validation/fixed_prediction_panel": wandb.Image(str(panel), caption=f"fixed validation step {step}")},
                step=step,
            )
    for key, value in status.items():
        run.summary[key] = value
    artifact = wandb.Artifact(
        name=f"{run_id}-evidence",
        type="article4-run-evidence",
        metadata={"config_hash": spec.content_hash, "replayed": True},
    )
    for path in (
        manifest_path,
        run_dir / "training_metrics.jsonl",
        run_dir / "validation_metrics.jsonl",
        status_path,
    ):
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact, aliases=["latest", "complete"] if status.get("status") == "complete" else ["latest"])
    url = str(run.url)
    run.finish()
    wandb_status_path.write_text(
        json.dumps(
            {"schema_version": 1, "state": "synced_complete" if status.get("status") == "complete" else "synced_partial", "run_id": run_id, "url": url},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--experiment-id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    outcomes = {}
    for entry in all_entries(campaign):
        if args.experiment_id and entry["experiment_id"] != args.experiment_id:
            continue
        spec = load_campaign_run_spec(ROOT / entry["config"])
        outcomes[spec.experiment_id] = sync(spec, force=args.force)
    if args.experiment_id and not outcomes:
        raise ValueError(f"unknown experiment id: {args.experiment_id}")
    print(json.dumps(outcomes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
