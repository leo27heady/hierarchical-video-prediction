"""Record a fail-closed cloud-completeness audit for a scientific W&B group."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wandb

from hvp_rev.provenance import atomic_write_json


def parse_minimum(value: str) -> tuple[str, int]:
    try:
        name, count = value.rsplit("=", 1)
        minimum = int(count)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected JOB_TYPE=COUNT") from exc
    if not name or minimum < 0:
        raise argparse.ArgumentTypeError("expected non-empty JOB_TYPE and non-negative COUNT")
    return name, minimum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="leo27heady")
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--min-runs", type=int, required=True)
    parser.add_argument(
        "--min-job-type",
        action="append",
        default=[],
        type=parse_minimum,
        metavar="JOB_TYPE=COUNT",
    )
    parser.add_argument(
        "--require-artifact-job",
        action="append",
        default=[],
        metavar="JOB_TYPE",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # The desktop process can inherit a stale key. Removing only that process-level value lets
    # W&B use the user's authenticated _netrc without exposing credentials to the report.
    os.environ.pop("WANDB_API_KEY", None)
    api = wandb.Api(timeout=90)
    runs = list(
        api.runs(
            f"{args.entity}/{args.project}",
            filters={"group": args.group},
            per_page=200,
        )
    )
    rows = []
    failures: list[str] = []
    for run in sorted(runs, key=lambda item: (item.created_at or "", item.id)):
        artifacts = [
            {
                "name": artifact.name,
                "type": artifact.type,
                "digest": artifact.digest,
            }
            for artifact in run.logged_artifacts()
        ]
        rows.append({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "job_type": run.job_type or "",
            "created_at": run.created_at,
            "url": run.url,
            "logged_artifacts": artifacts,
        })
    job_counts = Counter(row["job_type"] for row in rows)
    if len(rows) < args.min_runs:
        failures.append(f"group has {len(rows)} runs; minimum is {args.min_runs}")
    unfinished = [row["id"] for row in rows if row["state"] != "finished"]
    if unfinished:
        failures.append(f"unfinished W&B runs: {unfinished}")
    for job_type, minimum in args.min_job_type:
        if job_counts[job_type] < minimum:
            failures.append(
                f"job type {job_type!r} has {job_counts[job_type]} runs; minimum is {minimum}"
            )
    for job_type in args.require_artifact_job:
        missing = [
            row["id"]
            for row in rows
            if row["job_type"] == job_type and not row["logged_artifacts"]
        ]
        if missing:
            failures.append(f"{job_type!r} runs without logged artifacts: {missing}")

    report = {
        "schema_version": 1,
        "status": "WANDB_GROUP_AUDIT_PASS" if not failures else "FAIL",
        "entity": args.entity,
        "project": args.project,
        "group": args.group,
        "runs": rows,
        "run_count": len(rows),
        "job_type_counts": dict(sorted(job_counts.items())),
        "requirements": {
            "minimum_runs": args.min_runs,
            "minimum_job_types": dict(args.min_job_type),
            "artifact_required_job_types": sorted(set(args.require_artifact_job)),
            "all_runs_finished": True,
        },
        "failures": failures,
    }
    atomic_write_json(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "run_count": report["run_count"],
        "job_type_counts": report["job_type_counts"],
        "failures": failures,
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
