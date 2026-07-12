"""Fail-closed W&B connectivity and write-path gate for scientific launches."""
from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="hgrid")
    parser.add_argument("--group", default="launch-gates")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=30)
    viewer = api.viewer
    if not viewer:
        raise RuntimeError("W&B viewer/authentication unavailable")
    run = wandb.init(
        project=args.project,
        name=args.run_id,
        id=args.run_id,
        resume="never",
        group=args.group,
        job_type="connectivity-gate",
        tags=["cpu-gate", "prelaunch"],
        config={"gate": "wandb-online-write", "unix_time": time.time()},
        settings=wandb.Settings(init_timeout=60, _disable_stats=True),
    )
    run.log({"gate/pass": 1})
    run.summary["gate_status"] = "PASS"
    url = run.url
    run.finish()
    print(f"WANDB_ONLINE_PASS {url}")


if __name__ == "__main__":
    main()
