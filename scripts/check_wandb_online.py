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

    # Exercise the operation required by the scientific run itself.  The public
    # ``Api.viewer`` query is not part of the logging path and can fail for an
    # otherwise valid login (observed with W&B 0.27.2), so it is not a sound
    # fail-closed launch criterion.
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
    if run is None or not run.url:
        raise RuntimeError("W&B online run creation unavailable")
    run.log({"gate/pass": 1})
    run.summary["gate_status"] = "PASS"
    url = run.url
    run.finish()
    print(f"WANDB_ONLINE_PASS {url}")


if __name__ == "__main__":
    main()
