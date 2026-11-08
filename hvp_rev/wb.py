"""Crash-safe wandb wrapper for the H-grid campaign. Every call is a no-op on any failure so
logging can NEVER take down a training run. One run per (arm, seed): deterministic run id ->
checkpoint-resumed trainings continue the same wandb run."""
from __future__ import annotations

import os

_run = None
_warned = False


def init(name, config=None, project="hgrid", run_id=None, tags=None, *, resume="allow",
         group=None, job_type=None, required=False):
    global _run, _warned
    try:
        import wandb
        _run = wandb.init(project=project, name=name, id=(run_id or name).replace("/", "-"),
                          resume=resume, config=config or {}, tags=tags or [], group=group,
                          job_type=job_type,
                          settings=wandb.Settings(init_timeout=60, _disable_stats=True))
        print(f"[wandb] run {name}: {_run.url}", flush=True)
    except Exception as e:  # noqa: BLE001
        if not _warned:
            print(f"[wandb] DISABLED ({type(e).__name__}: {e}) - training continues without it", flush=True)
            _warned = True
        _run = None
        if required:
            raise RuntimeError("W&B is required for this scientific run") from e
    return _run


def log(metrics, step=None):
    if _run is None:
        return
    try:
        _run.log(dict(metrics), step=step)
    except Exception:  # noqa: BLE001
        pass


def image(key, path, step=None, caption=None):
    if _run is None or not os.path.exists(path):
        return
    try:
        import wandb
        _run.log({key: wandb.Image(path, caption=caption or os.path.basename(path))}, step=step)
    except Exception:  # noqa: BLE001
        pass


def summary(d):
    if _run is None:
        return
    try:
        for k, v in d.items():
            _run.summary[k] = v
    except Exception:  # noqa: BLE001
        pass


def finish():
    global _run
    if _run is None:
        return
    try:
        _run.finish()
    except Exception:  # noqa: BLE001
        pass
    _run = None


def artifact(name, artifact_type, paths, *, metadata=None, aliases=None):
    """Upload small/medium scientific outputs and their manifests to the active run."""
    if _run is None:
        raise RuntimeError("cannot log W&B artifact without an active run")
    import wandb
    item = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        item.add_file(path)
    _run.log_artifact(item, aliases=aliases or [])
    return item


def active_url():
    return None if _run is None else _run.url
