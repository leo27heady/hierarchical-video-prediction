#!/usr/bin/env bash
# Auto-resume launcher for DLH training. Re-runs train.py with --auto-resume on crash, so a transient
# failure (OOM, CUDA hiccup) costs minutes not hours. train.py exits 0 when it reaches total_iters.
# Usage: scripts/run_until.sh <train.py args...>   (must include --run-name and --total-iters)
# Example: scripts/run_until.sh --dataset mmnist --levels 3 --run-name foo --total-iters 5000 ...
set -u
PY=venv/Scripts/python.exe
for attempt in $(seq 1 12); do
  echo "=== run_until attempt $attempt ===" >&2
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" -m hvp_dlh.train --auto-resume true "$@"
  ec=$?
  if [ $ec -eq 0 ]; then echo "=== COMPLETED (exit 0) ===" >&2; exit 0; fi
  echo "=== train.py crashed (exit $ec); resuming in 5s ===" >&2
  sleep 5
done
echo "=== gave up after 12 attempts ===" >&2
exit 1
