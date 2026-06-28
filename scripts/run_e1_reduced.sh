#!/usr/bin/env bash
# E1 reduced first pass: 3 arms x 3 training seeds, M=8, warm-started. Each run resumable
# (trainer loads its own _last.pt), so re-launching this driver after any crash continues.
cd "C:/Users/leoni/Documents/projects/hierarchical-video-prediction" || exit 1
PY="venv/Scripts/python.exe"
mkdir -p logs_rollout
for arm in lambda0 nograd withgrad; do
  for seed in 0 1 2; do
    name="e1_${arm}_s${seed}"
    echo "=== $(date +%H:%M:%S) launching ${name} ==="
    "$PY" -m hvp_rev.train_rollout --arm "$arm" --seed "$seed" --steps 8000 --batch 32 \
      --lr 1e-4 --eval-every 2000 --ckpt-every 2000 --log-every 200 --run-name "$name" \
      > "logs_rollout/${name}.log" 2>&1 || echo "!!! ${name} exited nonzero (will resume on relaunch)"
  done
done
echo "=== $(date +%H:%M:%S) ALL E1 RUNS DONE ==="
