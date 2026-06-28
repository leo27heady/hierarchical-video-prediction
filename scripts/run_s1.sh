#!/usr/bin/env bash
# Step-1 renderer: 1 seed x 3 arms first (l2/lpips/gan) for the decisive A/B/C spectrum.
# Resumable (trainer loads its own _last.pt). Add seeds only if the gan arm shows signal.
cd "C:/Users/leoni/Documents/projects/hierarchical-video-prediction" || exit 1
PY="venv/Scripts/python.exe"
SEED="${1:-0}"
for arm in l2 lpips gan; do
  name="s1_${arm}_s${SEED}"
  echo "=== $(date +%H:%M:%S) ${name} ==="
  "$PY" -m hvp_rev.train_renderer --arm "$arm" --seed "$SEED" --steps 8000 \
    --train-clips 120 --eval-clips 24 --eval-every 2000 --ckpt-every 2000 \
    --run-name "$name" > "logs_render/${name}.log" 2>&1 || echo "!!! ${name} nonzero (resumes on relaunch)"
done
echo "=== $(date +%H:%M:%S) S1 seed ${SEED} DONE ==="
