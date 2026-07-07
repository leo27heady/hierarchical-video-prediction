#!/bin/bash
# H-grid Tier-A chain 1 (seed 0): decisive-pair-first ordering per HGRID_DESIGN.md v2 §7.
# d1-diff (2.5h) -> d0-diff-long (6.1h) => G-H1 readable; then d1-reg (2.5h) -> d0-reg-long
# (6.1h) => G-H2. Uniform regime: B=8 x 40k, bf16, EMA 0.999, warmup 2k. One job at a time.
set -e
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
COMMON="--batch 8 --steps 40000 --bf16 --train_path data/hgrid_train_1200x480.npy --test_path data/hgrid_test_300x480.npy --world 768 --warm 192 --n_val 150"

for ARM in d1-diff d0-diff-long d1-reg d0-reg-long; do
  echo "=== [chain1] $ARM start $(date) ==="
  $PY -u -m hvp_rev.rhvd --arm $ARM --seed 0 $COMMON 2>&1 | tee logs_rhcp/rhvd_$ARM.log
  echo "=== [chain1] $ARM done $(date) ==="
done
echo "CHAIN1 COMPLETE $(date)"
