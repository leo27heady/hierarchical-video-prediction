#!/usr/bin/env bash
# Step-1 sweep: matched seq50/batch32 runs. A0 (tanh control) + sigmoid x {std 1.0, 0.5, 0.3}.
# All deltas are measured vs A0 (the matched control), not vs the seq100 baseline. Idempotent
# (each run uses --auto-resume, so re-running the sweep resumes incomplete runs / skips finished ones).
set -u
cd "$(dirname "$0")/.."
COMMON="--dataset mmnist --levels 3 --seq-len 50 --batch-size 32 --num-workers 0 \
 --data-root ../cwvae/data_mmnist --force-tick-iters 1500 --beta-warmup-iters 1500 \
 --total-iters 5000 --ckpt-every 1000 --eval-every 2500 --log-every 200 \
 --eval-clips 24 --eval-samples 5 --ctx-frames 36 --eval-horizon 100 \
 --wandb-project dlh-sweep1 --wandb-mode online"

echo "##### A0: tanh, std 1.0 (matched control) #####"
bash scripts/run_until.sh $COMMON --run-name s1_A0_tanh_std1   --dec-out-activation tanh    --dec-stddev 1.0
echo "##### A: sigmoid, std 1.0 #####"
bash scripts/run_until.sh $COMMON --run-name s1_A_sig_std1     --dec-out-activation sigmoid --dec-stddev 1.0
echo "##### B: sigmoid, std 0.5 #####"
bash scripts/run_until.sh $COMMON --run-name s1_B_sig_std05    --dec-out-activation sigmoid --dec-stddev 0.5
echo "##### C: sigmoid, std 0.3 #####"
bash scripts/run_until.sh $COMMON --run-name s1_C_sig_std03    --dec-out-activation sigmoid --dec-stddev 0.3
echo "##### SWEEP DONE #####"
