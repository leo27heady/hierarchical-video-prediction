#!/usr/bin/env bash
# PHASE C — standard-MMNIST long-horizon control column (pre-registered; METRIC MANDATE active).
# LAUNCH AFTER `run_cwstyle_regtwins.sh core` completes and BEFORE the optional depth/E-320 tail.
# Three training seeds are unconditional: both a null and a win require replication. ~28-32 h.
#
#   bash scripts/run_phaseC_mmnist.sh 2>&1 | tee -a logs_rhcp/phasec.log
set -e
PY=venv/Scripts/python.exe
mark () { echo "=== $1 ($(date '+%F %T')) ==="; }

# ---------- C0: data (CPU) + zero-action files + fresh AE + digit classifier ----------
mark "C0 data gen (train 3000x100, test 500x348; OpenSTL-matched generator)"
[ -f data/mmnistL_train_3000x100.npy ] || $PY scripts/gen_moving.py --src mnist --frames 100 \
    --n 3000 --out data/mmnistL_train_3000x100.npy
[ -f data/mmnistL_test_500x348.npy ] || $PY scripts/gen_moving.py --src mnist --frames 348 \
    --n 500 --out data/mmnistL_test_500x348.npy
$PY - << 'EOF'
import numpy as np, os
for p in ("data/mmnistL_train_3000x100.npy", "data/mmnistL_test_500x348.npy"):
    a = p.replace(".npy", "_actions.npy")
    if not os.path.exists(a):
        clips = np.load(p, mmap_mode="r")
        np.save(a, np.zeros(clips.shape[:2], np.int8))   # zero-action conditioning, disclosed
        print("wrote", a)
EOF
mark "C0 fresh AE (standing rule: fresh AE per dataset)"
[ -f logs_rhcp/hier_ae_mmnistL.pt ] || $PY -u -m hvp_rev.hier_convae --train \
    --data data/mmnistL_train_3000x100.npy --out logs_rhcp/hier_ae_mmnistL.pt
mark "C0 digit classifier (METRIC MANDATE)"
[ -f logs_rhcp/mmnist_digit_clf.pt ] || $PY -u -m hvp_rev.mmnist_metrics --train_classifier
mark "C0_DONE"

# ---------- C1/C2: replicated 1152-token pair (distinct arm names -> distinct ckpts) ----------
for SEED in 0 1 2; do
  for ARM in mmL-flat mmL-hier; do
    mark "PHASEC $ARM seed $SEED train+eval (B16x30k, WARM=48, 300-frame rollouts)"
    $PY -u -m hvp_rev.rhvd --arm $ARM --steps 30000 --batch 16 --bf16 --seed "$SEED" \
        --train_path data/mmnistL_train_3000x100.npy --test_path data/mmnistL_test_500x348.npy \
        --ae_path logs_rhcp/hier_ae_mmnistL.pt --warm 48
  done
  mark "PHASEC seed $SEED compare (fg-MSE, paired cluster CIs)"
  $PY -u -m hvp_rev.rhvd --compare mmL-flat,mmL-hier --seed "$SEED" \
      --test_path data/mmnistL_test_500x348.npy --warm 48
  mark "PHASEC seed $SEED factor metrics (identity/trajectory/count)"
  for ARM in mmL-flat mmL-hier; do
    $PY -u -m hvp_rev.phasec_eval --arm "$ARM" --seed "$SEED" \
        --train_path data/mmnistL_train_3000x100.npy --test_path data/mmnistL_test_500x348.npy \
        --ae_path logs_rhcp/hier_ae_mmnistL.pt
  done
done
mark "PHASEC replicated equivalence summary (seed-visible; no frame pooling across seeds)"
$PY -u -m hvp_rev.phasec_summary --seeds 0,1,2
mark "PHASEC_ALL_DONE  # then: fig15 + result-contingent text swaps"
