#!/usr/bin/env bash
# PHASE C — standard-MMNIST long-horizon control column (pre-registered; METRIC MANDATE active).
# LAUNCH AFTER `run_cwstyle_regtwins.sh core` completes and BEFORE the optional depth/E-320 tail.
# Three training seeds are unconditional: both a null and a win require replication. ~28-32 h.
#
#   bash scripts/run_phaseC_mmnist.sh 2>&1 | tee -a logs_rhcp/phasec.log
set -e
PY=venv/Scripts/python.exe
mark () { echo "=== $1 ($(date '+%F %T')) ==="; }

# Scientific safety gate (2026-07-11): the legacy identity/trajectory/coverage evaluator is
# invalid for Phase-C interpretation.  This fails closed before data generation or GPU work even
# if an obsolete master queue reaches this script.  Use run_phasec_v2.ps1 only after its CPU,
# classifier, protocol-freeze, authorization and post-c2 integration gates are satisfied.
$PY scripts/check_experiment_gate.py phase-c-legacy
exit 97

# ---------- C0: data (CPU) + zero-action files + fresh AE + digit classifier ----------
mark "C0 data gen (train 3000x100, test 500x348; OpenSTL-matched generator)"
[ -f data/mmnistL_train_3000x100.npy ] || $PY scripts/gen_moving.py --src mnist --frames 100 \
    --n 3000 --seed 1234 --out data/mmnistL_train_3000x100.npy
[ -f data/mmnistL_test_500x348.npy ] || $PY scripts/gen_moving.py --src mnist --frames 348 \
    --n 500 --seed 5678 --out data/mmnistL_test_500x348.npy
$PY - << 'EOF'
import hashlib
import numpy as np, os

train = np.load("data/mmnistL_train_3000x100.npy", mmap_mode="r")
test = np.load("data/mmnistL_test_500x348.npy", mmap_mode="r")
train_h0 = {hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest() for x in train[:, 0]}
test_h0 = {hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest() for x in test[:, 0]}
collisions = train_h0 & test_h0
assert not collisions, (
    f"Phase-C split collision: {len(collisions)} test frame-0 images occur in train. "
    "Delete mmnistL data and regenerate with distinct seeds."
)
print("Phase-C split audit PASS: no frame-0 hash collisions (train seed 1234; test seed 5678)")

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
