#!/usr/bin/env bash
# Seed-1 replications for claim-bearing CW-style-table cells (binding two-seed rule +
# pre-B2 amendment). LAUNCH ONLY AFTER CWSTYLE_PHASE_B_ALL_DONE (chain must be idle).
#
#   bash scripts/run_cwstyle_seed1.sh 2>&1 | tee -a logs_rhcp/cwstyle_seed1.log
#
# S1-a (UNCONDITIONAL — B1 landed a CI-clean every-cell win at s0): hier-s2-min seed 1.
#      flat-short seed-1 partner npz already exists -> compare only needs the hier arm.
# S1-b (NOW REQUIRED — B2 measured 2026-07-10 19:08: strided-flat BEAT hier-long +0.0197
#      CI-clean at s0; per the amendment the positive claim needs seed-1 -> block active).
set -e
PY=venv/Scripts/python.exe
mark () { echo "=== $1 ($(date '+%F %T')) ==="; }

mark "S1-a hier-s2-min seed 1 train+eval"
$PY -u -m hvp_rev.rhvd --arm hier-s2-min --steps 30000 --batch 16 --bf16 --seed 1
mark "S1-a compare (vs existing flat-short s1)"
$PY -u -m hvp_rev.rhvd --compare flat-short,hier-s2-min --seed 1
mark "S1A_DONE"

mark "S1-b strided-flat seed 1 train+eval (REQUIRED per B2 outcome + amendment)"
$PY -u -m hvp_rev.rhvd --arm strided-flat --steps 30000 --batch 16 --bf16 --seed 1
mark "S1-b compares (vs existing hier-long s1 + flat-budget s1)"
$PY -u -m hvp_rev.rhvd --compare hier-long,strided-flat --seed 1
$PY -u -m hvp_rev.rhvd --compare flat-budget,strided-flat --seed 1
mark "S1B_DONE"

$PY scripts/make_cwstyle_table.py
mark "CWSTYLE_SEED1_DONE  # then: add s1 point to fig13, escalate CH1/CH8 clauses if confirmed"
