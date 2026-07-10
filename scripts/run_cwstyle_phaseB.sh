#!/usr/bin/env bash
# CW-style table Phase B chain (pre-registered SPATIAL_FACTORIAL.md 2026-07-10 final block).
# LAUNCH ONLY AFTER R-G RESOLVES (flat-c2 trained + compared). One job at a time; chain markers
# to logs_rhcp/cwstyle_phaseB.log; every stop is resumable (atomic ckpts).
#
#   bash scripts/run_cwstyle_phaseB.sh 2>&1 | tee -a logs_rhcp/cwstyle_phaseB.log
set -e
PY=venv/Scripts/python.exe
LOG=logs_rhcp/cwstyle_phaseB.log
mark () { echo "=== $1 ($(date '+%F %T')) ==="; }

# ---------- 0. AE top branch (frozen trunk, ~15 min) + GO-gates + timing probes ----------
mark "AE-TOP train"
$PY -u -m hvp_rev.hier3 --train --base logs_rhcp/hier_ae_mom.pt \
    --data data/loopmom_train_3000x100.npy --out logs_rhcp/hier_ae_mom_top.pt
mark "GO-GATES (oracle buffer/stride phasing; PASS required before training)"
gate () {  # run a GO-gate and STOP the chain unless it prints PASS
  out=$($PY -u -m hvp_rev.rhvd --go_gate --arm "$1" --no_wandb 2>&1); echo "$out"
  echo "$out" | grep -q -- "-> PASS" || { echo "GO-GATE $1 FAILED -- chain stopped"; exit 1; }
}
gate hier-s2-min
gate strided-flat
gate hier3
mark "TIMING PROBES (30 steps each)"
$PY -u -m hvp_rev.rhvd --probe_steps 30 --arm hier-s2-min  --bf16 --no_wandb
$PY -u -m hvp_rev.rhvd --probe_steps 30 --arm strided-flat --bf16 --no_wandb
$PY -u -m hvp_rev.rhvd --probe_steps 30 --arm hier3        --bf16 --no_wandb
mark "PHASE_B_GATES_DONE"

# ---------- B1: stride-2 family edge-law shot @512 tok (partner = existing flat-short) ----------
mark "B1 hier-s2-min train+eval"
$PY -u -m hvp_rev.rhvd --arm hier-s2-min --steps 30000 --batch 16 --bf16 --seed 0
mark "B1 compare (vs existing flat-short s0)"
$PY -u -m hvp_rev.rhvd --compare flat-short,hier-s2-min --seed 0
mark "B1_DONE  # if CI-clean hier win: queue seed 1 (flat-short s1 exists) before headline use"

# ---------- B2: temporal-only span control @1088 tok vs the existing 1152 pair ----------
mark "B2 strided-flat train+eval"
$PY -u -m hvp_rev.rhvd --arm strided-flat --steps 30000 --batch 16 --bf16 --seed 0
mark "B2 compares (vs existing hier-long + flat-budget s0)"
$PY -u -m hvp_rev.rhvd --compare hier-long,strided-flat --seed 0
$PY -u -m hvp_rev.rhvd --compare flat-budget,strided-flat --seed 0
mark "B2_DONE"

# ---------- B3: depth-3 @1152 tok (2x2/4x4/8x8) vs hier-long + flat-budget ----------
mark "B3 hier3 train+eval"
$PY -u -m hvp_rev.rhvd --arm hier3 --steps 30000 --batch 16 --bf16 --seed 0
mark "B3 compares"
$PY -u -m hvp_rev.rhvd --compare hier-long,hier3 --seed 0
$PY -u -m hvp_rev.rhvd --compare flat-budget,hier3 --seed 0
mark "B3_DONE"

# ---------- A2: level-blinding probes (eval-only, existing + new ckpts) ----------
mark "A2 blinding probes"
$PY -u -m hvp_rev.rhvd --arm hier-long --eval_only --steps 30000 --seed 0 --blind coarse --no_wandb
$PY -u -m hvp_rev.rhvd --arm hier-long --eval_only --steps 30000 --seed 0 --blind fine   --no_wandb
$PY -u -m hvp_rev.rhvd --arm hier3     --eval_only --steps 30000 --seed 0 --blind top    --no_wandb
mark "A2_DONE"

# ---------- regenerate the table + curves with the new cells ----------
$PY scripts/make_cwstyle_table.py
mark "CWSTYLE_PHASE_B_ALL_DONE"
