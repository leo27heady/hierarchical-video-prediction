#!/usr/bin/env bash
# REG TWINS + 320-token edge cell (pre-registered SPATIAL_FACTORIAL.md 2026-07-10 evening).
# LAUNCH AFTER S1-a (run_cwstyle_seed1.sh) COMPLETES. One job at a time.
#
#   bash scripts/run_cwstyle_regtwins.sh 2>&1 | tee -a logs_rhcp/cwstyle_regtwins.log
set -e
PY=venv/Scripts/python.exe
mark () { echo "=== $1 ($(date '+%F %T')) ==="; }
gate () {
  out=$($PY -u -m hvp_rev.rhvd --go_gate --arm "$1" --no_wandb 2>&1); echo "$out"
  echo "$out" | grep -q -- "-> PASS" || { echo "GO-GATE $1 FAILED -- chain stopped"; exit 1; }
}

# ---------- E-320: the third edge-law point (GO-gate first: first stride-4 coarse rollout on loopmom)
mark "E-320 GO-gate hier-s4-min"
gate hier-s4-min
mark "E-320 hier-s4-min train+eval (320 tok)"
$PY -u -m hvp_rev.rhvd --arm hier-s4-min --steps 30000 --batch 16 --bf16 --seed 0
mark "E-320 flat-320 train+eval (320 tok partner)"
$PY -u -m hvp_rev.rhvd --arm flat-320 --steps 30000 --batch 16 --bf16 --seed 0
mark "E-320 compare"
$PY -u -m hvp_rev.rhvd --compare flat-320,hier-s4-min --seed 0
mark "E320_DONE"

# ---------- REG twins: the head axis at the new shape cells
mark "REG flat-short-reg train+eval (512 tok reg partner)"
$PY -u -m hvp_rev.rhvd --arm flat-short-reg --steps 30000 --batch 16 --bf16 --seed 0
mark "REG hier-s2-min-reg train+eval"
$PY -u -m hvp_rev.rhvd --arm hier-s2-min-reg --steps 30000 --batch 16 --bf16 --seed 0
mark "REG compare 512 pair"
$PY -u -m hvp_rev.rhvd --compare flat-short-reg,hier-s2-min-reg --seed 0
mark "REG strided-flat-reg train+eval (1088 tok)"
$PY -u -m hvp_rev.rhvd --arm strided-flat-reg --steps 30000 --batch 16 --bf16 --seed 0
mark "REG compares strided-flat-reg (vs reg-long + reg-flat)"
$PY -u -m hvp_rev.rhvd --compare reg-long,strided-flat-reg --seed 0
$PY -u -m hvp_rev.rhvd --compare reg-flat,strided-flat-reg --seed 0
mark "REG hier3-reg train+eval (1152 tok; needs --ae_top, default path)"
$PY -u -m hvp_rev.rhvd --arm hier3-reg --steps 30000 --batch 16 --bf16 --seed 0
mark "REG compares hier3-reg"
$PY -u -m hvp_rev.rhvd --compare reg-long,hier3-reg --seed 0
$PY -u -m hvp_rev.rhvd --compare reg-flat,hier3-reg --seed 0
mark "REGTWINS_DONE"

$PY scripts/make_cwstyle_table.py
mark "CWSTYLE_REGTWINS_ALL_DONE"

# ---------- S2-TRIO (pre-registered UNCONDITIONALLY 2026-07-10 before any S1 result;
# ---------- third seed for the article-critical 1152 trio; ~9 GPU-h)
mark "S2-TRIO hier-long seed 2"
$PY -u -m hvp_rev.rhvd --arm hier-long --steps 30000 --batch 16 --bf16 --seed 2
mark "S2-TRIO flat-budget seed 2"
$PY -u -m hvp_rev.rhvd --arm flat-budget --steps 30000 --batch 16 --bf16 --seed 2
mark "S2-TRIO strided-flat seed 2"
$PY -u -m hvp_rev.rhvd --arm strided-flat --steps 30000 --batch 16 --bf16 --seed 2
mark "S2-TRIO compares (seed 2)"
$PY -u -m hvp_rev.rhvd --compare flat-budget,hier-long --seed 2
$PY -u -m hvp_rev.rhvd --compare hier-long,strided-flat --seed 2
$PY -u -m hvp_rev.rhvd --compare flat-budget,strided-flat --seed 2
$PY scripts/make_cwstyle_table.py
mark "S2TRIO_DONE"
