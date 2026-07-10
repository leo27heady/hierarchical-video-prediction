#!/usr/bin/env bash
# REG TWINS + S2-TRIO + 320-token edge cell (pre-registered SPATIAL_FACTORIAL.md 2026-07-10).
# LAUNCH AFTER run_cwstyle_seed1.sh COMPLETES. One job at a time.
#
# Modes:
#   core      article-critical/head-control work only (run before Phase C)
#   optional  depth + third-edge probes only (run after Phase C)
#   all       both groups, retained for convenience; the publication-priority queue uses
#             core -> Phase C -> optional.
#
# ORDER (reprioritized 2026-07-10 late, external-review round 5, BEFORE launch): article-critical
# evidence first so an interruption never delays the core paper —
#   1. strided-flat-reg s0+s1 (ESSENTIAL: replicated B2 head axis — L4's open half)
#   2. 512 reg pair       (family-edge head control)
#   3. S2-TRIO            (ESSENTIAL: seed-level stability of the article-critical 1152 trio)
#   4. hier3-reg          (non-core depth probe)
#   5. E-320              (non-core third edge-law point; GO-gate travels with it)
#
#   bash scripts/run_cwstyle_regtwins.sh core 2>&1 | tee -a logs_rhcp/cwstyle_regtwins.log
#   bash scripts/run_cwstyle_regtwins.sh optional 2>&1 | tee -a logs_rhcp/cwstyle_optional.log
set -e
MODE=${1:-all}
case "$MODE" in
  core|optional|all) ;;
  *) echo "usage: $0 [core|optional|all]" >&2; exit 2 ;;
esac
PY=venv/Scripts/python.exe
mark () { echo "=== $1 ($(date '+%F %T')) ==="; }
gate () {
  out=$($PY -u -m hvp_rev.rhvd --go_gate --arm "$1" --no_wandb 2>&1); echo "$out"
  echo "$out" | grep -q -- "-> PASS" || { echo "GO-GATE $1 FAILED -- chain stopped"; exit 1; }
}

if [[ "$MODE" == "core" || "$MODE" == "all" ]]; then
# ---------- 1. ESSENTIAL: the B2 head axis, replicated unconditionally ----------
for SEED in 0 1; do
  mark "REG-1 strided-flat-reg seed $SEED train+eval (1088 tok)"
  $PY -u -m hvp_rev.rhvd --arm strided-flat-reg --steps 30000 --batch 16 --bf16 --seed "$SEED"
  mark "REG-1 seed $SEED compares (vs same-seed reg-long + reg-flat)"
  $PY -u -m hvp_rev.rhvd --compare reg-long,strided-flat-reg --seed "$SEED"
  $PY -u -m hvp_rev.rhvd --compare reg-flat,strided-flat-reg --seed "$SEED"
done
mark "REG1_DONE  # L4 head-axis result now carries two training seeds regardless of sign"
$PY scripts/make_cwstyle_table.py

# ---------- 2. family-edge head control (512 reg pair) ----------
mark "REG-2 flat-short-reg train+eval (512 tok reg partner)"
$PY -u -m hvp_rev.rhvd --arm flat-short-reg --steps 30000 --batch 16 --bf16 --seed 0
mark "REG-2 hier-s2-min-reg train+eval"
$PY -u -m hvp_rev.rhvd --arm hier-s2-min-reg --steps 30000 --batch 16 --bf16 --seed 0
mark "REG-2 compare"
$PY -u -m hvp_rev.rhvd --compare flat-short-reg,hier-s2-min-reg --seed 0
mark "REG2_DONE"
$PY scripts/make_cwstyle_table.py

# ---------- 3. ESSENTIAL: S2-TRIO (pre-registered UNCONDITIONALLY before any S1 result) ----------
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
mark "S2TRIO_DONE  # report seed-level effects individually; never pool frames across seeds"
$PY scripts/make_cwstyle_table.py
mark "CWSTYLE_CORE_ALL_DONE  # publication-priority handoff: run Phase C next"
fi

if [[ "$MODE" == "optional" || "$MODE" == "all" ]]; then
# ---------- 4. non-core: depth probe under regression ----------
mark "REG-4 hier3-reg train+eval (1152 tok; --ae_top default path)"
$PY -u -m hvp_rev.rhvd --arm hier3-reg --steps 30000 --batch 16 --bf16 --seed 0
mark "REG-4 compares"
$PY -u -m hvp_rev.rhvd --compare reg-long,hier3-reg --seed 0
$PY -u -m hvp_rev.rhvd --compare reg-flat,hier3-reg --seed 0
mark "REG4_DONE"

# ---------- 5. non-core: E-320 third edge-law point ----------
mark "E-320 GO-gate hier-s4-min (first stride-4 coarse rollout on loopmom)"
gate hier-s4-min
mark "E-320 hier-s4-min train+eval (320 tok)"
$PY -u -m hvp_rev.rhvd --arm hier-s4-min --steps 30000 --batch 16 --bf16 --seed 0
mark "E-320 flat-320 train+eval (320 tok partner)"
$PY -u -m hvp_rev.rhvd --arm flat-320 --steps 30000 --batch 16 --bf16 --seed 0
mark "E-320 compare"
$PY -u -m hvp_rev.rhvd --compare flat-320,hier-s4-min --seed 0
mark "E320_DONE"

$PY scripts/make_cwstyle_table.py
mark "CWSTYLE_OPTIONAL_ALL_DONE"
fi
