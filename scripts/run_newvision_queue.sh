#!/usr/bin/env bash
# Single-owner post-Phase-B queue for the packaging-reframe article/dissertation programme.
# Never run concurrently with run_cwstyle_phaseB.sh or another agent's launch command.
#
# Modes:
#   full         seed replications -> core shape/head work -> c2 60k -> Phase C -> optional probes
#   after-seed1  same, but assumes run_cwstyle_seed1.sh has already completed
#   core-only    seed replications -> core shape/head work -> Phase C; leaves optional probes
#   after-seed1-core  core shape/head work -> Phase C; seed work already done, optional deferred
#
# Examples:
#   bash scripts/run_newvision_queue.sh full 2>&1 | tee -a logs_rhcp/newvision_queue.log
#   bash scripts/run_newvision_queue.sh after-seed1 2>&1 | tee -a logs_rhcp/newvision_queue.log
set -euo pipefail
MODE=${1:-full}
case "$MODE" in
  full|after-seed1|core-only|after-seed1-core) ;;
  *) echo "usage: $0 [full|after-seed1|core-only|after-seed1-core]" >&2; exit 2 ;;
esac

PHASEB_LOG=logs_rhcp/cwstyle_phaseB.log
if ! grep -q "CWSTYLE_PHASE_B_ALL_DONE" "$PHASEB_LOG"; then
  echo "Phase B is not marked complete in $PHASEB_LOG; refusing to overlap the live queue." >&2
  exit 3
fi

LOCK=logs_rhcp/newvision_queue.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Queue lock exists at $LOCK; another controller may be active. Refusing to launch." >&2
  exit 4
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

mark () { echo "=== $1 ($(date '+%F %T')) ==="; }
run_logged () {
  local log=$1; shift
  "$@" 2>&1 | tee -a "$log"
}

if [[ "$MODE" == "full" || "$MODE" == "core-only" ]]; then
  mark "QUEUE 1/4 seed replications"
  run_logged logs_rhcp/cwstyle_seed1.log bash scripts/run_cwstyle_seed1.sh
fi

mark "QUEUE 2/5 article-critical shape/head work"
run_logged logs_rhcp/cwstyle_regtwins.log bash scripts/run_cwstyle_regtwins.sh core

mark "QUEUE 3/5 article-critical upper-edge optimization control"
run_logged logs_rhcp/frontier_c2_60k.log bash scripts/run_frontier_c2_60k.sh

mark "QUEUE 4/5 replicated Phase C boundary test"
run_logged logs_rhcp/phasec.log bash scripts/run_phaseC_mmnist.sh

if [[ "$MODE" == "full" || "$MODE" == "after-seed1" ]]; then
  mark "QUEUE 5/5 optional depth/third-edge probes"
  run_logged logs_rhcp/cwstyle_optional.log bash scripts/run_cwstyle_regtwins.sh optional
fi

mark "NEWVISION_QUEUE_ALL_DONE"
