#!/usr/bin/env bash
# Article-critical optimization-budget control for the apparent 30k upper frontier edge.
# R-C showed that the hierarchical 1152-token arm was materially under-converged at 30k
# while its flat partner was saturated. This pair asks whether the seed-0 c2 loss at 2432
# tokens survives under the same B8x60k regime. The _60k tag preserves canonical 30k files.
set -euo pipefail
PY=venv/Scripts/python.exe
mark () { echo "=== $1 ($(date '+%F %T')) ==="; }

mark "C2-60K hier-c2 seed 0 (B8x60k; tagged eval)"
$PY -u -m hvp_rev.rhvd --arm hier-c2 --steps 60000 --batch 8 --bf16 --seed 0 \
    --eval_tag _60k
mark "C2-60K flat-c2 seed 0 (B8x60k; tagged eval)"
$PY -u -m hvp_rev.rhvd --arm flat-c2 --steps 60000 --batch 8 --bf16 --seed 0 \
    --eval_tag _60k
mark "C2-60K paired comparison"
$PY -u -m hvp_rev.rhvd --compare flat-c2,hier-c2 --seed 0 --eval_tag _60k
mark "C2_60K_DONE"
