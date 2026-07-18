# Experiment control and ownership

**Policy date:** 2026-07-18 post-E6 terminal integration
**Launch authority:** `experiments/authorization.json`
**Principle:** exactly one execution owner; all other contributors are read-only with respect to
GPU launches and queue mutation.

## Current ownership boundary

The pre-Phase-C queue is complete and archived. The verified boundary is recorded in
`docs/HANDOFF_POST_C2_2026-07-12.md`; raw evidence is hash-pinned in
`artifacts/post_c2_evidence_manifest_2026-07-12.json` and independently backed up.

The **author retains scientific and GPU-launch authority**. Codex has time-bounded execution
autonomy only inside an author-approved tranche and the fail-closed authorization ledger. E7c,
E7a, E8a and E6 are complete, independently validated and frozen. The current workstream is M2
CPU/document/release QA; no GPU study is launchable.

## Authorized live work

| Study | Status | Meaning |
|---|---|---|
| seed-2 trio | completed | registered cells completed; evidence integrated and hash-pinned |
| c2 B8x60k | completed | matched-example changed-batch/update/noise/EMA control; evidence integrated |
| legacy Phase C | prohibited | endpoint semantics are invalid |
| untouched loopmom E2 | completed/frozen | one-time confirmation complete; forbidden for model or robustness selection |
| E4a/E4b | completed/frozen | five-seed factorial integrated; no seed 5+ |
| E5a | completed/not replicated | E5b/E5c closed by the frozen gate |
| E7c | completed/directionally robust | 35/35 frozen evaluations; 20/20 positive effects, 18/20 simultaneous lower bounds positive; 5.9976 GPU-hours; no extension |
| E7a | completed/directionally robust | seven trainings and 14 evaluations complete; no extension |
| E8a | completed/privileged reference | exact-pose raw-memory reference complete; no extension |
| E6 | completed/mixed boundary | six trainings/evaluations complete in 20.5127 GPU-hours; no adaptive arm, seed or endpoint |
| E9 | skipped/terminal | `SKIPPED_POST_FREEZE_NO_INCREMENTAL_UTILITY`; no implementation, budget or launch authorization; a future version requires a new scope |

## Handoff checklist

- [x] Final branch, commit and dirty-file list recorded.
- [x] Seed-2 trio completion attested in the preserved log.
- [x] Old controller and children absent.
- [x] c2 completion and exit 0 attested in preserved logs.
- [x] No project process/command line contains the Phase C launcher.
- [x] No project GPU process; unrelated GPU activity is out of scope.
- [x] Queue lock removed.
- [x] Checkpoints, NPZs, figures, tables and logs enumerated.
- [x] SHA-256 manifest generated and independent backup verified.
- [x] Resume/retry history recorded, including non-restored legacy RNG state and W&B reuse.
- [x] Current preregistrations attached.
- [x] Time-bounded delegation recorded below.

## Ownership record

| Interval | Owner | Authority |
|---|---|---|
| through seed-2+c2 completion | existing pre-C watcher/agent | monitor and finish only the authorized live sequence |
| after verified handoff | author | retains scientific and launch authority |
| approved execution tranches | Codex | time-bounded execution autonomy constrained by protocol, budget and authorization ledger |

**Current transfer marker:** the author authorized gated execution through terminal E6. That
experimental tranche is complete, E9 has failed its post-freeze utility gate, and M2 finalization
is active. Any future counterfactual study is a new programme requiring an explicit scope
decision, frozen implementation and its own fail-closed authorization.

## Status-change procedure

To make a blocked study launchable:

1. complete its required code, tests, protocol and markers;
2. update the claim ledger with the estimand and outcome-independent interpretation branches;
3. obtain explicit user approval for the GPU cost;
4. change only that experiment's status in `experiments/authorization.json`;
5. run `scripts/check_experiment_gate.py <experiment>` on CPU;
6. record the RunSpec hash and launch command before starting the process.

Results never retroactively authorize an experiment or alter its endpoint.
