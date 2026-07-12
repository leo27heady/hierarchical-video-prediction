# Experiment control and ownership

**Policy date:** 2026-07-12 post-c2 handoff
**Launch authority:** `experiments/authorization.json`
**Principle:** exactly one execution owner; all other contributors are read-only with respect to
GPU launches and queue mutation.

## Current ownership boundary

The pre-Phase-C queue is complete and archived. The verified boundary is recorded in
`docs/HANDOFF_POST_C2_2026-07-12.md`; raw evidence is hash-pinned in
`artifacts/post_c2_evidence_manifest_2026-07-12.json` and independently backed up.

The **author retains scientific and GPU-launch authority**. Codex has time-bounded execution
autonomy only inside an author-approved tranche and the fail-closed authorization ledger. No
handoff grants blanket permission for the experiment portfolio. The current allowed work is
evidence/text integration and CPU-only preparation of the untouched confirmation path.

## Authorized live work

| Study | Status | Meaning |
|---|---|---|
| seed-2 trio | completed | registered cells completed; evidence integrated and hash-pinned |
| c2 B8x60k | completed | matched-example changed-batch/update/noise/EMA control; evidence integrated |
| legacy Phase C | prohibited | endpoint semantics are invalid |
| untouched loopmom E2 | preparation only | H3a/H4 CPU gates and frozen analysis required before launch authorization |
| all later studies | blocked or conditional | require claim trigger, budget allocation, H3b for training, ledger change and gates |

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

**Transfer marker:** verified 2026-07-12 in `docs/HANDOFF_POST_C2_2026-07-12.md`. The next allowed
action is CPU-only H3a/H4 preparation; no GPU experiment is yet launchable.

## Status-change procedure

To make a blocked study launchable:

1. complete its required code, tests, protocol and markers;
2. update the claim ledger with the estimand and outcome-independent interpretation branches;
3. obtain explicit user approval for the GPU cost;
4. change only that experiment's status in `experiments/authorization.json`;
5. run `scripts/check_experiment_gate.py <experiment>` on CPU;
6. record the RunSpec hash and launch command before starting the process.

Results never retroactively authorize an experiment or alter its endpoint.
