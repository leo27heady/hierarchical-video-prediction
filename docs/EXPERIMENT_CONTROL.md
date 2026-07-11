# Experiment control and ownership

**Policy date:** 2026-07-11
**Launch authority:** `experiments/authorization.json`
**Principle:** exactly one execution owner; all other contributors are read-only with respect to
GPU launches and queue mutation.

## Current ownership boundary

The pre-Phase-C queue predates the remediation plan and remains under its existing watcher until
the following sequence is complete:

1. finish the running seed-2 trio;
2. terminate the old master queue before Phase C;
3. run c2 B8x60k alone;
4. verify the GPU is idle, Phase C is absent, the old controller is gone, and the stale lock is
   removed;
5. deliver logs, artifacts, hashes, resume history, commands, and worktree state.

No file used by that live queue may be semantically changed before the handoff.  In particular,
`hvp_rev/rhvd.py`, `scripts/run_newvision_queue.sh`, and `scripts/run_frontier_c2_60k.sh` remain the
legacy execution path until c2 completes.

After the handoff, the current remediation task becomes the sole experiment execution owner.  No
new GPU run is implied by the transfer: the first work is CPU-only validation and protocol freeze.

## Authorized live work

| Study | Status | Meaning |
|---|---|---|
| seed-2 trio | finish only | closes already-running registered cells without a code change |
| c2 B8x60k | run alone | matched-example batch/update/EMA-regime control |
| legacy Phase C | prohibited | endpoint semantics are invalid |
| all later studies | blocked or conditional | require ledger status change plus gates |

## Handoff checklist

- [ ] Final branch, commit and dirty-file list recorded.
- [ ] Seed-2 trio completion marker present.
- [ ] Old controller and children absent.
- [ ] c2 completion marker present, or failure explicitly documented.
- [ ] No process/command line contains the Phase C launcher.
- [ ] GPU idle.
- [ ] Queue lock removed.
- [ ] Checkpoints, NPZs, figures, tables and logs enumerated.
- [ ] SHA-256 manifest generated for new evidence.
- [ ] Resume/retry history recorded, including non-restored legacy RNG state.
- [ ] Current preregistrations attached.
- [ ] Explicit ownership transfer recorded below.

## Ownership record

| Interval | Owner | Authority |
|---|---|---|
| through seed-2+c2 completion | existing pre-C watcher/agent | monitor and finish only the authorized live sequence |
| after verified handoff | current remediation task | sole authority to propose launches; still constrained by the authorization ledger |

**Transfer marker:** pending c2 completion.  Until it is filled, no second controller may launch.

## Status-change procedure

To make a blocked study launchable:

1. complete its required code, tests, protocol and markers;
2. update the claim ledger with the estimand and outcome-independent interpretation branches;
3. obtain explicit user approval for the GPU cost;
4. change only that experiment's status in `experiments/authorization.json`;
5. run `scripts/check_experiment_gate.py <experiment>` on CPU;
6. record the RunSpec hash and launch command before starting the process.

Results never retroactively authorize an experiment or alter its endpoint.
