# Verified post-c2 handoff

**Handoff supplied:** 2026-07-12 15:05 Europe/Kyiv

**Independent verification:** 2026-07-12 15:07–15:15 Europe/Kyiv

**Branch:** `research/ebt-hierarchical-video`

**HEAD at verification:** `13347d372b0c283613e5c1cdba65851f50ce2fc3`

## Authority and allowed next action

The author retains scientific and GPU-launch authority and has granted Codex time-bounded execution
autonomy under the approved handoff and experiment-order plans. The next allowed action is H1
evidence preservation/integration. No experiment is authorized merely by this handoff; each launch
still requires its dated protocol, code/data/statistical gates, RunSpec/identity record, W&B check
and explicit ledger status.

## Process, GPU and queue boundary

- No project trainer, queue controller, c2 process, Phase C process or orphaned project process was
  present. The verification PowerShell process itself was the only command line matching the
  project path.
- `logs_rhcp/newvision_queue.lock` is absent.
- `stop_before_phasec.log` records termination of the original controller tree before c2/Phase C.
- The c2-only controller started at `2026-07-11 23:22:31.095 +03:00` and exited code 0 at
  `2026-07-12 13:35:55.357 +03:00`.
- Measured project-controller ownership for c2 was approximately **14 h 13 min 24 s = 14.223 h**,
  including both arms, rollout evaluation and paired comparison. This replaces the stale 4.5-hour
  planning allowance; the approximately 9.723-hour difference reduces a later conditional Pool-A
  allocation rather than increasing the 181-hour closure envelope.
- `nvidia-smi` showed desktop processes and 604 MiB/28% transient activity, but no project process.
  Global GPU idleness is not required; project-scoped ownership is absent.
- Phase C and the optional queue tail were not launched.

The shell scripts emit completion lines into logs rather than standalone marker files. Therefore
`S2TRIO_DONE` and `C2_60K_DONE` are attested by the preserved logs and matching artifact times, not
by independent marker files. H3 will replace this with hash-bound completion attestations.

## Git/worktree boundary

The pre-existing dirty files at verification were:

- modified `docs/CWSTYLE_TABLE.md`;
- modified `docs/figures/fig12_cwstyle_curves.png`;
- untracked `docs/REVIEW_RFT_REMEDIATION_2026-07-11.md`.

The author handoff assigns the two live result outputs to Codex for H1 result integration. The
independent review remains a separate review artifact and must not be silently folded into a result
commit. No unrelated dirty file is to be staged by a broad Git command.

## c2 artifact verification

Verified evaluation artifacts:

- `logs_rhcp/rhvd_eval_hier-c2_s0_60k.npz`;
- `logs_rhcp/rhvd_eval_flat-c2_s0_60k.npz`.

Both contain 15,200 finite records and the same schema:

`h, hard, fg, sharp, pers_fg, clip, dt, seen`.

The paired identity fields `clip`, `h`, `hard`, `dt` and `seen` are exactly aligned. `pers_fg` is a
model outcome and correctly differs. Both 60k checkpoints load on CPU, report step 60,000, and
contain `model`, `opt`, `ema`, and `step`.

The legacy checkpoints do **not** contain Python/NumPy/Torch/CUDA RNG or sampler state. They are
valid completed runs but are not trajectory-reproducible resumes.

Independent recomputation with 10,000 clip-cluster bootstrap replicates gives:

- flat mean foreground MSE: `0.17077144`;
- hierarchical mean foreground MSE: `0.17849931`;
- hierarchy advantage convention (flat minus hierarchy): `-0.00772787`;
- 95% interval: `[-0.01233090, -0.00304468]`;
- records/clusters: `15,200 / 100`.

This matches the logged `-0.0077 [-0.0123, -0.0030]`: flat remains better at the tested 2,432-token
B8×60k regime, while the magnitude is smaller than under B16×30k.

## W&B caveat

Both B8×60k arms trained freshly, but reused the 30k W&B run IDs. W&B rejected or obscured early
steps below the previous monotonic step. Local logs/checkpoints/NPZs are complete; cloud provenance
is incomplete. All future runs must use new immutable identities, log the full resolved config and
artifacts, and pass an online connectivity check before launch.

## H0 verdict

**PASS with documented legacy caveats.**

The queue boundary is clean, Phase C is absent, paired c2 evidence is readable and internally
consistent, the worktree boundary is recorded, and author delegation is explicit. Proceed to H1
only. Artifact hashes and the immutable evidence manifest are H1 deliverables.
