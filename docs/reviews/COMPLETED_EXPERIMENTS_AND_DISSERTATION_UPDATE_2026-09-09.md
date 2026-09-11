# Completed experiments and dissertation update — 2026-09-09

## Outcome

The primary A4 campaign completed all 86 runs. Its checkpoint hashes, final metrics, selected roles and held-out arrays were audited before manuscript insertion. Completed structural results, diagnostic results, CUDA parity and short training screens are now reported in Chapter 4 and reflected in the conclusions and appendix registry. Final integrated quality and valid resource measurements remain pending. No successful outcome was invented for unfinished experiments.

## Launcher error, repair and current execution

The pasted command reached the final confirmation preflight after successfully completing data preparation, profiling, four 200-update screens and VAL300. It stopped because the old idle check rejected approximately 1.7 GiB of ordinary desktop GPU memory against a 1.5 GiB cap. This was a launcher failure, not a failed training result.

The repaired launcher allows the larger of 3 GiB or 20% of GPU capacity, still requires utilization below 20%, explicitly rejects Python CUDA processes, and requires three stable idle samples. It waits up to three minutes for transient load. The outer queue and nested confirmation controller use the same check. Training code and experiment configurations were not changed by this repair.

The authorized queue was resumed through `run_remaining_experiments.py run --execute --wait`. Original completed artifacts and source snapshots were preserved; the repaired HVP controller runs from `source/hvp_launchfix_20260909`. Do not start another copy while this queue is running.

At verification, INT_MM_S00 had passed 6,150 updates with finite loss, nonzero gradients, approximately 6.7 updates/s and about 1.03 GB peak allocated CUDA memory. Parent PID 12372 and training PID 25044 were alive. These are progress checks, not final quality evidence.

Logging is present at all relevant levels:

- Queue heartbeat every 30 seconds: `D:/hierarchical-video-prediction-storage/dissertation_remaining_2026-09-07/logs/relaunch_20260909.stdout.log`.
- Queue stage and active command: `queue_status.json`; detailed controller output: `logs/confirmations.log`.
- Each run: `confirmations/<run>/metrics.jsonl`, `stdout.log`, `stderr.log`, `status.json`, resolved configuration, provenance, checkpoints, panels and W&B directory. The controller requests online W&B; local logs remain the independent record.

## Resource profiling finding

The original profiling diagnostic returned the original tensor from a saved-tensor hook. This retained an autograd graph and inflated allocator baselines across successive modes. Consequently, the original full-model resource peaks and timings are excluded from comparative claims. Original files remain available for audit.

The hook now returns a detached tensor, with cleanup and an allocator-retention guard before measurements. A CPU weak-reference regression test demonstrates that the output graph is released. Corrected profiles use schema 3 and record the guard. CUDA parity from the original profiles remains usable because it ran before the leaking diagnostic; the manuscript reports its tolerances and pass status, explicitly noting that maximum discrepancies were not recorded.

A separate finite controller, PID 23584, waits for all four confirmations to complete, then performs FP32 and BF16 profiling on both datasets. It writes to `profiles_corrected_20260909` and preserves the original profiles. Its status was `waiting_for_confirmations`. If the confirmation parent exits without completion, it records a failure rather than profiling alongside an unresolved queue. GPU validation of the repaired profiler is therefore still pending.

## Evidence and substantive results

The audit checked 86 checkpoint SHA-256 values, terminal 30,000-update metrics against raw logs, campaign completion, role selection and confirmation contracts. Eight confirmation arrays contain 500 distinct aligned clips at four horizons; six VAL300 arrays contain 300 clips. All audited metric arrays were finite. Each short screen completed exactly 200 updates without evaluating the test set.

- Geometric allocation improves development averages, but its held-out advantage is seed-dependent: h=4 flat-minus-geometric differences are 0.00323 and -0.00035. Only the first corrected interval excludes zero.
- STS K16 with geometric temporal spacing gives held-out h=4 foreground MSE 0.02500 and 0.02381 versus flat 0.03669 and 0.03406. Both paired corrected intervals favor STS. Both policies use B=1792; the selected resource role is not evidence of fewer tokens.
- Additional context helps the endpoint average but intermediate changes are not monotonic. Three levels did not beat the two-level reference under this budget. Nested overlap has the best two-seed mean among the tested overlap schemes. These findings do not support a blanket claim that more hierarchy always helps.
- Re-entry diagnostics comprise 76 clip–horizon records, not 76 independent clips. Attention summaries use a fixed diagnostic window, not all 500 clips. Neither establishes a causal mechanism.
- The four short integrated screens establish execution and trainability only. Their validation losses do not rank final predictive quality.

Four descriptive held-out contrasts use 10,000 paired clip-bootstrap repetitions with Bonferroni correction. Their intervals describe clip sampling within each seed, not uncertainty over independently trained models. They are distinguished from the previously registered E2–E8 maximum-statistic procedure. Development and VAL300 aggregation are also distinguished.

Reproduction and source copies are in `completed_evidence_2026-09-09/audit.json` and accompanying files. The audit and integration scripts are `scripts/audit_completed_evidence_20260909.py` and `scripts/integrate_completed_evidence_20260909.py`. The latter is a dated migration from its saved pre-edit sources, not a general-purpose future-results updater; do not replay it over subsequent editorial changes.

## Manuscript integration and flow review

Tables 4.7–4.14, 4.18 and 4.20 now contain verified measurements. Figures 4.13 and 4.14 now show structural comparisons and held-out horizon curves/paired intervals. Interpretation follows each experimental question: allocation and context, temporal sampling, levels and overlap, independent confirmation, then integrated predictor correctness and quality. The conclusions report mixed findings and separate numerical parity from quality and memory claims.

The figure generator preserves these measured figures; provenance and evidence registry were synchronized. The appendix now distinguishes verified, running and conditional evidence. The static/adaptive controls' shared-feature explanation was qualified because the flat package also changes its feature composition. Display formulas were preserved and checked against the prior manuscript. The audit does not claim an independent derivation of every historical equation or revalidation of every historical experiment.

Coverage is 144 verified detailed evidence slots and 166 pending slots out of 310. Slot counts are measurements, not numbers of experiments. The compact manuscript retains 28 pending-result cells and the 11 known administrative fields. Pending results remain in Tables 4.15, 4.19, 4.21 and 4.22 and their corresponding figures; conditional experiments remain explicitly conditional.

## Remaining execution matrix

| Run / set | Purpose | Status | Estimated duration |
|---|---|---|---|
| INT_MM_S00 | Full adaptive Moving MNIST, seed 0 | Running | About 14.2 h from start |
| INT_MM_S01 | Independent Moving MNIST replication | Queued | About 14.2 h |
| INT_TB_S00 | Full adaptive TaxiBJ, seed 0 | Queued | About 3.5 h |
| INT_TB_S01 | Independent TaxiBJ replication | Queued | About 3.5 h |
| Corrected RESOURCE | Both datasets × FP32/BF16; packing and core modes | Waiting for the four runs | Four profiles, each capped at 22.5 min; at most 90 min of subprocess time |
| ROLL64 | Conditional long recursive rollout diagnostic without retraining | Not launched | Not measured; retain only if a recursive-horizon claim is needed |
| COMP | Conditional matched comparator for a stronger final-quality claim | Not launched | Not measured; scope depends on the claim and observed full-run results |

The four training estimates total approximately 35.5 hours from queue start, based on the completed screens with a 20% allowance and validation/test proxy. They are estimates, not deadlines; later epochs and evaluation may differ. Corrected profiling follows them. No speculative additional training was added.

## Verification and limitations

The launcher/controller tests passed (17 tests), and the profiler tests passed (3 tests). These CPU checks ran without competing for CUDA. Actual resumed training supplies a separate live execution check. The exact final manuscript checks, source hashes and rendered-PDF audit are recorded in `completed_evidence_2026-09-09/verification/`.

The PDF was rebuilt and all pages rendered for automated bounds, missing-glyph and near-blank-page inspection. Revised results pages were visually reviewed for table headers, figure readability, caption attachment and flow. The final release verifier checks manuscript consistency, narrative coverage, regression tests, whitespace, unchanged display math, source/build agreement and PDF/render identity.

Final outcome: all seven release checks passed; 301 pages, 176 bookmarks and 193 links. No outside-page text blocks, near-blank pages or replacement characters were detected. PDF SHA-256: `3b59d3ce181824352feb78068e0cec33746753dfe705009f5c06fec1abfc28fa`. Generated SVG trailing whitespace was normalized after export; byte-for-byte comparison of their rendered PNGs confirmed identical figure pixels.

This is a results-integrated review version, not a fully completed submission: integrated quality, corrected resource results, conditional claims and administrative fields remain unresolved. No commits or pushes were made, and the completed A4 worktree remains clean.
