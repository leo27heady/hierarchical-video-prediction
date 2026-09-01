# Article 4 30k campaign: implementation, operations, and release record

**Decision owner:** Leonid Kulyk  
**Decision date:** 2026-09-01  
**Status:** implementation complete; verification and launch gate recorded below  
**Campaign:** `article4-primary-30k-2026-09-01`  
**W&B project:** `hierarchical-video-prediction-article4-v2`  
**W&B group:** `article4-30k-primary-2026-09-01`

This is the operating document for the clean Path B Article 4 campaign. It records the decisions,
the exact experiment inventory, how data and positional identities move through the model, what is
logged, how interruption recovery works, and the commands required to launch or resume the entire
campaign. The experiment matrix and decision register remain authoritative for scientific intent;
the campaign JSON and individual run JSON files are authoritative for executable identities.

## 1. Locked decisions

1. Every primary training run uses **30,000 optimizer updates**. The rejected 16k contingency is
   retained only for provenance. Validation occurs every 2k and is diagnostic; the primary result
   always comes from the 30k checkpoint.
2. The primary campaign uses exactly **two seeds, 0 and 1**, for every unique configuration.
3. TRB is the only Article 4 dataset. Long Moving MNIST is excluded.
4. A4-FRT uses budgets `320, 768, 1152, 1792, 2432, 3072` and only Flat, Linear, and Geometric
   allocation strategies. Balanced is not a frontier strategy.
5. A4-ALC uses only budget 1,792 and all seven agreed allocation policies. Flat, Linear, and
   Geometric reuse identical A4-FRT checkpoints; four additional policies are trained.
6. A4-CEX uses coarse-time counts `0, 4, 8, 16, 32, 64` with six complete recent fine times.
7. A4-LVL uses budget 1,792 and levels 1, 2, and 3. Level 1 and the selected level-2 allocation are
   reused. The one new level-3 cell is generated only after the best non-flat A4-ALC policy has
   been selected by the two-seed mean final validation metric.
8. A4-OVR compares nested, exclusive, and residual support at budget 1,792. Nested is reused. The
   base policy is the best A4-ALC policy among the overlap-feasible set: Fine-heavy, Balanced, and
   Coarse-heavy. This feasibility restriction is explicit even if another allocation wins A4-ALC.
9. A4-STS uses budget 1,792 only, fine-token reserve 256, spatial quotas `64, 32, 16, 8`, and
   temporal placements Recent, Uniform, and Geometric.
10. Machine run IDs remain verbose and unique. Dissertation prose uses short labels (`F`, `H-L`,
    `H-G`, `H-F`, `H-B`, `H-C`, `H-R`) with a separate configuration table.
11. Local files are the canonical scientific record. W&B is a comprehensive mirror and monitoring
    surface; a W&B outage cannot terminate a scientific run.
12. The fresh confirmation set is generated only after all development runs finish and a hashed
    development-only selection ledger has been written. Holdout scores never select or reselect a
    configuration.

## 2. Complete experiment inventory

| Family | Displayed cells | New unique configurations | Runs at two seeds | Purpose |
|---|---:|---:|---:|---|
| A4-FRT | 18 | 18 | 36 | quality-memory frontier over six budgets and three allocation laws |
| A4-ALC | 7 | 4 | 8 | allocation comparison at exactly 1,792 tokens |
| A4-CEX | 6 | 6 | 12 | marginal value of progressively longer coarse support |
| A4-LVL | 3 | 1 | 2 | one-, two-, and three-level hierarchy comparison |
| A4-OVR | 3 | 2 | 4 | nested, exclusive, and residual cross-level representations |
| A4-STS | 12 | 12 | 24 | spatial quota by temporal-placement surface |
| **Total** | **49** | **43** | **86** | primary training campaign |

Execution has two training waves:

- Wave 1 contains 80 fully specified runs: A4-FRT, the four A4-ALC additions, A4-CEX, and A4-STS.
- The dependency resolver reads only the final 30k validation values from both seeds, writes a
  hash-bound resolution ledger, and creates the six A4-LVL/A4-OVR Wave 2 run specifications.
- Wave 2 trains those six dependent runs.
- The post-training finalizer performs A4-MEC and sealed confirmation evaluation without new model
  fitting.

The executable inventory is
`experiments/v2/campaigns/article4_primary_30k_2026-09-01.json`; Wave 1 run specifications are in
`experiments/v2/runspecs/article4_primary_30k/`. Generated Wave 2 specifications and all results are
under ignored `artifacts/v2/article4_primary_30k/` so a resumed campaign cannot silently change the
tracked protocol.

## 3. Model and data flow

The worker performs the following flow for every run:

```text
hash-locked TRB manifest
        |
        v
frozen hierarchical AE -> train-only channel statistics -> standardized CPU latent store
        |                                                   |
        |                                                   v
        |                                     deterministic AllocationPlan
        |                                                   |
        |                                                   v
        +--------------------------------> CPU packed selected state
                                                            |
                                                            v
                                     only selected latents + IDs transferred to CUDA
                                                            |
                                                            v
                     level + physical time + spatial cell + role + action embeddings
                                                            |
                                                            v
                          one joint WindowDiT attention sequence and v-diffusion loss
                                                            |
                                                            v
                      EMA validation, decoded metrics, raw arrays, and image panels
```

Important implementation properties:

- The 3,000 development clips are split by the tracked manifest into 2,700 train and 300
  validation clips. Latent means and standard deviations are computed only from the training clip
  indices and then applied to both splits.
- Frames are encoded once per worker into a CPU latent store. The measured setup cost is about
  15--20 seconds, below 0.3% of a typical 30k run. Persisting a multi-gigabyte latent cache would
  add cache invalidation and storage risk for only about 20--30 minutes across the complete
  campaign, so it is intentionally not used.
- Fine state is gathered directly as `(batch, selected_tokens, latent_channels)`. The loader never
  creates a dense `batch x reach x channels x 8 x 8` candidate tensor on the GPU.
- Coarse and top tensors contain only selected time positions. Targets and future actions are
  invariant across allocations for the same sampling seed.
- The common first-target anchor is 96 and the temporal embedding table contains 100 positions.
  Context tokens carry their physical positions inside the shared 96-frame coordinate system;
  target horizons occupy positions 96--99. Shorter policies are not renumbered to look longer.
- Fine tokens retain physical 8x8 cell IDs and receive the existing learned fine spatial
  embeddings. Dynamic, content-dependent routing is not used in Article 4.

### Three levels

The 1,792-token level-3 cell uses 96 top positions at four tokens each (384 tokens). The remaining
1,408 tokens preserve the selected A4-ALC two-level coarse-to-fine payload ratio, with the coarse
payload rounded to complete 16-token time slots. The top stream is context-only, so all arms retain
the same four target frames and coarse/fine prediction objective. The level-3 model has 7,557,792
parameters versus 7,556,512 for the two-level model: the 1,280-parameter difference is the added
top level/spatial embedding and must be disclosed rather than called exact parameter parity.

### Overlap modes

- **Nested:** coarse and fine tokens may describe the same physical time; each keeps its own level
  and spatial identity.
- **Exclusive:** fine identities and token counts are unchanged, while coarse positions are moved
  to the newest physical ages not represented at fine level. The total stays exactly 1,792.
- **Residual:** token identities remain nested, but each packed fine content token is represented as
  `fine projection - same-time coarse-parent projection + fine spatial embedding`. The linear bias
  cancels exactly. This is a real representation change, not a renamed nested run.

## 4. Training protocol and performance choices

Every run uses width 256, depth 6, eight heads, batch 16, AdamW, fp32 master parameters, bf16 CUDA
autocast, EMA, and diffusion v-prediction. Q/K RMSNorm modules use a tested mixed-precision wrapper
that retains fp32 master weights while supplying dtype-compatible operands to the fused CUDA path.
Ordinary attention uses fused scaled-dot-product attention. Explicit attention matrices are formed
only for one batch-size-1 post-training diagnostic forward.

The A4-COR pair established 30k because early ordering reverses between 12k and 16k and neither arm
meets the registered late-curve rule at 16k. “Ranking reversal” is an observed crossing of validation
curves, not a code mechanism.

The measured training subtotal is 179--193 GPU-hours. Validation, profiling, A4-MEC, and sealed
confirmation bring the direct estimate to 187--205 GPU-hours. Reserve 205--234 GPU-hours including
recovery margin. On one continuously available RTX 4060 Ti this is about 7.8--8.5 direct days, or
8.5--9.8 days with reserve; therefore a strict seven-calendar-day completion cannot be promised at
30k on this single GPU.

## 5. Crash recovery and idempotence

The controller and workers are restart-safe:

- Each run has a deterministic ID derived from its experiment ID and configuration hash.
- The worker acquires a per-run lock and converts dead-process locks to timestamped stale-lock
  evidence. It refuses to run concurrently under an active PID.
- `rolling_resume.pt` is written atomically every 1,000 updates and includes model, EMA, optimizer,
  PyTorch CPU/CUDA RNG states, and the batch sampler RNG state.
- Training and validation JSONL files are truncated to the restored checkpoint before continuation;
  future panels and milestones from an incomplete attempt are also removed.
- Immutable compact milestones are saved at 10k, 20k, and 30k. The completed 30k milestone must
  exist before a run is marked complete.
- The controller skips completed config hashes. A failed child is retried twice by default. If it
  still fails, the campaign stops at an explicit incomplete state. Running the same command again
  continues from the last atomic checkpoint.
- At most 999 updates can be repeated after a hard interruption. Completed evidence is never
  overwritten by a different configuration.
- Post-training selection, confirmation generation, and evaluation also skip matching completed
  artifacts. Partial files from this campaign's known confirmation basename are regenerated; no
  unrelated files are touched.

## 6. Local and W&B evidence

Each run writes:

- canonical config, config hash, allocation hash, campaign hash, source commit, environment,
  hardware, model parameter count, expected sequence length, latent statistics, and byte accounting;
- training loss, pre-clipping gradient norm, learning rate, seconds/update, samples/second,
  context-tokens/second, attempt elapsed time and ETA every 200 updates;
- CUDA allocated, reserved, and peak allocated memory; W&B additionally captures GPU utilization,
  temperature, host CPU, RAM, network, and process telemetry;
- fixed-validation diffusion loss, latent MSE, decoded pixel MSE/MAE, foreground/background MSE,
  PSNR, global SSIM, and horizon-specific versions every 2,000 updates;
- a fixed validation panel every 2,000 updates with last context, prediction, target, and absolute
  error for every horizon; the raw panel arrays are retained as compressed NPZ;
- attempt history, status transitions, stdout/stderr logs, final checkpoint hash, and hashes of
  metric ledgers.

W&B uses deterministic resumable run IDs. If W&B initialization or logging fails, the local run
continues and `wandb_status.json` records the unsynced condition. After connectivity returns,
`scripts/v2/sync_article4_wandb.py` replays local train/validation histories, panels, and evidence
artifacts without retraining.

## 7. Post-training selection and confirmation

The finalizer first aggregates all 86 final development-validation records. It then locks:

- best Flat at 1,792 tokens;
- best non-flat A4-ALC hierarchy at 1,792 tokens;
- worst/failed A4-ALC hierarchy at 1,792 tokens for bounded mechanism explanation;
- smallest hierarchical stored state whose two-seed mean is within 5% of the matched Flat metric;
  if none qualifies, the ledger explicitly falls back to the best hierarchy and does not claim a
  memory-Pareto win.

Only after writing and hashing that ledger does it generate a fresh 500-clip, 200-frame TRB set
with seed 26090141. It verifies artifact hashes, schema, action-position dynamics, empirical action
probabilities, uniqueness, and zero exact-clip collision with the development data. Every selected
seed checkpoint is evaluated at start 196 with the same four horizons.

A4-MEC records deterministic temporal coverage and per-level token use, normalized target-query
attention entropy and attention mass by source level for one fixed diagnostic forward, plus
foreground error on geometrically identified re-entry and non-re-entry records. The fresh holdout
does not change the selected models.

## 8. Commands

Run all commands from
`C:\Users\leoni\Documents\projects\hierarchical-video-prediction-path-b` in PowerShell:

```powershell
$python = 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction\venv\Scripts\python.exe'

# Verify tracked generated manifests have not drifted.
& $python scripts/v2/generate_article4_campaign.py --check

# Run the complete automated test suite.
& $python -m pytest -q

# Read-only campaign status.
& $python scripts/v2/run_article4_campaign.py status

# Full launch gate: clean commit, hashes, data, AE, CUDA, disk, and W&B write access.
& $python scripts/v2/run_article4_campaign.py preflight

# One command for Wave 1, dependency resolution, Wave 2, A4-MEC, and sealed confirmation.
& $python -u scripts/v2/run_article4_campaign.py run --execute
```

If the terminal or machine is interrupted, run the **same final command**. Do not delete run
directories or rename configs. To inspect progress from another terminal:

```powershell
& $python scripts/v2/run_article4_campaign.py status
Get-Content artifacts/v2/article4_primary_30k/campaign/logs/*.log -Tail 30
```

If W&B was unavailable during any runs:

```powershell
& $python scripts/v2/sync_article4_wandb.py
```

For a local two-update representative smoke that never writes to the scientific result namespace
or W&B:

```powershell
& $python -u scripts/v2/run_article4_campaign.py run --execute --smoke --allow-dirty --no-wandb
```

## 9. Verification record

The release verification covers:

- deterministic generation and drift checking of all 80 Wave 1 run specs;
- exact counts and reuse rules for 43 configurations / 86 seed runs;
- exact token accounting, physical time IDs, fine cell IDs, target invariance, residual-parent
  gathers, three-level and overlap contracts;
- CPU unit/integration tests for config authorization, dependency selection, resume bookkeeping,
  post-training selection, explicit confirmation windows, and attention diagnostic aggregation;
- CUDA forward/backward worker smoke for the smallest and largest frontier sequences, A4-ALC,
  A4-CEX, and A4-STS;
- CUDA exact-path checks for level 3, exclusive overlap, and residual overlap;
- an interrupted rolling-checkpoint continuation from step 2 to step 4 with retained/nonduplicated
  metric history;
- full Python compile and test-suite execution;
- online W&B connectivity/project-creation gate from the final clean source commit.

Release evidence:

- verified implementation commit: `f43b0c51fe5a3d0115a3ea0a1f132b62ab6ef38c` (including the
  slow-network W&B initialization fix);
- full suite: **201 passed**; Python compilation passed;
- representative CUDA campaign smoke: five configurations passed, including 320- and 3,072-token
  extremes; dependent CUDA checker: level 3, exclusive, and residual paths passed;
- resume exercise: step 2 restored and continued to step 4 with exact nonduplicated step ledgers;
- clean online preflight: PASS on RTX 4060 Ti 16 GB with the 3,000/2,700/300 TRB schema and hashes;
- W&B project and connectivity run:
  `https://wandb.ai/leo27heady/hierarchical-video-prediction-article4-v2/runs/article4-30k-preflight-f2be4544227f`;
- machine-readable final launch commit/environment is recorded in
  `artifacts/v2/article4_primary_30k/campaign/preflight.json` each time preflight is run.

The 86 scientific training runs were intentionally not started by code preparation; the explicit
`run --execute` command remains the launch boundary.
