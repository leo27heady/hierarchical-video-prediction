# Integrated OpenSTL Verification Report

Date: 2 September 2026

## Outcome

The integrated predictor, OpenSTL adapter, data contracts, resumable four-run controller, evidence
logger, and paired CUDA-memory profiler are implemented in isolated worktrees. CPU correctness and
restart behavior are verified. The implementation is ready for bounded CUDA gates after Article 4
releases the GPU; it is deliberately not yet authorized for the four long runs.

This is a code-readiness result. It is not yet evidence of OpenSTL-level prediction quality or an
end-to-end memory advantage.

## Locked decisions

- One new article combines the controlled TRB hierarchy study and integrated OpenSTL evidence.
- Final integrated confirmation uses Moving MNIST and TaxiBJ only, seeds 0 and 1: four long runs.
- The dissertation remains based on the current Ukrainian manuscript. Chapters 1 and 2 retain
  their structure and most content; new passages are English. Chapters 3 and 4 are changed only as
  the new evidence warrants. Manuscript editing remains deferred until experimental evidence is
  locked.
- Long integrated runs are not started automatically. Capacity, quota, precision, batch size,
  route mode, epochs/updates, and measured queue duration are locked only after bounded gates and
  Leonid's confirmation.
- The existing Article 4 worktree and its pinned commit are read-only while its controller runs.

## Code locations and revisions

| Responsibility | Worktree | Branch | Verified revision |
|---|---|---|---|
| OpenSTL model, method, configs, loaders, preparation tool, contract tests | `C:\Users\leoni\Documents\projects\OpenSTL-integrated` | `codex/hvp-integrated-v2` | `55a023d6d34344430ecc1add39a9422b6b86c319` (model/evaluation implementation `325c268dc2b3200ebeb0447224ee7ee924b0499b`) |
| plans, final manifest, worker, controller, profiler, resume tests | `C:\Users\leoni\Documents\projects\hierarchical-video-prediction-integration` | `codex/integrated-openstl-v2` | implementation baseline `9c49f5a3d7f1342586eedc135842d71f9cc6ecc2`; later verification hardening is in the commit containing this report |

The original OpenSTL checkout with Leonid's TempVerseFormer changes and the live Article 4
worktree were not modified.

## Architecture that was accepted

1. A shared convolutional pyramid produces stride-4 fine candidates and stride-8 full-coverage
   coarse tokens for every observed frame.
2. A fixed-quota router selects genuine fine tokens across the complete space--time volume. It can
   enforce minimum coverage per frame and exposes static, density, hybrid, mixed-training, and
   dense-control policies.
3. Every retained token carries physical x/y, signed time, scale, and hierarchy level. Continuous
   Fourier coordinate features replace positional embeddings tied to packed slot numbers.
4. The packed sequence enters an internally reversible additive-coupling transformer. Vanilla and
   reversible modes implement the same forward equations; only activation storage/recomputation
   differs.
5. Physically positioned future coarse queries cross-attend to packed memory. A convolutional
   decoder predicts full frames as residuals over the last observation.

Routing remains outside the reversible boundary. That is intentional: top-k selection is not
invertible, while the repeated transformer core is. The learned route score remains trainable
through a selected-score embedding even though top-k indices themselves are discrete.

## Dynamic-routing and positional-encoding conclusion

Dynamic regions can be combined with the hierarchy and reversible core without positional
ambiguity because position is derived from the selected token's physical metadata, not from its
rank or slot in the packed tensor. A test permutes packed tokens together with metadata and verifies
that the decoded prediction is invariant within numerical tolerance.

Turning dynamic routing on after static-only training is mechanically possible but scientifically
unsafe. The configured `mixed` policy exposes training to static and hybrid memberships and always
uses hybrid at evaluation. A separate switch gate still compares static, density, and hybrid on the
same validation checkpoint. No inference-only routing claim is accepted without that gate.

At the same quota, dynamic routing retains exactly the same bytes as static routing. Its valid win
is higher quality/stability at equal memory or equal quality at a smaller quota. It is not a third
multiplicative memory reduction.

## Memory hypotheses and comparators

| Mechanism | What should decrease | Required paired evidence | Current status |
|---|---|---|---|
| hierarchy | retained tokens, attention work, context bytes | flat fine grid versus coarse plus detail at identical core width/weights where possible | analytic ratio verified; CUDA peak pending |
| reversibility | retained activations inside the repeated core during training | identical checkpoint and forward result, vanilla versus reversible | numerical parity verified; CUDA peak pending |
| dynamic routing | quota needed to meet a quality target | static/density/hybrid at equal K, then a smaller-K frontier | structurally verified; quality frontier pending |
| full model | end-to-end peak memory at competitive quality | matched same-host OpenSTL baselines | pending bounded and long runs |

The provisional configurations retain 800 versus 2,560 Moving-MNIST tokens and 80 versus 256
TaxiBJ tokens, an exact 3.2x retained-token ratio. This cannot be reported as 3.2x end-to-end GPU
memory. Reversibility affects only the core activation share, so its end-to-end reduction will also
be smaller than its isolated-core ratio.

## Verification performed without GPU interference

- 19 OpenSTL model/data tests pass with CUDA hidden.
- Five HVP controller/worker tests pass with CUDA hidden.
- Both dataset shapes complete forward/backward and checkpoint round trips.
- Real fixed quota, unique indices, temporal minimums, spatial/temporal router response, mixed-mode
  behavior, coordinate equivariance, and packed-permutation invariance pass.
- Reversible and vanilla outputs, input gradients, and parameter gradients pass in fp32 and CPU
  bfloat16 autocast tolerance; inverse reconstruction drift passes.
- OpenSTL registry, method training/validation steps, and optimizer construction pass.
- A complete tiny TaxiBJ job trains, validates, tests, writes panels/metrics/checkpoints, stops at a
  development cap, resumes, and reaches the exact originally planned eight optimizer updates.
- Moving MNIST now accepts `num_workers=0`; fixed test/validation arrays are memory mapped.
- TaxiBJ normalization occurs per sample rather than copying multi-gigabyte arrays; validation is a
  chronological training tail separated by an excluded gap.
- Test aggregation is streaming. It records OpenSTL-style MSE/MAE/RMSE, per-frame PSNR/SSIM, pixel
  metrics, and per-horizon curves without retaining all predictions in host RAM.
- `py_compile` and `git diff --check` pass for implementation files.

The two expected test warnings are non-failures: one unit test calls `self.log` without attaching a
Lightning Trainer, and CPU-only loading warns that pinned memory has no accelerator.

## Campaign and evidence behavior

The prelock manifest contains exactly:

| Run | Dataset | Seed |
|---|---|---:|
| `INT_MM_S00` | Moving MNIST | 0 |
| `INT_MM_S01` | Moving MNIST | 1 |
| `INT_TB_S00` | TaxiBJ | 0 |
| `INT_TB_S01` | TaxiBJ | 1 |

One controller command executes the ordered queue sequentially. Completed runs are skipped. Each
active run writes an atomic rolling checkpoint; after interruption, `run` resumes from it and uses
a fixed optimizer-update total so a mid-epoch Lightning restart cannot silently shorten training.
An exclusive controller lock prevents duplicate queue owners.

Local JSONL/checkpoints are authoritative. W&B mirrors train/validation loss, OpenSTL metrics,
learning rate, gradient and parameter norm, update/sample throughput, allocated/reserved and peak
CUDA memory, routing entropy/coverage/churn/margin, configuration/provenance, and validation/test
panels. W&B failure is recorded and disables only the mirror, never local evidence.

W&B 0.27.2 was installed in the OpenSTL environment and the requested project was created with a
setup-only run:
`https://wandb.ai/leo27heady/hierarchical-video-prediction-integrated-v2`.

Final authorization is bound to the manifest SHA-256 plus a full lock fingerprint over both Git
commits, all configs, and every source/prepared data file. `run` recomputes it and rejects drift.

## Clean preflight result

The cheap preflight passed with both repositories clean at the then-current commits, all four
configs present, exact dataset/seed cells, source data present, CUDA visible, and sufficient D:
space. Prepared validation/cache files are intentionally missing and were warning-only because
conversion is deferred until Article 4 releases host I/O/RAM.

Observed resources at the final audit:

- GPU: RTX 4060 Ti, 16,380 MiB total, approximately 4,926 MiB used and 99% utilized by Article 4;
- host RAM: 13.0 GiB free;
- C: 32.85 GiB free; and
- D: 74.94 GiB free, with an 8 GiB conservative integrated-output projection and a mandatory
  15 GiB post-projection reserve.

Prepared data adds approximately 0.15 GiB for Moving-MNIST validation and 1.28 GiB for TaxiBJ
float32 memmaps on C:. There is sufficient capacity, but conversion remains deferred to avoid
competing with the live run.

## Remaining gates after Article 4

1. Prepare and hash the distinct Moving-MNIST validation set and TaxiBJ float32 memmaps.
2. Tiny-overfit 32--64 clips for each dataset.
3. Measure 200 updates per dataset to select capacity, batch size, precision, and compute an actual
   four-run GPU-hour estimate.
4. Profile identical-checkpoint flat/hierarchical and vanilla/reversible paths, both core-only and
   end-to-end.
5. Compare static, density, and hybrid routing at equal K; add a smaller-K point only if the quality
   gate supports it.
6. Run a 5--10 epoch screen on both datasets and compare against matched local OpenSTL checkpoints
   or runs.
7. Freeze configs, commits, data hashes, update/epoch schedule, baseline protocol, and duration;
   obtain Leonid's confirmation; then authorize and launch the four-run queue.

The current width-128/depth-6 model has about 1.91 million parameters, much less than the local
SimVP configurations (about 46.77 million on Moving MNIST and 9.96 million on TaxiBJ). This may be
an efficiency advantage or insufficient capacity. It must be decided by the bounded screen, not by
assumption. Baseline tables will separate parameter memory from activations/context and compare
quality at both matched protocol and memory-aware operating points.

## Article 4 status and timing boundary

At the latest status check, Article 4 wave one had 17 complete, one running at 3,000/30,000, 62
pending, and zero failures; six dependent wave-two runs and finalization follow. The first 16 runs
averaged 53.2 minutes, but remaining larger-token runs are slower. Linear extrapolation from the
320/768/1,152-token timings places the remaining campaign near 150--170 GPU hours, approximately
6--8 uninterrupted days. This is an operational estimate, not a deadline.

## Commands

Status and cheap preflight:

```powershell
Set-Location 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction-integration'
& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  scripts\v2\run_integrated_openstl_campaign.py status
& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  scripts\v2\run_integrated_openstl_campaign.py preflight `
  --allow-missing-prepared-data --check-wandb
```

Preparation, CUDA-gate commands, final authorization format, launch, and memory profiling are in
`INTEGRATED_OPENSTL_IMPLEMENTATION_AND_OPERATIONS_2026-09-02.md`. Do not run preparation or any
integrated CUDA command while Article 4 owns the GPU.
