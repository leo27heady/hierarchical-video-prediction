# Integrated results review — 11 September 2026

All four queued integrated training runs and all four corrected profiling jobs completed. The four training workers used 30.27 hours in total, below the original 35.45-hour estimate. The previous launch failure file is historical; completion is supported by terminal worker records, final results and the corrected-profile completion manifest.

## Verification and results

Each run completed its planned schedule: Moving MNIST 200 epochs / 250,000 optimizer updates, TaxiBJ 100 epochs / 115,000 updates. Each evaluated the test set once after validation-based checkpoint selection. The audit checked log/result agreement, configuration hashes, checkpoint hashes, minimum validation loss, selected checkpoint metadata and finite checkpoint weights. Full test inference was not rerun; aggregate metrics were checked against terminal logs and their per-horizon arithmetic identities. The first fixed test clip was separately regenerated from each seed-0 checkpoint for readable qualitative figures, without selection by quality.

| Dataset | Seed | Test pixel MSE | Test pixel MAE | SSIM | PSNR, dB | Training hours |
|---|---:|---:|---:|---:|---:|---:|
| Moving MNIST | 0 | 0.00989430 | 0.02895909 | 0.82064 | 20.66096 | 11.740 |
| Moving MNIST | 1 | 0.01003525 | 0.02924634 | 0.81896 | 20.59542 | 11.546 |
| TaxiBJ | 0 | 0.000279200 | 0.00960623 | 0.96524 | 37.68943 | 3.526 |
| TaxiBJ | 1 | 0.000276278 | 0.00961522 | 0.96437 | 37.68326 | 3.459 |

Moving MNIST uses 10,000 test clips and TaxiBJ 500. Pixel MSE/MAE are in normalized loader units and are not foreground MSE. Frame-sum MSE/MAE use multipliers 4096 and 2048 respectively. SSIM/PSNR use predictions and targets clipped to [0,1], while MSE/MAE do not. TaxiBJ errors have not been converted back to physical flow units.

Both seeds show similar learning trajectories and final quality. Moving MNIST error increases with the direct prediction horizon, and the qualitative example shows increased deformation and blur at later frames. This supports successful optimization and prediction under the stated protocol. It does not establish superiority over a fully trained matched flat or static baseline. Two seeds and aggregate test metrics do not justify reconstructed confidence intervals; per-clip errors were not retained.

## Corrected resource evidence

The corrected profiler passed its graph-retention guard and CUDA output/gradient parity in both FP32 and BF16. Its source hash matches each result. All 72 recorded whole-model cases were checked: two datasets, two precisions, three packing policies, three core execution modes, and inference/training phases, with five repeats per case. All repeats are finite. The original leaking profiles remain excluded.

For adaptive packing, BF16 training-step allocated peaks are:

| Dataset | Eager, GiB | Checkpointed, GiB | Reversible, GiB | Reversible reduction versus eager |
|---|---:|---:|---:|---:|
| Moving MNIST, batch 8 | 0.7642 | 0.6900 | 0.6729 | About 12% |
| TaxiBJ, batch 16 | 0.1737 | 0.1588 | 0.1557 | About 10% |

Allocator baselines are equal across these modes, addressing the earlier diagnostic retention defect. These are whole-predictor measurements on synthetic inputs with fixed initial weights and a nonzero head. The profiler creates cold AdamW state with learning rate zero; full training uses Adam. Their timing and memory numbers must not be equated with an entire training epoch or a warm optimizer step.

Timing is substantially variable for TaxiBJ: reversible BF16 spans 0.052–0.197 seconds across five repeats. The manuscript records median timings but does not claim stable speed superiority. Packed token values remain FP32 under BF16 autocast; token-count reduction is not a byte-reduction factor.

The profiler measures the whole model only. It does not populate the 72 separately planned core-operation slots. Marking all 144 resource slots complete would have overstated coverage.

## Dissertation changes and review

- Table 4.19 now contains corrected full-model resource measurements and their experimental limits, replacing the rejected-profile placeholders.
- Table 4.21 now contains both seeds for both datasets, with explicit normalization, aggregation, SSIM/PSNR definitions and checkpoint-selection details.
- Figure 4.16 now contains actual fixed-clip predictions and errors. Both TaxiBJ channels are shown with shared scales; the cramped original worker panels were replaced by a figure regenerated from checkpoint outputs.
- Figure И.1 in the appendix shows the two validation trajectories and per-horizon test errors for each dataset, linked from the main results text.
- Chapter 3 no longer says the full CUDA profile is pending. The conclusions now include verified integrated quality and bounded memory savings, while retaining the absence of a matched final-quality comparator.
- Evidence registry and figure provenance were updated. The native placeholder generator preserves the completed figures.

The narrative connects numerical correctness, controlled resource cost, trainability, and full predictive evaluation in that order. It keeps the structural TRB experiments separate from the integrated model's normalized pixel metrics. Existing displayed formulas remain unchanged; this update verifies their preservation and metric usage, not an independent derivation of every historical formula. No training implementation or completed experiment output was modified.

## What remains

The authorized queue is finished. No additional training was launched.

The detailed registry now contains **220 verified and 90 pending slots** out of 310. The 90 pending slots comprise 72 core-operation measurements, 16 conditional recursive-rollout measurements, and two conditional comparator records. They are measurement slots, not 90 experiments. ROLL64 and the matched comparator remain conditional; the current conclusions do not depend on successful outcomes from either. The 11 known administrative fields also remain.

The final PDF and exact release checks are in `output/pdf/DISSERTATION_UKRAINIAN_RESULTS_2026-09-11.pdf` and `docs/reviews/integrated_evidence_2026-09-11/verification/`. Supporting results, source copies, hashes, original panels, regenerated example arrays and checkpoint metadata checks are retained in `docs/reviews/integrated_evidence_2026-09-11/`.

Existing staged changes were preserved. No commit, push, archival deletion or experiment relocation was performed.

Final verification: all seven release checks passed. The 303-page PDF matches the current manuscript sources and rendered-page audit; no out-of-bounds text, replacement characters or near-blank pages were detected. Revised resource and quality tables, example figures, appendix curves and their captions were visually inspected. PDF SHA-256: `3378bf49014efc53cc8bf69249b0741a0bfb502bd38b16f5dc676d51f37bd27a`.
