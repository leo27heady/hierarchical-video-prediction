# T6c inner-loop memory panel (measured)

date: 2026-07-03 20:45 | device: NVIDIA GeForce RTX 4060 Ti | torch 2.11.0+cu126 | fp32 | B=12 | EnergyTransformer d=256 depth=6 heads=8 n_ctx=6 (560-token joint sequence) | descent: momentum beta=0.9 alpha=0.1 | loss SmoothL1(y_K, target) | AdamW, 3 measured steps after 1 warmup | spill guard 60s/step

Arms 2-4 share the identical truncate-style objective (second-order through the FINAL descent step only; CPU-fp64-verified grad-cos(reversible, checkpoint) > 0.999999); naive is the full-second-order-graph baseline. Pre-registered expectation: naive OOMs/spills by K~8-16; truncate/checkpoint/reversible within a few % of each other.

| arm | K | status | peak GB | s/step |
|---|---|---|---|---|
| naive | 4 | OK | 18.141 | 17.10 |
| naive | 8 | OOM | 29.763 | - |
| naive | 32 | OOM (implied) | - | - |
| naive | 128 | OOM (implied) | - | - |
| truncate | 4 | OK | 5.282 | 0.80 |
| truncate | 8 | OK | 5.282 | 0.83 |
| truncate | 32 | OK | 5.282 | 2.23 |
| truncate | 128 | OK | 5.282 | 7.85 |
| checkpoint | 4 | OK | 5.282 | 0.59 |
| checkpoint | 8 | OK | 5.283 | 0.82 |
| checkpoint | 32 | OK | 5.286 | 2.23 |
| checkpoint | 128 | OK | 5.296 | 7.85 |
| reversible | 4 | OK | 5.282 | 0.82 |
| reversible | 8 | OK | 5.282 | 1.29 |
| reversible | 32 | OK | 5.282 | 4.11 |
| reversible | 128 | OK | 5.282 | 15.37 |

states-vs-graph accounting: per-step (y,v) trajectory state = 0.117 MB; checkpoint stores K x that = 15.0 MB at K=128 = 0.28% of its measured 5.296 GB peak -> the MOST reversibility could save on this axis. Reversible's constant fp64 (y,v) shadow = 0.234 MB.

status legend: OK = measured | OOM = torch.cuda.OutOfMemoryError | SPILL = a step exceeded 60s wall-clock (WDDM shared-memory spill) | (implied) = skipped, a smaller K in the same arm already failed.
