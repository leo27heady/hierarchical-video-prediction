# Article 4 runtime, data-path, and implementation audit

## Scope

Local profiling used the RTX 4060 Ti 16 GB, PyTorch 2.11/CUDA 12.6, bf16 autocast, the real TRB
development array, and the frozen `hier_ae_mom.pt` codec. Probes executed complete optimizer
updates, not forward-only synthetic estimates.

## Measured update profiles

### Historical 384x8 instrument

| Context tokens | Arm/proxy | Batch | s/update | Peak allocated GiB | 30k hours |
|---:|---|---:|---:|---:|---:|
| 320 | flat | 16 | 0.142 | 3.03 | 1.2 |
| 1,152 | hierarchical | 16 | 0.400 | 6.38 | 3.3 |
| 1,792 | flat | 16 | 0.606 | 9.01 | 5.0 |
| 2,432 | flat | 16 | 0.859 | 11.60 | 7.2 |
| 3,072 | flat | 8 | 0.564 | 7.29 | 4.7 |

The 3,072-token arm historically used batch 8 and 60k updates to retain approximately the same
number of training examples as batch-16/30k arms, making its comparable duration about 9.4 hours.
The complete two-seed revised matrix cannot use this protocol within one week.

### Proposed 256x6 controlled instrument

| Context tokens | Batch | s/update | Peak allocated GiB | 12k hours |
|---:|---:|---:|---:|---:|
| 320 | 16 | 0.065 | 1.47 | 0.22 |
| 768 | 16 | 0.130 | 2.40 | 0.43 |
| 1,152 | 16 | 0.201 | 3.21 | 0.67 |
| 1,792 | 16 | 0.336 | 4.55 | 1.12 |
| 2,432 | 16 | 0.487 | 5.88 | 1.62 |
| 3,072 | 16 | 0.648 | 7.21 | 2.16 |

The explicit mixed-precision RMSNorm adaptation improved the 1,792-token point to 0.286 s/update,
3.76 GiB peak, and 0.95 hours/12k. It removed the PyTorch fused-RMSNorm dispatch warning. The
same adaptation must be rerun at every budget before final queue estimates are frozen.

## Data path

- Encoding 3,000 x 100 frames to standardized coarse/fine latents takes about 14--16 seconds.
- At 1,792 tokens, legacy batch construction plus host-to-device transfer takes 1.06 ms/batch;
  the optimized complete update takes 286 ms. Batch assembly is less than 0.5% of update time.
- Full fused self-attention is the dominant compute. Fused scaled-dot-product attention is already
  available and enabled.
- Re-encoding in every process costs about 30--35 minutes over 120 runs. A latent cache is
  operationally helpful but not a decisive week-scale optimization.

## TRB correctness findings

The development manifest at
`experiments/v2/manifests/trb_v2_development.manifest.json` records:

- exact source clip/action SHA-256 identities;
- a deterministic, disjoint 2,700/300 train/validation split;
- valid uint8 clips and int8 actions in `{-1,0,1}`;
- zero failures in 4,096 sampled pixel-exact action/viewport transitions;
- answerable re-entry content beyond a six-frame recent window in 43.3% of evaluated sampled
  frames.

The legacy test file is not a new holdout and is excluded. The new confirmation set does not exist
until protocol lock. The source development artifact has one irrecoverable provenance limitation:
its original generator seed was not recorded.

## Implementation findings and corrections

1. **Fixed target:** the existing fixed-target control is the correct semantic basis; context
   allocation must not change the four-frame target.
2. **Selected-token indexing defect:** the prepared global fine-selection path used an absolute
   context frame as an index into a short local fine tensor. It failed when coarse reach exceeded
   fine-window length. Local gather identity and absolute temporal conditioning are now separate,
   with a regression test.
3. **Dense candidate staging:** a selected-token experiment should not transfer the entire dense
   historical fine tensor to CUDA and then gather it. An optional already-packed `(B,K,C)` path now
   produces the same tokens/metadata as dense gather. The v2 batch builder must gather on CPU and
   transfer only the packed state.
4. **RMSNorm fallback:** attention q/k RMSNorm previously mixed bf16 inputs and fp32 affine weights,
   preventing fused dispatch. The Path B adapter retains fp32 master weights and casts only the
   kernel operand; fp32 output/gradient parity tests pass.
5. **Article 4 positions:** learned physical 8x8 cell embeddings plus explicit absolute temporal
   indices are adequate for deterministic fixed-grid Article 4 allocation. They are not adequate
   for Article 5's moving/variable-scale regions; Article 5 still requires coordinate/scale-aware
   encoding.

## Remaining launch blockers

- implement and test the exact Balanced/Linear/Geometric and STS allocation schedules;
- make the v2 batch builder consume the manifest's split indices and CPU-pack selected cells;
- complete the config-driven trainer, compact-checkpoint policy, and run registry;
- pass causal, target, exact-token, deterministic, tiny-overfit, and convergence gates;
- profile at least one hierarchical packed configuration at every budget, because equal sequence
  length strongly predicts but does not prove identical throughput.

No Article 4 training run is authorized before these blockers pass.
