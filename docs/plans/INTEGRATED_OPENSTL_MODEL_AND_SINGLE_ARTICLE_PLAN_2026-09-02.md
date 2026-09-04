# Integrated OpenSTL Model and Single-Article Plan

Date: 2 September 2026  
Status: implementation contract; GPU execution is deferred until the Article 4 campaign releases
the device  
HVP branch: `codex/integrated-openstl-v2`  
OpenSTL branch: `codex/hvp-integrated-v2`

## 1. Publication and dissertation scope

The redesigned hierarchy study and the integrated model form one article. The article has two
linked evidence blocks:

1. controlled TRB experiments identify how a fixed token budget should be divided across spatial
   and temporal scales; and
2. an OpenSTL-native predictor uses the selected hierarchy together with reversible internal
   computation and information-density-aware routing, then tests whether the mechanisms compose on
   Moving MNIST and TaxiBJ.

The historical `A4_*` and `A5_*` experiment prefixes remain because changing identifiers after
execution would damage provenance. They mean evidence block, not publication count.

The dissertation will continue from
`docs/dissertation/condensed/_translated/_humanized/_ukrainian`. Chapters 1 and 2 retain their
structure and most text. This language instruction is superseded by the author-approved plan of
2026-09-04: new dissertation passages are written in formal Ukrainian; only the mandatory English
abstract is translated from the final Ukrainian abstract. Chapter 1 will later distinguish:

- spatial hierarchy, which trades fine spatial coverage for a smaller retained state;
- temporal hierarchy, which preserves long reach at coarser resolution;
- reversible computation, which targets training activations rather than retained context; and
- dynamic routing, which allocates a fixed detail quota to high-information space--time regions.

Chapters 3 and 4 keep their present semantic roles but replace weak or redundant experiments with
the registered TRB and OpenSTL evidence. Whether the integrated result receives a large standalone
section is decided only after the two-dataset gates. Existing Articles 1--3 text and figures are
reused through the provenance ledger, not rewritten wholesale.

## 2. Why the legacy integrated prototype is not the release model

The inspected `FixedRegionRevPredictor`, `HierarchicalPackedTransition`, and
`ReversibleSecondOrderRollout` establish useful feasibility facts but are not a defensible final
combination:

- the fixed-region tokenizer adds learned embeddings for eight immutable slots; changing which
  physical region occupies a slot changes its positional meaning;
- soft weighting inside all fixed regions does not remove retained tokens and therefore is not
  dynamic compression;
- the packed-SimVP path repeatedly decodes to pixels and re-encodes, which introduces an expensive,
  lossy state transition unrelated to the desired transformer core;
- the external second-order update `p'=q, q'=p+F(q)` is invertible but is not repeated application
  of `F`; and
- the prototype is specialized to one-channel 64x64, 10-to-10 Moving MNIST and does not implement
  TaxiBJ's two-channel 32x32 contract.

Reusable parts are limited to fixed-quota accounting, coordinate metadata tests, and the principle
of custom backward for an internal additive-coupling stack.

## 3. Release architecture

### 3.1 External contract

The OpenSTL model accepts `(B,T_in,C,H,W)` and returns `(B,T_out,C,H,W)`. `T_out` is the configured
`aft_seq_length`, so both Moving MNIST 10-to-10 and TaxiBJ 4-to-4 use one implementation. The model
is end-to-end trainable and does not require a pretrained autoencoder.

### 3.2 Frame pyramid

A shared convolutional encoder produces:

- fine candidates on a grid with stride four; and
- coarse tokens on a grid with stride eight.

For 64x64 Moving MNIST this is 256 fine candidates and 64 coarse tokens per input frame. For 32x32
TaxiBJ it is 64 fine candidates and 16 coarse tokens. Both levels have the same transformer width.
The coarse path always covers the complete observed sequence.

### 3.3 Space--time router and genuine packing

The router scores every fine candidate using content features, information lost by coarse pooling,
local temporal change, and an optional learned correction. It supports:

- `dense`: retain every fine candidate (flat control);
- `static`: retain a content-independent, space--time-stratified fixed quota;
- `density`: deterministic information-density routing;
- `hybrid`: density prior plus a learned score; and
- `mixed`: route-aware training that samples static and hybrid membership.

`Static`, `density`, and `hybrid` are inference membership policies. `Mixed` is a training-exposure
regime, not a fourth inference routing policy. A same-checkpoint static/density/hybrid switch is a
distribution-shift diagnostic; a static-trained versus mixed-trained comparison is a separately
trained development ablation and is not guaranteed by the four final full-model runs.

Selection is global over the observed space--time volume, with a configurable minimum per input
frame to prevent temporal collapse. The chosen features and their physical metadata are gathered
into `(B,K,D)` before transformer execution. `K` is a real fixed quota for a run, not a padded
maximum. Route indices are sorted only for stable execution; their tensor slots carry no semantic
position.

Dynamic routing can be selected at inference through an explicit model method or forward override.
This is mechanically safe because position follows metadata. It is scientifically accepted only
if training exposed the model to dynamic membership and the switch test passes.

### 3.4 Coordinate-aware positions

Each token receives continuous Fourier features of normalized spatial centre, signed normalized
time, cell scale, and a learned hierarchy-level embedding. Coarse and fine observations use past
times ending at zero; prediction queries use positive future times. No learned table is indexed by
packed detail slot. Permuting tokens together with their metadata must permute their embeddings and
leave model output invariant up to numerical tolerance.

### 3.5 Reversible predictor core

The packed context enters a stack of additive-coupling transformer blocks. Each block splits the
channel dimension, applies attention and an MLP through exactly invertible additions, and excludes
dropout/stochastic depth. Two execution modes expose the same mathematical forward pass:

- `vanilla`: ordinary autograd stores block activations; and
- `reversible`: custom backward reconstructs inputs and recomputes the same functions.

Routing, the convolutional encoder, and the output decoder remain ordinary non-reversible
boundaries. Therefore reversibility can reduce only the activation share attributable to the
repeated core; total-model reduction must be measured rather than inferred.

### 3.6 Future decoder

Future coarse-grid queries are initialized from the last observed coarse feature and positioned by
their future physical coordinates/times. A compact cross-attention decoder reads the transformed
packed memory. A convolutional upsampler predicts a residual over the last input frame. This gives
full-frame coverage even where no fine detail token was selected and a strong persistence prior for
TaxiBJ without routing-specific scatter seams.

## 4. Hypotheses and honest component claims

| Component | Valid isolated claim | Required comparator | Invalid shortcut |
|---|---|---|---|
| hierarchy | fewer retained/context tokens and lower attention workload at acceptable quality | flat fine-grid model with the same width/core/decoder | claiming 3--4x before measuring end-to-end peak |
| reversibility | lower training activation peak for the identical forward function | same checkpoint/config in vanilla and reversible modes | comparing a second-order recurrence with ordinary rollout |
| dynamic routing | better quality or robustness than static allocation at the same quota; possibly same quality at a smaller quota | static and density/hybrid routes with physical metadata | calling equal-quota routing another memory multiplier |
| full model | competitive quality and a measured Pareto advantage | matched OpenSTL baselines on the same host/protocol | multiplying component ratios |

Targets remain directional until profiling: hierarchy retained-token target at least 2x (3--4x is a
stretch); reversible core incremental activation target at least 1.7x; quality within 5% relative
for memory-focused non-inferiority and within 2% for parity language.

## 5. Verification gates before GPU training

All checks below run on tiny CPU tensors with CUDA hidden:

1. Moving MNIST and TaxiBJ shape/range contracts.
2. Exact quota and real tensor-size reduction for hierarchy versus flat.
3. Coarse full coverage and valid unique dynamic indices.
4. Spatial and temporal router response to controlled changes.
5. Coordinate-encoding equivariance to packed-slot permutation.
6. Whole-model invariance to a packed detail permutation with metadata.
7. Static/density/hybrid inference switching returns finite equal-shaped predictions.
8. Reversible/vanilla output, input-gradient, and parameter-gradient parity.
9. Reversible inverse reconstruction drift below tolerance.
10. End-to-end forward/backward for both dataset shapes with nonzero encoder, router, core, and
    decoder gradients.
11. Checkpoint save/load round trip and deterministic seed replay.
12. OpenSTL method registry, training step, validation step, and optimizer construction smoke tests.

These checks establish correctness and compatibility, not competitiveness or CUDA-memory savings.

## 6. Experiment matrix after Article 4 releases the GPU

### 6.1 Bounded architecture gate (development only)

Use seed 0 and the validation split. These are not final article quality points.

| ID | Dataset | Change | Purpose |
|---|---|---|---|
| INT-G0 | Moving MNIST | tiny overfit, 32--64 clips | verify loss can fall and decoder can learn motion |
| INT-G1 | TaxiBJ | tiny overfit, 32--64 clips | verify two-channel smooth-field path |
| INT-G2 | Moving MNIST | flat/vanilla vs hierarchical/vanilla vs hierarchical/reversible | verify same forward equations and measured memory ordering |
| INT-G3 | Moving MNIST | static vs density vs hybrid at one fixed quota | select stable route mode by validation quality, churn, coverage, and entropy |
| INT-G4 | both | short 5--10 epoch full-pipeline screen | reject divergence, OOM, or dataset-specific shape assumptions |

Only a few hundred to a few thousand updates are permitted here. Failed gates cause architecture
repair, not long-run multiplication.

### 6.2 Component evidence

Article 4 provides the broad hierarchy/allocation frontier. The integrated model adds paired
profiles using the same trained checkpoint wherever mathematically possible:

| Pair | What changes | Quality interpretation | Memory endpoint |
|---|---|---|---|
| flat vs hierarchical | retained fine grid versus coarse plus packed detail | separate bounded training is needed | retained bytes, inference peak, training peak |
| vanilla vs reversible | execution mode only; identical weights and outputs | exact parity required | incremental core and end-to-end training peak |
| vanilla vs checkpointed vs reversible | activation-storage strategy only; identical model and loss | exact/declared numerical agreement required | allocated/reserved peak, throughput, recomputation |
| static vs dynamic | membership policy at identical quota | route-aware checkpoint plus paired validation | equal retained bytes; compare quality/stability |
| large vs smaller dynamic quota | packed `K` | smaller quota must remain within tolerance | actual retained and peak frontier |

The profiler records allocated and reserved CUDA peaks, batch/precision, parameter and optimizer
bytes, retained payload including metadata, tokens, wall time, throughput, recomputation cost, and
the measurement baseline. At least five warm iterations and twenty measured iterations are used
after the final batch size is selected.

Activation checkpointing is the practical comparator for the reversible core. If it cannot be
profiled under the same model, checkpoint, batch, precision, horizon, and loss, the final claim is
restricted to a comparison with ordinary autograd storage and does not assert the best available
memory--compute trade-off.

Every intended result is classified before writing as a CPU correctness gate, same-checkpoint
profile, separately trained development ablation, long full-model confirmation, matched local
baseline, or published contextual reference. Missing component/baseline cells contract the
corresponding dissertation/article claim; four long full-model runs do not imply that a six-cell
component-training matrix was completed.

### 6.3 Four final full-model confirmation runs

| Run | Dataset | Seed | Protocol |
|---|---|---:|---|
| INT-MM-S00 | Moving MNIST | 0 | OpenSTL 10-to-10, full integrated configuration |
| INT-MM-S01 | Moving MNIST | 1 | same configuration and split |
| INT-TB-S00 | TaxiBJ | 0 | OpenSTL 4-to-4, full integrated configuration |
| INT-TB-S01 | TaxiBJ | 1 | same configuration and split |

Checkpoint selection uses validation only. The test split is evaluated once after configuration and
checkpoint-selection rules are locked. Train long enough to compare fairly with the selected local
OpenSTL baseline schedule; define training by epochs and report optimizer updates, examples, GPU
hours, and early learning curves as secondary evidence.

Matched local SimVP and TAU checkpoints/runs are preferred. Published OpenSTL values are context,
not a substitute for same-host peak-memory profiles. If baseline checkpoints already satisfy the
exact data/config contract, profile rather than retrain them; otherwise the extra baseline GPU time
must be declared separately from the four integrated runs.

## 7. Expected runtime and storage

Exact GPU hours cannot be defensibly fixed before INT-G2 measures updates/second and maximum batch
size. The release estimate is calculated as:

`epochs * steps_per_epoch * measured_seconds_per_step / 3600`

for each dataset, multiplied by two seeds and a 15% recovery/evaluation margin. A go/no-go estimate
is produced after 200 warm measured updates on each dataset. Long runs are not launched if the
four-run queue cannot fit the available time window.

Storage policy:

- one rolling resumable `last.ckpt` and one compact best checkpoint per run;
- JSONL metrics and immutable run manifest locally;
- validation panels at sparse registered intervals, not every step;
- predictions for the locked test saved once; and
- W&B mirrors logs/artifacts but is not the authoritative store.

At the observed 33 GiB free C: drive, datasets remain in the existing OpenSTL checkout and are not
copied. Run outputs should use the existing external experiment-storage drive if available. The
preflight fails closed unless projected free space after the campaign remains at least 15 GiB.

## 8. Campaign and W&B requirements

The final runner consumes one manifest containing the four runs, deterministic run IDs, config/data
hashes, source commits for both repositories, and ordered dependencies. It must:

- resume `last.ckpt` after interruption and skip a completed config hash;
- use atomic status/checkpoint writes and reject duplicate active processes;
- preserve stdout/stderr plus JSONL train, validation, route, timing, and memory events;
- create/use W&B project `hierarchical-video-prediction-integrated-v2` with dataset groups;
- log train/validation losses, MSE/MAE/SSIM/PSNR where applicable, learning rate, gradient norm,
  parameter norm, throughput, GPU memory, routing entropy/coverage/churn/margin, and component
  toggles;
- log sparse input/target/prediction/error panels and final tables;
- continue locally if W&B is unavailable and support a later sync; and
- expose `preflight`, `run`, `status`, `resume`, `profile`, `finalize`, and `sync-wandb` operations.

No new W&B project is contacted and no GPU run is launched while the Article 4 campaign owns the
device.

## 9. Immediate implementation order

1. Implement the OpenSTL-native pyramid encoder, packer/router, physical positions, reversible
   core, future decoder, and method wrapper in the clean OpenSTL worktree.
2. Add the CPU gates and run them with CUDA hidden and constrained thread counts.
3. Add Moving MNIST and TaxiBJ configs plus component overrides.
4. Implement the HVP-side memory profiler, manifest schema, single resumable launcher, and local
   evidence ledger.
5. Verify preflight against existing dataset/checkpoint paths without loading full arrays into RAM.
6. When Article 4 ends, run INT-G0--G4, measure speed/memory, calculate the queue, and ask Leonid to
   confirm the locked long-run configuration before launching the four final runs.
7. Only after results are locked, refactor the dissertation and assemble the single article.

## 10. Continuation checklist

After any context compaction, read all files in `docs/plans/`, beginning with the decision register,
this file, and the master plan. Then inspect both integration worktrees, the current Article 4
campaign status, and
`C:\Users\leoni\Documents\projects\hierarchical-video-prediction\docs\reviews\DISSERTATION_CODE_EXPERIMENT_AND_RESTRUCTURE_REVIEW_2026-08-31.md`.
Never run integrated CUDA work until the Article 4 controller has released the GPU.
