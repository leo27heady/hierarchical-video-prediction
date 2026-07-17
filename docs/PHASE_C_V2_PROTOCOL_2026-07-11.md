# E6 / Phase C v2: persistent-object long Moving-MNIST boundary

**Original redesign:** 2026-07-11
**Binding amendment and preparation freeze:** 2026-07-17
**Status:** article-utility gate passed; preparation frozen; GPU launch remains fail-closed

This protocol replaces—not edits away—the invalid legacy Phase C identity, trajectory and
equivalence design. The legacy scripts remain prohibited evidence paths.

## 1. Scientific role and question

LoopMom remains the causal memory instrument because it exposes exact geometric observability and
re-entry. E6 is a recognizable external boundary on standard Moving-MNIST dynamics; it does not
instantiate the same off-screen mechanism and cannot prove natural-video transfer.

The registered question is:

> On a familiar 300-frame autoregressive Moving-MNIST rollout, do the token-matched 1,152-token
> flat and hierarchical packages differ in foreground fidelity, identity, trajectory, or object
> survival when position-order identity assumptions and survivorship filtering are removed?

Positive, null, heterogeneous and metric-limited outcomes are all publishable boundary results.

## 2. Fixed data roles

All sources are official MNIST IDX files from the PyTorch dataset mirror and are hash pinned.

| Role | Glyph split | Generator seed | Clips × frames | Permitted use |
|---|---|---:|---:|---|
| training | official train | 20260718 | 3,000 × 100 | codec and predictor training |
| development | official test | 20260719 | 500 × 348 | monitoring, classifier/evaluator gates |
| protected holdout | official test | 20260720 | 500 × 348 | one registered final evaluation only |

The generator saves schema `moving-mnist-persistent-v1`: stable object IDs, source glyph indices,
classes, top-left trajectories, per-frame boxes, pairwise overlap state, all-zero action
conditioning, source hashes and an immutable manifest. Training and official-test glyph pools are
isolated. Exact rendered-clip hashes must have zero intersection among all three roles.

The protected array is generated only after implementation, classifier, evaluator and dispatch
semantics are frozen. Each evaluation logs manifest identity and RunSpec before opening the array.

## 3. Fixed model matrix

The two configurations are existing registered RHVD arms:

| Arm | Coarse context | Fine context | Reach | Context tokens |
|---|---:|---:|---:|---:|
| `mmL-flat` | 0 | 18 | 18 frames | 1,152 |
| `mmL-hier` | 48 | 6 | 48 frames | 1,152 |

Both use the same freshly trained two-level codec, diffusion head, 30,000 optimizer updates,
batch 16, learning rate 3e-4, 2,000-update warm-up, EMA 0.999 and bf16. Seeds are 0, 1 and 2.
Predictor training and validation use only training/development data. No protected frame is opened
during training.

The final evaluation uses holdout clips 250–349, 48 GT warm-up frames, a 300-frame self-fed
rollout, four-frame windows, DDIM NFE 16, sampler seed 0 and common random numbers across arms.

This is a whole-configuration comparison. It does not isolate hierarchy, temporal reach, target
layout or optimization difficulty individually.

## 4. Detection, assignment and intention-to-score semantics

Predictions are thresholded at 0.3; connected components smaller than 30 pixels are discarded.
All remaining components are retained, including extras. Predicted components are assigned jointly
to generator-provided persistent objects with Hungarian matching over:

`0.5 × normalized centroid distance + 0.5 × (1 − box IoU)`.

Costs above 0.65 are rejected. Detection list order and horizontal position carry no identity.
The GT trajectory coordinate is the center of the generator-provided 28×28 glyph box.

- Every labelled GT object is in the primary population.
- A miss receives the 64×64 frame-diagonal trajectory penalty.
- A miss or classifier abstention is an identity error.
- Extra components remain explicit false positives.
- Arms are never intersected by finite/scorable records.
- Matched-only identity/trajectory results are diagnostics, not substitutes for primary results.
- All endpoints are additionally stratified by generator-labelled overlap.

## 5. Classifier and evaluator reliability gates

The identity classifier is trained only on official MNIST-train glyphs, sampled uniformly between
canonical 28×28 inputs and deterministic crops produced by the exact production component
extractor on those same glyphs. Abstention is disabled (confidence threshold 0.0), eliminating a
result-dependent coverage trade-off. Before holdout generation, a versioned report must pass:

| Domain | Minimum accuracy |
|---|---:|
| clean official test | 0.98 |
| translated/canvas crop | 0.95 |
| non-overlap rendered component | 0.94 |
| AE reconstruction | 0.90 |
| deliberately degraded development sample | 0.75 |
| overlapping component | diagnostic; no pass threshold |

Per-class confusion and ten-bin calibration error are retained. Overlap classification is
explicitly limited because max-composited digits may form one ambiguous connected component.

The exact production evaluator is then run on 2,000 seeded development frames using raw GT frames
and AE round trips. Its preregistered non-overlap floors are: raw coverage 0.95, raw identity
0.92, raw trajectory error ≤6 px, AE coverage 0.85 and AE identity 0.80. All-frame and overlap
ceilings are reported even when the non-overlap gate passes.

## 6. Endpoints and inference

One simultaneous 95% studentized max-statistic clip-cluster bootstrap family contains five
co-primary effects for each of three training seeds (15 effects total). Positive always favors
the hierarchical package:

1. foreground MSE: flat minus hierarchy;
2. union-mask MSE: flat minus hierarchy;
3. intention-to-score trajectory error: flat minus hierarchy;
4. intention-to-score identity accuracy: hierarchy minus flat;
5. object assignment coverage: hierarchy minus flat.

The bootstrap uses 10,000 replicates and seed 20260722. Training seeds remain visible; evaluation
clip resampling does not estimate training-seed uncertainty.

Secondary descriptive endpoints are full/background MSE, foreground precision/recall/F1,
missed/extra objects, component IoU, identity coverage, matched-only diagnostics and
overlap-stratified versions.

Equivalence is disabled. The earlier ±0.005 foreground margin is withdrawn before protected access
because it lacked a defensible native-unit smallest effect of interest. An interval crossing zero
is inconclusive, not equivalence.

The terminal label is:

- `HIERARCHY_ADVANTAGE` if all three simultaneous foreground-MSE intervals are positive, every
  other co-primary has at least two positive point estimates and no interval wholly favors flat;
- `FLAT_ADVANTAGE` under the sign-reversed symmetric rule;
- `MIXED_BOUNDARY` for a complete valid result meeting neither directional rule;
- `UNSCORABLE` if protected alignment, evaluator reliability, completion or coverage validity
  fails.

## 7. Required prelaunch CPU and provenance gates

The suite must cover detection-order permutation, left/right crossing, overlap/separation,
missing and extra objects, empty predictions, wrong location, wrong class, threshold/tie behavior,
differing arm coverage, overwrite refusal, official source counts/hashes and zero exact-clip
collisions.

The launch controller additionally requires:

- a clean committed source state;
- immutable codec, classifier and six predictor RunSpecs;
- official-source, training/development and protected-manifest attestations;
- classifier and evaluator-reliability attestations bound to their report hashes;
- an exact six-training/six-evaluation dispatch;
- online W&B connectivity and cloud artifacts;
- mechanical wall-clock GPU accounting, including failures and retries;
- no other project trainer/evaluator;
- a 40 GPU-hour programme maximum and 39.35-hour allocated cell maximum.

The protected reserve cannot fund E6. No adaptive seed, arm, margin, threshold, endpoint, NFE or
holdout slice may be added after access.

## 8. Binding implementation

- configuration: `experiments/configs/phase_c_v2.json`;
- generator: `hvp_rev/data/moving_mnist.py`, `scripts/gen_moving_v2.py`;
- object evaluator: `hvp_rev/phasec_v2.py`, `hvp_rev/phasec_eval_v2.py`;
- classifier gate: `scripts/validate_phasec_classifier.py`;
- evaluator ceiling: `scripts/validate_phasec_evaluator.py`;
- controller: `scripts/run_phasec_v2.ps1`;
- legacy prohibited paths: `scripts/run_phaseC_mmnist.sh`, `hvp_rev/phasec_eval.py`,
  `hvp_rev/phasec_summary.py`.
