# E8a pose-indexed raw-memory reference protocol

**Status:** frozen before generation or access of the E8a reference set. Two pre-access audit
corrections are recorded: the answerable population is \(6 < dt \leq 48\), not the contradictory
draft expression \(dt > 48\); and the ceiling verdict is based on the four neural-to-raw
comparisons rather than requiring K48 to strictly beat K38 after geometric coverage may already
have saturated. All six effects remain in one simultaneous family.
**Scientific role:** privileged closure reference for explicit access; not a token-matched
competitor, learned-retrieval result, or new confirmation of the neural frontier.

## Question

E8a asks how much of the remaining LoopMom error is attributable to access and self-fed
generation rather than absence of the relevant past. It supplies a deterministic classical
system with the information that the neural predictors do not receive explicitly: the exact
viewport pose in world coordinates and a raw-pixel memory assembled from ground-truth warm-up
frames.

The reference is deliberately privileged. Its result cannot establish architectural superiority,
parameter efficiency, or budget-matched competitiveness. It estimates a static-world access
ceiling and tests whether the learned systems approach that ceiling under the same evaluation
records.

## Independent shared-population reference set

Generate one new 500-clip, 200-frame LoopMom set with seed `20260717` and purpose
`retrieval_reference_holdout`. The generator, MNIST-train glyph pool, 192-pixel torus, 64-pixel
viewport, three fixed digits, speed 4, momentum 0.9, hold probability 0.1, saved actions,
viewport positions, and persistent object metadata are otherwise identical to the E2/E7
generator path.

Indices 0--249 are an unused split retained for schema symmetry. The registered evaluation
population is indices 250--349: 100 clips, 48 ground-truth warm-up frames, and 152 future frames.
No frame may be displayed, selected, or used for training. Access must be manifest-checked and
logged before array opening.

The set is independent of development, E2, E7c, and E7a. This prevents a purpose mismatch with
the codec-robustness holdout and permits exact paired comparison between the classical reference
and the frozen neural checkpoints.

## Frozen pose-mosaic algorithm

For memory span \(K \in \{6,14,38,48\}\), use only warm-up frames
\([48-K, \ldots, 47]\). For every stored frame and viewport column \(u\), map all 64 row pixels to
world column

\[
  x=(p_t+u)\bmod 192,
\]

where \(p_t\) is the saved viewport left edge. The first observation of a world pixel is retained.
Every repeated observation must be byte-identical; disagreement is a validity failure because
LoopMom is static in world coordinates.

At each future frame, query the map with the saved future pose and crop the corresponding 64
world columns. Columns not observed in the registered memory span are filled with a frozen
row-wise intensity prior computed only from the original predictor-training array
`data/loopmom_train_3000x100.npy`. The prior is one mean value per image row, averaged over
training clips, frames, and horizontal positions. Its artifact and source hash must be recorded.

The map is frozen at frame 47. It is never updated with future ground truth, neural predictions,
or its own output. No decoder, learned parameter, future pixel, object metadata, or glyph label is
used to form a prediction. Object metadata is available only for later diagnostic auditing.

The four spans have prespecified roles:

- `pose-mosaic-k6`: short-window geometric control matching the hardness instrument;
- `pose-mosaic-k14`: raw-access counterpart to the 896-token flat reach;
- `pose-mosaic-k38`: raw-access counterpart to the tested 2,432-token flat reach;
- `pose-mosaic-k48`: primary privileged reference matching the hierarchy's maximum tested
  LoopMom reach.

These are temporal-span correspondences, not memory-budget matches. Report unique stored world
columns, raw pixel bytes, pose bytes, and mask bytes for every variant.

## Frozen neural reference matrix

Evaluate the seven existing original-codec predictor-seed-0 checkpoints on the same E8a set:
`hier-min`, `flat-min`, `hier-long`, `flat-budget`, `strided-flat`, `hier-c2`, and `flat-c2`.
Training is prohibited. Evaluation remains BF16, NFE 16, sampler seed 0, warm-up 48, 100 clips,
152 rollout frames, complete metric version `complete-v2-thr0.5`, and online W&B logging.

The seven-arm matrix preserves the established lower frontier, interior reach comparison,
near-matched 1,088-token temporal versus 1,152-token spatial package comparison, and upper tested
inversion. It is a shared-population bridge to the classical reference, not an additional neural
confirmation study and not authority to reopen E2, E7c, or E7a.

## Endpoints and registered family

All methods must emit exactly 15,200 aligned `(clip, horizon)` records with the existing
`hard`, `dt`, and `seen` labels and the full false-positive-sensitive metric suite.
Pose-mosaic records additionally include current-crop observed-column coverage, stored-world
coverage, observed-pixel MSE, and unobserved-pixel MSE.

The primary endpoint is legacy ground-truth-foreground MSE over all registered records, retained
for exact comparability. Six positive-direction effects form one studentized max-statistic
clip-cluster simultaneous 95% family:

1. `pose-mosaic-k14` error minus `pose-mosaic-k48` error;
2. `pose-mosaic-k38` error minus `pose-mosaic-k48` error;
3. `hier-min` error minus `pose-mosaic-k48` error;
4. `hier-long` error minus `pose-mosaic-k48` error;
5. `strided-flat` error minus `pose-mosaic-k48` error;
6. `flat-c2` error minus `pose-mosaic-k48` error.

Positive effects favor the 48-frame explicit-access reference. The four neural-to-K48 effects
(items 3--6) determine the registered ceiling category. Items 1--2 are prespecified raw-span
diagnostics within the same multiplicity-controlled family: K14→K48 tests a raw-memory
dose response, while K38→K48 can identify geometric saturation. With a 64-pixel viewport and
roughly four pixels of motion per frame, 38 frames can already expose all 192 world columns; a
zero K38→K48 effect is therefore compatible with a valid raw-access ceiling rather than evidence
against it. Three populations are frozen as
secondary mechanistic reads: the K48-answerable re-entry band
(`6 < dt <= 48`, `hard > 0.1`, `seen >= 0.5`), the registered 14-to-48 reach-differential band,
and the registered 38-to-48 reach-differential band. The earlier draft expression `dt > 48`
would have selected evidence lying outside K48 reach and was corrected before generation,
access, or result observation. Full-frame, background, union,
precision, recall, F1, missed/extra components, coverage, horizon windows, and empty/non-empty
populations are corroborating and cannot override the primary category.

## Interpretation categories

- `RAW_ACCESS_CEILING_CI_CLEAN`: all four registered neural-to-K48 point estimates and their
  simultaneous lower bounds are positive.
- `RAW_ACCESS_DIRECTIONAL_REFERENCE`: all four neural-to-K48 point estimates are positive, but
  at least one simultaneous lower bound includes zero.
- `RAW_ACCESS_CEILING_NOT_ESTABLISHED`: at least one neural-to-K48 point estimate is
  non-positive.
- `INVALID`: static-world consistency, dataset identity, access logging, record alignment,
  artifact integrity, or numerical-finiteness checks fail.

The K14→K48 and K38→K48 effects are always reported alongside a coverage-saturation flag but do
not override these categories. These labels describe this static LoopMom reference only. Even a clean ceiling does not prove that
hierarchical generation is unnecessary; it shows that exact pose-indexed access to raw observed
content can remove error that remains under learned compression and self-fed generation.

## Validity and stop rules

Stop before interpretation if any of the following occurs:

- a repeated world-coordinate observation differs at the byte level;
- a prediction uses a post-warm-up ground-truth pixel;
- an unknown column is filled from anything other than the frozen training-only prior;
- the E8a set collides by clip/action/position/object hash with an earlier locked set;
- any neural checkpoint, codec, sampling, metric, or evaluation-geometry hash differs from the
  frozen configuration;
- record keys fail exact pair alignment;
- a required W&B run or immutable artifact is missing.

No adaptive memory span, fill rule, checkpoint, sampler, threshold, endpoint, or comparison may
be added after the E8a set is generated or any result is visible.

## Budget and reporting

E8a performs no training. The expected GPU cost of the seven frozen neural evaluations is about
1.0 hour; the hard project-ownership maximum is 4.0 GPU-hours including gates, failed launches,
retries, evaluation, and upload. The pose-mosaic evaluation and bootstrap analysis are CPU-only.

The article and dissertation must label the result as a **privileged pose-indexed raw-memory
reference**. Tables must separate it visually from token-matched neural configurations and report
its explicit poses, raw-pixel storage, stored coverage, and lack of self-fed state. Pose payload
accounting includes both stored warm-up positions and all exact future query positions; reporting
only the warm-up poses would understate the privileged side information.
