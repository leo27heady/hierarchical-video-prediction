# Untouched confirmation protocol

**Status:** frozen 2026-07-12 before holdout generation; holdout not generated/accessed
**Purpose:** restore confirmatory status for the central reach/fidelity findings after adaptive use
of the existing test pool.

## Scientific question

Under a bounded context-token budget, do packages whose reach contains the memory-relevant past
outperform shorter packages specifically on records outside the shorter package's reach, and is
the conclusion robust to metrics that penalize hallucinated foreground?

This is a test of the frozen package contrasts, not an opportunity to select new arms, thresholds,
bins, horizons, examples, or stopping rules.

## Dataset

Generate exactly 500 loopmom clips × 200 frames with required seed `20260712`, world/viewport
192/64, three sprites, speed 4, `p_cont=0.9`, `p_hold=0.1`, and the MNIST training-image glyph pool
used by the legacy generator. This preserves distribution continuity while ensuring a new explicit
RNG stream. The evaluation population is fixed to clip indices 250–349: the first 100 clips of the
last-250 validation/evaluation pool, matching legacy loader semantics. Record:

- generator commit and dirty-diff hash;
- complete configuration and RNG algorithm/seed;
- source-glyph split and indices;
- clip/frame geometry;
- frame, action, position and metadata hashes;
- a manifest with `purpose=confirmation_holdout`.

Training and monitoring loaders must reject this purpose.  The confirmation loader must require an
explicit switch and append the dataset manifest hash, RunSpec hash and timestamp to an access log.
No confirmation image may appear in a training-time grid or qualitative design discussion.

The generator must be pixel/RNG-parity tested against the legacy `scripts/gen_looping.py` path,
save aligned actions and viewport positions, and save source glyph indices/world coordinates. It
must refuse overwrite. The generated clip hash must differ from the legacy train and test hashes.

## Frozen primary contrasts

1. `hier-min` versus `flat-min` on the pair-derived band `14 < dt <= 48`.
2. `hier-s2-min` versus `flat-short` on the pair-derived band `8 < dt <= 48`.

The numeric limits are consequences of the compared arm geometries.  The implementation derives
them from metadata and verifies them against these preregistered values.

## Confirmatory hierarchy

### Primary family A — continuity effects

For both selected pairs and every registered seed, reproduce the legacy paired foreground-MSE
effect on the frozen legacy population and report the exact legacy fallback semantics. This tests
the original configuration signs without silently changing the estimand.

### Primary family B — pair-specific reach mechanism

The primary mechanistic population is non-empty, geometrically answerable records in the frozen
pair-specific reach band. Report the legacy and non-empty GT-foreground effects for:

1. `hier-min` versus `flat-min`, `14 < dt <= 48`, seeds 0 and 1;
2. `hier-s2-min` versus `flat-short`, `8 < dt <= 48`, seeds 0 and 1.

The B1 result remains a whole-configuration confirmation because its target interface is coupled.

### Secondary family — complete frozen frontier

Evaluate every run declared in `experiments/configs/e2_untouched_confirmation.json` once. Treat the
frontier contrasts as one correlated secondary family and use a 10,000-replicate, 95% studentized
max-statistic clip-cluster bootstrap (bootstrap seed 0). c2 B16×30k and B8×60k remain separate
regimes. The 480-versus-512 near-bottom comparison is visibly labelled as a near-match. No new best
cell, interpolated crossing or primary comparison may be selected after results.

### Corroborating complete metrics

Report, without arm-wise survivorship intersection:

1. union-mask, full-frame and GT-background MSE;
2. paired GT-foreground non-empty MSE;
3. foreground precision/recall/F1;
4. component counts, misses/extras, matched IoU and centroid distance;
5. each arm's non-empty/scoring coverage and the empty-GT stratum;
6. true record-level sharpness plus the explicitly labelled legacy batch statistic if retained.

Positive effect convention must be declared in the generated table.  Bootstrap at the clip level;
report clips, records and unique clusters.  The analysis remains seed-resolved.

Metric threshold is fixed at 0.5 and minimum connected-component area at 4 pixels. Preregistered
horizon strata and all non-empty answerable records are diagnostics, not a source of new primary
claims.

Empty-GT frames are a separate diagnostic stratum.  They never inherit full-frame values under a
foreground label.

## Interpretation branches

- **Primary intervals support the development sign on both complete metrics:** strengthened
  confirmation that reach explains the package effect at the tested configurations.
- **GT-foreground supports but union/full/component metrics do not:** the legacy result is
  foreground-fidelity-specific and incomplete with respect to hallucinations.
- **Signs vary by seed or metric:** report heterogeneity; do not pool it into equivalence.
- **Intervals reverse or include practically material opposing effects:** central claim is revised
  or demoted; the holdout is not rerun after tuning.

No outcome establishes a continuous unique frontier crossing or a universal law.

## Freeze procedure

Before creating the gate marker:

- [x] record exact dataset size/seed and generator configuration;
- [x] record artifact/AE/model hashes in the frozen E2 config;
- [x] freeze all thresholds and component settings;
- [x] freeze arm/run selection and sampler seeds;
- [x] freeze bootstrap implementation and number of replicates;
- [x] freeze table and figure shells;
- [x] run the complete generator/loader/checkpoint/rollout/metric path on development fixtures;
- [x] obtain adversarial review of the scientific plan and incorporate its amendments;

After this frozen text and config are committed, create hash-bound JSON attestations for the
protocol, design and complete CPU validation report. Editing an attested subject invalidates its
gate automatically.

The authorization ledger remains blocked until explicit user approval.
