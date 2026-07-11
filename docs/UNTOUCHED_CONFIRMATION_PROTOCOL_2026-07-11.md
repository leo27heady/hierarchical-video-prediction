# Untouched confirmation protocol

**Status:** prepared, not frozen, not accessed
**Purpose:** restore confirmatory status for the central reach/fidelity findings after adaptive use
of the existing test pool.

## Scientific question

Under a bounded context-token budget, do packages whose reach contains the memory-relevant past
outperform shorter packages specifically on records outside the shorter package's reach, and is
the conclusion robust to metrics that penalize hallucinated foreground?

This is a test of the frozen package contrasts, not an opportunity to select new arms, thresholds,
bins, horizons, examples, or stopping rules.

## Dataset

Use a newly generated test set whenever compatible with the frozen generator.  Record:

- generator commit and dirty-diff hash;
- complete configuration and RNG algorithm/seed;
- source-glyph split and indices;
- clip/frame geometry;
- frame, action, position and metadata hashes;
- a manifest with `purpose=confirmation_holdout`.

Training and monitoring loaders must reject this purpose.  The confirmation loader must require an
explicit switch and append the dataset manifest hash, RunSpec hash and timestamp to an access log.
No confirmation image may appear in a training-time grid or qualitative design discussion.

## Frozen primary contrasts

1. `hier-min` versus `flat-min` on the pair-derived band `14 < dt <= 48`.
2. `hier-s2-min` versus `flat-short` on the pair-derived band `8 < dt <= 48`.

The numeric limits are consequences of the compared arm geometries.  The implementation derives
them from metadata and verifies them against these preregistered values.

## Primary populations and endpoints

The primary population is non-empty, geometrically answerable records in the pair-specific reach
band.  Report, without arm-wise survivorship intersection:

1. paired union-mask MSE difference;
2. paired GT-foreground non-empty MSE difference;
3. foreground precision/recall and component missed/extra differences;
4. each arm's scoring coverage.

Positive effect convention must be declared in the generated table.  Bootstrap at the clip level;
report clips, records and unique clusters.  The analysis remains seed-resolved.

## Secondary endpoints

- full-frame MSE;
- GT-background MSE;
- all non-empty answerable records;
- preregistered horizon strata;
- component IoU and centroid distance;
- legacy GT-foreground metric for continuity only.

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

- [ ] record exact dataset size/seed and generator configuration;
- [ ] record artifact/AE/model hashes;
- [ ] freeze all thresholds and component settings;
- [ ] freeze arm/run selection and sampler seeds;
- [ ] freeze bootstrap implementation and number of replicates;
- [ ] freeze table and figure shells;
- [ ] run the complete pipeline on development fixtures;
- [ ] obtain an adversarial review;
- [ ] write `gates/confirmation_protocol_frozen.txt` with commit and protocol hash.

The authorization ledger remains blocked until explicit user approval.
