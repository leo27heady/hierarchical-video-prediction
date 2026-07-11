# Phase C v2: persistent-object long-horizon control

**Status:** redesigned and CPU-prepared; GPU launch blocked
**Replaces:** the legacy Phase C identity/trajectory/equivalence design

## Question

Does a long-reach hierarchical package differ from a token-matched short flat package on a
standard Moving-MNIST long rollout when identity, trajectory, hallucinations, misses, extras, and
coverage are measured without position-order identity assumptions?

The result may be positive, null, heterogeneous, or unscorable.  Phase C does not assume that a
standard benchmark must reproduce the looping-world mechanism.

## Dataset contract

- Training clips draw glyph instances only from the official MNIST training split.
- Evaluation clips draw glyph instances only from the official MNIST test split.
- Every object has a stable ID, source glyph index, class, trajectory, per-frame box, overlap
  state, and generator seed.
- The dataset uses schema `moving-mnist-persistent-v1` and an immutable manifest.
- Development and confirmation purposes are distinct.

The generator is implemented in `hvp_rev/data/moving_mnist.py`; the CLI is
`scripts/gen_moving_v2.py`.

## Assignment and coverage

Predicted connected components are matched jointly to visible ground-truth objects using Hungarian
assignment over normalized centroid distance and bounding-box overlap.  Detection order is
irrelevant.  Assignments above the frozen maximum cost remain unmatched.

- Missing objects receive the frame-diagonal trajectory penalty.
- Extra objects are counted and remain visible in foreground/background/union metrics.
- Identity accuracy is intention-to-score over all labelled GT objects; a missed labelled object
  is incorrect.
- Matched-only accuracy and trajectory are diagnostics.
- Coverage, ambiguity, misses and extras are co-primary outputs.
- Arms are not intersected to retain only mutually finite records.

## Frame endpoints

- full-frame MSE;
- non-empty GT-foreground MSE;
- GT-background MSE;
- union-mask MSE;
- foreground precision, recall and F1;
- component precision/recall, IoU, misses and extras.

## Object endpoints

- intention-to-score trajectory error;
- identity accuracy and identity coverage;
- object scoring coverage;
- count error, missed objects and extra detections;
- overlap-stratified performance.

## Classifier gate

The classifier is not accepted on clean MNIST accuracy alone.  A seeded, versioned report must
cover:

- clean official test glyphs;
- translated/canvas crops;
- component crops from rendered clips;
- overlapping digits, with the expected limitation disclosed;
- AE reconstructions;
- generated or deliberately degraded development samples;
- per-class confusion and confidence calibration.

Thresholds and any abstention rule are frozen from development data.  The classifier gate marker
records its checkpoint and report hashes.

## Equivalence policy

Equivalence testing is disabled initially.  Margins may be enabled only when justified in native
units from evaluator reliability, classifier repeatability, and a substantive smallest effect of
interest.  A confidence interval crossing zero is inconclusive, not equivalence.  Replication
requires every preregistered seed to satisfy the justified rule.

## CPU cases required before authorization

- object detection order permutation;
- left/right crossing and position-order swap;
- overlap and separation;
- one missing object;
- one extra object;
- empty prediction;
- same class but wrong localization;
- correct localization but wrong class;
- assignment threshold and tie behavior;
- differing arm coverage;
- output and dataset overwrite refusal.

The current suite covers the core assignment, miss/extra, empty, identity and provenance cases.
Classifier-domain validation and integration with the post-c2 trainer remain required.

## Launch rule

`scripts/run_phasec_v2.ps1 -CheckOnly` must fail until all Phase C markers exist and the status is
explicitly changed to launchable.  The script contains no enabled training launch path while the
study is blocked.
