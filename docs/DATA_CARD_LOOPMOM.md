# Data Card: LoopMom Memory-Horizon Testbed

## Purpose

LoopMom is a controlled synthetic instrument for measuring memory-dependent re-entry. A
`192×64` toroidal world is observed through a panning `64×64` viewport, leaving most of the world
off-screen. The generator records camera/action geometry, allowing the time since relevant content
was visible and the pair-specific reach population to be computed exactly.

## Contents

The primary operating point uses 3,000 training clips of 100 frames and separately generated
500-clip, 200-frame evaluation sets. Purpose-locked sets exist for untouched confirmation,
sampler/NFE robustness, alternate-codec robustness and the explicit-access reference. Dataset
manifests record generator commit, parameters, purpose and hashes.

## Splits and access

Protected sets must not be used for training, model selection or qualitative inspection.
`hvp.provenance.assert_dataset_purpose` and holdout access logs enforce the intended operation.
The repository does not distribute large arrays through Git.

## Labels

- camera/action track;
- visibility and time-since-visible geometry;
- answerable and foreground-nonempty populations;
- generator-persistent object metadata where applicable.

## Appropriate claims

The data support controlled statements about addressable reach, fidelity loss and self-fed
robustness under the registered generator. Exact geometric observability is not
information-theoretic necessity, and LoopMom is not evidence of natural-video generalization.

## Ethical considerations

LoopMom is synthetic and contains no personal data. MNIST glyph sources retain their original
dataset licensing/terms; official train/test glyph separation is required for the persistent-object
Moving-MNIST boundary.
