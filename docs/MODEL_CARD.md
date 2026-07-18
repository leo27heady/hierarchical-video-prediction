# Model Card: Budgeted Hierarchical Video Memory

## Intended use

The model family is a research instrument for studying how a fixed context-token budget should be
allocated across spatial resolution, temporal rate and recent detail in long-horizon video
prediction. It is not intended as a general-purpose video generator or a safety-critical world
model.

## Architecture

A frozen two-level continuous convolutional codec provides coarse and fine latent tokens. A shared
transformer predictor receives context packages with explicit coarse/fine roles and predicts a
four-frame target window using diffusion or regression heads. The principal hierarchical packages
store a 48-frame coarse history plus a short fine tail; flat controls spend the same context budget
on contiguous full-resolution latent history.

## Measured behavior

- On untouched LoopMom confirmation data, hierarchy is confirmed at 896 and 1,152 context tokens;
  1,792 is unresolved; flat full-resolution context is confirmed at 2,432 and 3,072.
- The confirmed hierarchical packages provide the same nominal 48-frame reach with 2.7–3.4 times
  fewer context tokens than 48 fine frames.
- Sampler/NFE and independent-codec studies preserve all registered point-estimate directions,
  with disclosed precision limits at the upper inversion.
- On persistent-object long Moving-MNIST, hierarchy better preserves support/continuity, while flat
  context better preserves digit identity.

## Limitations

The hierarchy is not universally superior. Compression loses detail, the B1 stride-2 result
changes the target interface, regression package effects are seed-heterogeneous, the proposed
general late trust boundary did not replicate, and the evidence does not establish natural-video
generalization or an information-theoretic necessity.

## Evaluation

Use complete decoded-frame metrics and persistent-object endpoints from `hvp.evaluation`; pair
records by clip and use `hvp.statistics`. Do not rely on aggregate foreground MSE alone.
