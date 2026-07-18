# Code Architecture and Stability Contract

## Public layer

`hvp` is the supported, side-effect-free interface:

- `hvp.context` — typed context and target interfaces plus geometric reach;
- `hvp.evaluation` — complete decoded-frame metrics, persistent-object Hungarian assignment and
  pair-specific reach populations;
- `hvp.statistics` — clip-cluster paired effects, head × package contrasts and simultaneous
  max-statistic intervals;
- `hvp.provenance` — content-addressed run identity, atomic artifacts, protected-set access logs
  and RNG-complete checkpoints;
- `hvp.experiments` — outcome-independent launch authorization;
- `hvp.cli` — read-only command-line inspection.

Importing the public package does not load a dataset, initialize CUDA, access W&B or start a run.

## Frozen implementation

`hvp_rev` contains the implementation that produced the registered evidence. It includes early
prototypes, terminal controllers and a large legacy trainer. Its scientific value is exact
provenance, not API elegance. A direct structural rewrite would risk changing:

- batch and target construction;
- random-number consumption;
- optimizer/update and EMA trajectories;
- diffusion sampling;
- checkpoint compatibility;
- artifact hashes.

For that reason, release refactoring uses adapters and parity tests. A future v1 implementation may
split the trainer into data, model, training and evaluation components only after frozen-batch and
first-update numerical parity is demonstrated for every retained regime.

## Experiment controllers and scripts

`scripts/` contains dataset generation, launch control, analysis and manuscript generation.
`experiments/configs/` contains frozen scientific configurations.
`experiments/authorization.json` is the launch authority; a controller must fail closed when a
study is terminal, skipped, unregistered or missing an attestation.

## Data and artifacts

Large data and checkpoints are intentionally excluded from Git. Immutable run directories encode
the study, arm, seed and `RunSpec` identity. Final evidence hashes are recorded under `artifacts/`
and in the release evidence manifest.

## Stability levels

| Surface | Status | Compatibility promise |
|---|---|---|
| `hvp.*` | Public | Semantic compatibility within 0.3.x |
| `hvp_rev.evaluation`, `analysis`, `provenance` | Frozen core | Covered by parity/public-API tests |
| Other `hvp_rev.*` modules | Research internal | Artifact compatibility takes priority |
| `scripts/` | Workflow | Frozen configurations and gates are authoritative |
