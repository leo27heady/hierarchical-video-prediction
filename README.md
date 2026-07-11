# Hierarchical Video Prediction

This repository studies a controlled question: when a video predictor has a bounded context
budget, how should that budget be packaged across temporal rate, spatial resolution, and depth?

The experimental design manipulates **context packaging**.  The results are interpreted through
three deliberately separate axes:

- **reach** — whether the context contains the memory-relevant past;
- **fidelity** — how much relevant information survives the representation and token budget;
- **trust** — whether that information remains usable during repeated self-fed generation.

The project is exploratory where mechanisms or boundaries are being located and confirmatory only
where a frozen analysis is evaluated on an untouched set.  Measured configurations are not treated
as universal laws, and negative results remain part of the evidence.

## Current scientific status

- Existing looping-video experiments provide strong development evidence for reach/fidelity
  trade-offs and tested package inversions.
- Pair-specific reach bands are mechanistically strong but post-hoc on the current development
  pool; they are preregistered for untouched confirmation.
- The B1 stride-2 result is a replicated **configuration** result because the legacy model couples
  context and target stride.  A fixed-target-interface control is prepared but not authorized.
- The legacy Phase C identity/trajectory evaluator is retired from claim production.  Phase C v2
  requires persistent object metadata, one-to-one assignment, coverage, and validated margins.
- Trust is presently a measured failure boundary; it requires replication or horizon dose-response
  before being described as a general principle.

The governing plan is
[`docs/MASTER_IMPLEMENTATION_PLAN_REACH_FIDELITY_TRUST_2026-07-11.md`](docs/MASTER_IMPLEMENTATION_PLAN_REACH_FIDELITY_TRUST_2026-07-11.md).

## CPU validation

The default suite never requires local datasets or a GPU:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\venv\Scripts\python.exe -m pytest -q
```

It validates complete frame metrics, Hungarian object assignment, reach-band derivation,
persistent Moving-MNIST metadata, immutable artifacts, RNG restoration, experiment authorization,
and CPU-safe model plumbing.

## Experiment safety

`experiments/authorization.json` is the launch authority.  A new experiment fails closed unless:

1. its status is launchable;
2. its protocol/configuration exists;
3. every required CPU or protocol marker exists;
4. one execution owner is named in `docs/EXPERIMENT_CONTROL.md`.

Check a proposed run without launching it:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\venv\Scripts\python.exe scripts\check_experiment_gate.py phase-c-v2
```

The legacy Phase C study is explicitly prohibited.  The v2 launcher is intentionally disabled
until its dataset, classifier, evaluator, and preregistration gates are complete.

## Reproduction tiers

- **Tier 1 — CPU semantics:** install `requirements-cpu.txt`, then run pytest.
- **Tier 2 — artifact analysis:** retrieve hash-pinned NPZs/checkpoints from the evidence archive
  (archive publication remains a release task) and regenerate tables/figures.
- **Tier 3 — training:** use a recorded `RunSpec`, dataset/AE hashes, explicit resume, and the
  authorized experiment configuration.  Legacy runs are preserved but are not trajectory-exact
  because their checkpoints omit RNG and sampler state.

New datasets and runs must be stored with manifests.  Smoke runs and qualitative strips must use
run-specific directories and may not overwrite canonical artifacts.

## Repository map

- `hvp_rev/evaluation/` — complete frame/object/reach metrics.
- `hvp_rev/data/` — versioned persistent-object synthetic datasets.
- `hvp_rev/provenance.py` — `RunSpec`, atomic artifacts, RNG-complete checkpoints, holdout access.
- `hvp_rev/arm_registry.py` — typed package and target interfaces, including fixed-target controls.
- `experiments/` — authorization ledger and frozen/prepared experiment configurations.
- `docs/draft/` — fourth-article source material.
- `docs/dissertation/rewrite/` — dissertation source; its approved topic and three existing article
  strands are preserved.
- `docs/CLAIM_LEDGER.md` — scientific source of truth.

## Historical code

The current GPU queue uses the legacy `hvp_rev/rhvd.py` path.  It is being kept byte-stable until
the already-running pre-Phase-C queue finishes so later seed-2/c2 cells remain comparable with
earlier cells.  New validated components live alongside it and will become the default only after
the queue handoff and artifact freeze.
