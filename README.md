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

The central result is conditional rather than promotional: **hierarchy buys reach; flat context
buys fidelity**. In the frozen latent interface, 48 fine frames cost 3,072 context tokens, while
48 coarse frames plus a two- or six-frame fine tail cost 896 or 1,152. The confirmed hierarchical
packages therefore supply the same nominal reach with 2.7–3.4 times fewer context tokens, but flat
full-resolution context wins at the tested upper budgets.

## Current scientific status

- The untouched LoopMom confirmation verifies hierarchy-favoring lower/interior budgets,
  comparator-derived reach bands and flat-favoring upper-budget inversions. The family-wise
  confirmed tested bracket is 1,152 to 2,432 context tokens; no continuous crossing is inferred.
- The five-seed diffusion package-shape result is stable, while regression and the cross-head
  interaction are training-seed-heterogeneous.
- The B1 stride-2 result is a replicated **configuration** result because the legacy model couples
  context and target stride.  A fixed-target-interface control is prepared but not authorized.
- The seed-0 late self-fed inversion did not replicate at seeds 1–2. Trust is therefore a
  robustness question, not an established boundary or title-level law.
- Legacy Phase C object conclusions remain retired. Corrected E6 completed with persistent
  metadata, official glyph separation, Hungarian assignment, miss/extra accounting and protected
  evaluation. Its verdict is `MIXED_BOUNDARY`: hierarchy better preserves spatial support and
  continuity, while flat context better preserves digit identity.
- E7c and E7a establish directional sampler/NFE and whole-codec robustness with disclosed
  upper-inversion precision limits. E8a is a privileged exact-pose raw-memory reference, not a
  token-matched competitor.

| Tested context budget | Untouched simultaneous verdict |
|---:|---|
| 896 | hierarchy confirmed |
| 1,152 | hierarchy confirmed |
| 1,792 | unresolved |
| 2,432 | flat fidelity confirmed |
| 3,072 | flat fidelity confirmed |

The current content-finalization plan and claim matrix are
[`docs/CONTENT_FINALIZATION_EXECUTION_PLAN_2026-07-19.md`](docs/CONTENT_FINALIZATION_EXECUTION_PLAN_2026-07-19.md)
and
[`docs/CONTENT_FINALIZATION_CLAIM_MATRIX_2026-07-19.md`](docs/CONTENT_FINALIZATION_CLAIM_MATRIX_2026-07-19.md).
The completed implementation and honest remaining-work audit are recorded in
[`docs/CONTENT_FINALIZATION_REPORT_2026-07-19.md`](docs/CONTENT_FINALIZATION_REPORT_2026-07-19.md).
The binding dissertation/article narrative and English-authoring policy are recorded in
[`docs/RESOURCE_EFFICIENT_LONG_HORIZON_VISION_2026-07-18.md`](docs/RESOURCE_EFFICIENT_LONG_HORIZON_VISION_2026-07-18.md).

## CPU validation

The default suite never requires local datasets or a GPU:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\venv\Scripts\python.exe -m pytest -q
```

It validates complete frame metrics, Hungarian object assignment, reach-band derivation,
persistent Moving-MNIST metadata, immutable artifacts, RNG restoration, experiment authorization,
and CPU-safe model plumbing.

## Public Python API

The supported `hvp` layer is side-effect free and wraps the frozen experiment implementation:

```python
from hvp.context import ArmSpec, ContextGeometry
from hvp.evaluation import evaluate_frames, pair_specific_reach_mask
from hvp.statistics import paired_effect

hier = ArmSpec(name="hier-min", coarse_context_slots=48, fine_recent_slots=2)
assert hier.context_tokens == 896
assert hier.context_reach == 48
```

Check an experiment authorization without launching anything:

```powershell
.\venv\Scripts\hvp.exe gate counterfactual-loopworld-e9 `
  --ledger experiments\authorization.json --root .
```

The command returns a non-zero status because E9 is terminally skipped. See
[`docs/CODE_ARCHITECTURE.md`](docs/CODE_ARCHITECTURE.md) for the compatibility contract.
The installable distribution also carries the frozen `hvp_rev`, `hvp_blt` and `hvp_dlh`
namespaces because the adapters and reproduction paths depend on them; their presence in the wheel
does not make them supported public APIs.

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

The legacy Phase C study is explicitly prohibited. Phase C v2 is terminal and frozen; no adaptive
arm, seed or endpoint may be added. A new study requires a new authorization entry and protocol.

## Reproduction tiers

- **Tier 1 — CPU semantics:** install `requirements-cpu.txt`, then run pytest.
- **Tier 2 — artifact analysis:** retrieve hash-pinned NPZs/checkpoints from the evidence archive
  (archive publication remains a release task) and regenerate tables/figures.
- **Tier 3 — training:** use a recorded `RunSpec`, dataset/AE hashes, explicit resume, and the
  authorized experiment configuration.  Legacy runs are preserved but are not trajectory-exact
  because their checkpoints omit RNG and sampler state.

New datasets and runs must be stored with manifests.  Smoke runs and qualitative strips must use
run-specific directories and may not overwrite canonical artifacts.

External datasets and reference checkouts are not assumed to live at developer-specific paths.
Set `HVP_MNIST_GZ`, `HVP_FMNIST_GZ`, `HVP_OPENSTL_TEST`, `HVP_CWVAE_TEST`,
`HVP_CWVAE_TRAIN`, `HVP_TAXIBJ_NPZ`, `OPENSTL_REPO`, or `EBT_REPO` as required by the
corresponding bridge. Repository-relative defaults point under `data/external/`.

## Repository map

- `hvp/` — stable public API and read-only CLI.
- `hvp_rev/evaluation/` — complete frame/object/reach metrics.
- `hvp_rev/data/` — versioned persistent-object synthetic datasets.
- `hvp_rev/provenance.py` — `RunSpec`, atomic artifacts, RNG-complete checkpoints, holdout access.
- `hvp_rev/arm_registry.py` — typed package and target interfaces, including fixed-target controls.
- `experiments/` — authorization ledger and frozen/prepared experiment configurations.
- `docs/draft/` — archived development and detailed fourth-article source material.
- `docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md` — authoritative fourth-article source.
- `docs/dissertation/rewrite/` — dissertation source; its approved topic and three existing article
  strands are preserved.
- `docs/CLAIM_LEDGER.md` — scientific source of truth.
- `docs/MODEL_CARD.md` and `docs/DATA_CARD_LOOPMOM.md` — scope and responsible-use cards.

## Evidence and manuscript regeneration

The binding article source is
`docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md`. Regenerate the synchronized evidence table,
claim/artifact index, and assembled dissertation with:

```powershell
.\venv\Scripts\python.exe scripts\make_rft_final_tables.py
.\venv\Scripts\python.exe scripts\make_rft_result_figures.py
.\venv\Scripts\python.exe scripts\assemble_final_manuscripts.py
```

Assembly validates the authoritative article in place and writes the bound dissertation to
`docs/dissertation/final/DISSERTATION_MANUSCRIPT.md`. Administrative facts that only the applicant
or institution can supply are isolated in `docs/dissertation/AUTHOR_INPUT_CHECKLIST.md`.

The DOCX and PDF files dated 18 July 2026 are historical review renders and do not include the
19 July content refactor. They are neither institutional nor journal-template submissions.
Run the CPU scientific/source gate and wheel build with:

```powershell
.\venv\Scripts\python.exe scripts\release_check.py --wheel-dir output\wheel
```

The historical page-level and accessibility record is
[`docs/RELEASE_QA_REPORT_2026-07-18.md`](docs/RELEASE_QA_REPORT_2026-07-18.md).
It does not certify a current institutional render. Source finalization is tracked separately in
the 19 July execution plan.

The canonical execution/status ledger is
`docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md`.
