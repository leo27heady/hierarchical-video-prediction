# Chapter 8 — Conclusion

> **Status (2026-07-08):** journal-register prose, content-complete; all numbers measured except
> the Ch. 5 [EXPECTED] transfers (generalization campaign) and the R-C/R-D robustness reads.
> R-A + R-B landed 2026-07-08: BOTH band edges (896 hier-win, 3,072 flat-win) are TWO-SEED.

## 8.1 One claim

**The coarse level of a hierarchical latent video model earns its token budget exactly when the
prediction horizon exceeds what the affordable flat context can carry — and the budgets at which
this happens are predictable from geometry.** (The packaging is not unique: §4.4.3 measures a
temporally-subsampled full-resolution packaging of the same span — itself a two-rate temporal
hierarchy in the clockwork sense — that matches or exceeds it at
seed 0, seed-1 replication queued, on a static-world testbed maximally kind to temporal
subsampling. The claim is about long-span context at matched budget, with
the coarse level as its per-level-decodable, multi-resolution form.) On a testbed whose memory demand is an exact label,
the hierarchy-versus-flat frontier, measured at six matched budgets, has the shape the mechanism
account predicts: flat wins below the span threshold (≈500 tokens: no hierarchical configuration
can cover the ~48-frame loop; −0.0036 [−0.0063, −0.0010]); the hierarchy wins the band from the
geometric threshold — 48×16 coarse tokens plus a minimal fine tail ≈ 900 — where the effect is
largest (+0.0399 [+0.0299, +0.0501] seed 0 and +0.0358 [+0.0267, +0.0454] seed 1 at 896 tokens —
two seeds, every horizon × hardness cell on both) through the two-seed headline at 1,152
(+0.0189 / +0.0354); and flat wins again at window saturation (3,072 tokens, −0.0131 [−0.0201, −0.0069] seed 0,
−0.0107 [−0.0163, −0.0058] seed 1 — two seeds), where its full-resolution memory
information-dominates any coarse record of the same span.

## 8.2 The empirical laws, each under stated scope

Every law below is scoped by construction — testbed, head, representation, seed level — with the
single source of truth in the claim ledger (`docs/CLAIM_LEDGER.md`); a law appears in this
section only when its evidence row is complete, and its wording carries its scope.

- **L1 — Span-threshold law.** Long-span packaging wins only once its span covers the task's
  memory horizon while flat context cannot: flat wins at 512 tokens (two-seed), hierarchy from
  896 (two-seed, every cell, +0.0399/+0.0358), the switch-on at the token-arithmetic prediction.
  Scope: loopmom; edges measured under the diffusion head (regression confirms the 1,152
  interior pair, not the edges).
- **L2 — Saturation law.** Once flat full-resolution context covers the same span, the
  abstraction cost can dominate: flat wins at 3,072 (two-seed, −0.0131/−0.0107), and the
  flat/hier crossover is measured to lie in (1,792, 2,432). Same scope as L1.
- **L3 — Family-edge law.** The band's lower edge is a constant of the *family*, not the
  testbed: halving the coarse tick moved it from 896 to 512 tokens (+0.0330 [+0.0246, +0.0419],
  every cell, span dt-signature), at the arithmetic prediction both times; each family carries
  its own plateau level (stride-2 ≈ +0.017 above stride-1). Status: seed 0; seed-1 replication
  queued before headline use.
- **L4 — Shape law (resolving).** Spatial coarsening and temporal sparsification are not
  interchangeable: at this operating point, strided full-resolution history beats the coarse
  hierarchy under the diffusion head (+0.0197 [+0.0143, +0.0252], seed 0; seed-1 required and
  queued; regression twins queued — the head interaction is the open half). Scope: one
  static-world setting (maximally kind to temporal subsampling), this autoencoder's 4×4 code;
  data-dependence untested (a rate-stress suite is future work).
- **L5 — Depth hypothesis (allocation probe, not a law).** Depth should pay only where the data
  contain another scale to own; single-scale loopmom can measure only the null branch, and the
  three-stage probe is a depth-plus-objective-allocation experiment (loss-dilution disclosed).
  A depth *law* requires the multi-timescale testbed of §8.5.
- **L6 — Trustworthy-memory law.** A span sufficient for single-step retrieval can fail under
  self-fed rollout: at the ~213-frame-loop operating point, retrieval is real and
  span-threshold-gated (2/2 vs 0/4) yet inverts after ~24 self-fed frames — hierarchical memory
  pays where its contents remain trustworthy.

**Supporting boundary conditions (measured):** retrieval *emerges* only under a sufficient
optimization budget (null at 6k steps, two-seed win at 20k; matched-budget delta doubles at 2×
budget — short-budget ablations spuriously kill memory mechanisms); the matched-budget win is
mechanism-agnostic at the interior point (both heads, two seeds); the generation head is
regime-dependent (diffusion wins coverage, regression wins best-of-1 rollout; no "one best
head" survives); and the negative control holds — on standard Moving MNIST, with no memory
horizon, flat ≈ hierarchical (180.91 vs 181.29), exactly as the thesis predicts. [EXPECTED-
PhaseC: the null extends to ~300-frame rollouts with identity/trajectory metrics per the
pre-registered mandate.]

## 8.3 The two honest negatives that scope the claim

(i) **Energy-based descent** is a two-sided characterized negative at 16 GB-reachable scale:
one published recipe yields a stably mis-ordered landscape (21/21 probes); the other repairs the
ordering (15/15 from step 6k) while gradient descent — including the paper's own inference
recipe — still fails to reach structured outputs. Ordering and reachability dissociate; the
configuration gap to the published setting is tabulated. (ii) **Reversibility** delivers exact
gradients at constant memory versus BPTT (OOM at K=120 vs ≈1.1–1.2 GB through K=160) but is
~Pareto with checkpointing on the temporal axis and provably cannot pay on the
inner-optimization axis (0.26% of peak, measured; ≈0.6–1.0% analytic band) — and invertibility
does not imply stability (perturbation growth 2.7e4× at Lip(F) ≤ 1).

## 8.4 Practical reading

For a practitioner the frontier reads as a sizing rule: **spend tokens on span until the memory
horizon is covered — using the cheapest abstraction family that covers it — and only then on
resolution and recency; add depth only when the data has scales left to own.** At the tightest budgets that cover the
horizon, the hierarchical packaging delivers the largest gains (a quality that flat context
reaches only at ≈2.3–3× the budget, with quadratic attention making the compute gap larger
still); once the budget affords the full horizon at full resolution, the hierarchy's advantage
inverts into a small abstraction cost. The cheapest strong recipe measured is regression head +
coarse history + long window (reg-long: 0.1682 / 0.1503 across two seeds, mean 0.1593, at 1,152
tokens — at seed level the best absolute rollout number of the campaign).

## 8.5 Future work

Four directions follow from the measured boundaries rather than from speculation: (1) paper-scale
EBT — the ordering-repaired branch leaves scale as the open variable; (2) distribution-matching
rollout objectives (Self-Forcing-class) whose long-horizon training memory the store-the-noise
recurrence makes affordable — the direct route to pushing the trustworthy-memory boundary
outward; (3) a natural-video instantiation of the re-entry-stratified protocol (hardness labels
estimated rather than exact), building on the OpenSTL bridge; (4) a sharp renderer over the
frozen latents (the adversarial-renderer precedent) to lift the decoder-owned blur ceiling that
bounds all arms equally. A fifth, cheaper item is measurement, not construction: mapping where
between loop 48 (works) and loop 213 (fails) the trustworthy-memory boundary sits. A sixth
follows from the shape study's depth null [EXPECTED-B3]: the arena where depth *can* win is a
**multi-scale testbed with a visible slow factor** — a slowly evolving, always-visible global
variable (period ≫ the pan loop) layered under the fast re-entry dynamics. Slow-but-visible
state is continuously refreshed, so a deep hierarchy's top level can own it without fighting the
trustworthy-memory boundary that self-fed occluded recall runs into; this is the CW-VAE slowness
regime rather than the retrieval regime, and the natural home for the re-based factor-4 pyramid
(16×16 / 4×4 / 1×1) that a 16 GB budget cannot pair with long flat windows.
