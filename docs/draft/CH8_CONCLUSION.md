# Chapter 8 — Conclusion

> **Status (2026-07-08):** journal-register prose, content-complete; all numbers measured except
> the Ch. 5 [EXPECTED] transfers (generalization campaign) and the R-C/R-D robustness reads.
> R-A + R-B landed 2026-07-08: BOTH band edges (896 hier-win, 3,072 flat-win) are TWO-SEED.

## 8.1 One claim

**The coarse level of a hierarchical latent video model earns its token budget exactly when the
prediction horizon exceeds what the affordable flat context can carry — and the budgets at which
this happens are predictable from geometry.** On a testbed whose memory demand is an exact label,
the hierarchy-versus-flat frontier, measured at four matched budgets, has the shape the mechanism
account predicts: flat wins below the span threshold (≈500 tokens: no hierarchical configuration
can cover the ~48-frame loop; −0.0036 [−0.0063, −0.0010]); the hierarchy wins the band from the
geometric threshold — 48×16 coarse tokens plus a minimal fine tail ≈ 900 — where the effect is
largest (+0.0399 [+0.0299, +0.0501] seed 0 and +0.0358 [+0.0267, +0.0454] seed 1 at 896 tokens —
two seeds, every horizon × hardness cell on both) through the two-seed headline at 1,152
(+0.0189 / +0.0354); and flat wins again at window saturation (3,072 tokens, −0.0131 [−0.0201, −0.0069] seed 0,
−0.0107 [−0.0163, −0.0058] seed 1 — two seeds), where its full-resolution memory
information-dominates any coarse record of the same span.

## 8.2 Boundary conditions (each one measured)

- **Emergence:** retrieval-from-memory needs a large optimization budget — null at 6k steps, a
  two-seed CI-clean win at 20k (+0.0135/+0.0143 on hard re-entry; each budget its own complete
  annealed schedule; the 20k extension was a disclosed post-null follow-up, twice-replicated) —
  so short-budget ablations spuriously kill memory mechanisms.
- **Mechanism-agnosticism:** the matched-budget win replicates under the regression head
  (reg-long > reg-flat in every bin at both seeds, ALL +0.0334 s0 / +0.0501 s1) — it is a
  property of the
  memory packaging, not of the diffusion core.
- **Regime-dependent heads:** diffusion wins single-step best-of-N coverage on every column
  (0.0512 vs 0.0664); regression wins the deterministic best-of-1 rollout at equal sharpness
  (ALL CI [+0.0110, +0.0227]) — a pre-registered expectation refuted and reported as a reversal.
  No "one best head" statement survives.
- **Spatial abstraction is economic, not intrinsic:** isolated from the window confound, coarse
  history costs the diffusion head −0.0082 at 42% token savings and is statistically free for
  the regression head; its real value is the window the saved tokens buy (−0.027).
- **The trustworthy-memory boundary:** at a second operating point with a ~213-frame loop,
  single-step retrieval over strided slot memory is real and span-threshold-gated (2/2 span>loop
  runs CI-positive vs 0/4 below), but under pure self-feeding the slot contents degrade and the
  advantage inverts after ~24 frames (all pre-registered rollout gate cells null) — hierarchical
  memory pays where its contents remain trustworthy: continuously refreshed by the recent window,
  or near a ground-truth anchor.
- **The negative control:** on standard Moving MNIST — no off-screen re-entry, hence no memory
  horizon — flat ≈ hierarchical (180.91 vs 181.29 aggregate MSE), exactly as the thesis predicts
  on a benchmark that cannot reward memory.

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
horizon is covered, and only then on resolution.** At the tightest budgets that cover the
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
between loop 48 (works) and loop 213 (fails) the trustworthy-memory boundary sits.
