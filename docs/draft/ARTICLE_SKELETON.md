# ARTICLE SKELETON — historical source map

> **Canonical fourth-article structure changed 2026-07-11.** Use
> `ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md` for the submission narrative. This file remains a
> source/evidence map so measured material is not lost. Older law, exact-edge and Phase-C wording
> here does not override `docs/CLAIM_LEDGER.md`.

> **Spine:** *Hierarchy earns its keep where the memory horizon demands it.* A hierarchical latent
> video diffusion model whose coarse level carries long-range memory, studied with controlled
> mechanism science on a purpose-built memory-horizon testbed; reversibility homed as the
> constant-memory enabler theorem (Ch. D); the EBT study as an honest, two-sided negative.
>
> Status date: 2026-07-07 — **ALL CHAPTERS NOW EXIST AS PROSE**; this skeleton is the working
> master for tables/gates only. Canonical order (fixed in CH1_INTRO.md): Ch1 `CH1_INTRO.md` ·
> Ch2 `CH2_TESTBED.md` · Ch3 `CH3_MECHANISM.md` · Ch4 `CH4_RESULTS.md` (frontier §4.3.6 +
> pair-closure §4.5.6 integrated; BRANCH markers resolved) · Ch5 `CH5_GENERALIZATION.md` ·
> Ch6 `CH6_REVERSIBILITY.md` · Ch7 `CH7_RELATED.md` · Ch8 `CH8_CONCLUSION.md`. The abstract in
> §0.1 below is SUPERSEDED by the finalized abstract in CH1_INTRO.md (adds the frontier band).
> Title resolved to candidate 2 per the red-team rule (pair closed RED). Every table below is
> FILLED with verified numbers (source named); T7 measured 2026-07-06. Status ledger: §S.
> **Bibliography:** the article cites a subset of the MASTER reference list
> `docs/dissertation/rewrite/07_REFERENCES.md` (stable keys + binding table; built 2026-07-07
> from the full [REF:] inventory of both texts) — resolve inline [REF:] tags from it at the
> venue-format pass; [VERIFY]-flagged entries checked against PDFs per the CH7 §7.6 debt.

---

## 0. Title candidates

1. **Hierarchy Earns Its Keep Where the Memory Horizon Demands It: Mechanism, Memory and
   Reversibility in Hierarchical Latent Video Diffusion**
2. **When Does Hierarchy Pay? A Memory-Horizon Study of Hierarchical Latent Video Prediction at
   Matched Token Budgets**
3. **Long Coarse Memory: Controlled Mechanism Science for Hierarchical Video Diffusion, with an
   Exactly Invertible Route to Long-Horizon Training**

> **Red-team recommendation (2026-07-03, `docs/draft/REDTEAM_REVIEW_2026-07-03.md`):** candidate 1
> over-claims relative to the measured matrix (flat-long is the outright quality frontier;
> regression wins the best-of-1 rollout). **Null/efficiency-branch title = candidate 2 — "When
> Does Hierarchy Pay? A Memory-Horizon Study … at Matched Token Budgets"** — the claim the current
> data defends. A capability-branch title (to be drafted) becomes available only if the H-grid
> experiment (`docs/HGRID_DESIGN.md`) delivers a capability win; until that outcome, both branch
> titles stay listed and candidate 1 is retained for reference only.

## 0.1 Abstract (draft, ~180 words)

Hierarchical latent video models are usually motivated by efficiency, yet controlled evidence that
the hierarchy itself — rather than added context — improves prediction is scarce. We build a
memory-horizon testbed in which content verifiably exits a panning viewport and re-enters ~48
frames later at an analytically known position, so that "hard because it must be remembered" is an
exact label. At matched capacity and compute we compare prediction mechanisms (regression,
diffusion, energy-based descent): regression averages over futures, diffusion commits sharply, and
Energy-Based Transformers dissociate into a two-sided result — the S1 recipe learns a correctly
*ordered* energy landscape, yet descent never *reaches* a structured output at our scale. We then
show the headline: a two-level latent diffusion model whose coarse level carries a 48-frame memory
beats a token-budget-matched flat model precisely where the horizon demands it — no advantage
before frame 24, a CI-excluding-zero win on hard re-entry frames from horizon 25 onward over
rollouts of 152 autoregressive frames after a 48-frame ground-truth warmup. The benefit is
mechanism-agnostic — it replicates when both arms use a regression predictor — but the mechanism
ranking is regime-dependent: regression, not diffusion, wins the deterministic best-of-1 rollout
at equal sharpness (diffusion's advantage is a single-step best-of-N coverage result), and a
full-resolution flat context at 2.7× the token cost is the outright quality frontier, making the
hierarchy an efficiency point rather than a capability necessity in this regime. Finally, we
prove and measure an exactly invertible stochastic recurrence
(gradients equal to BPTT, memory constant in rollout length) and honestly delimit where
reversibility pays against checkpointing.

---

## 1. Chapter 1 — Introduction and dissertation arc

- **Opening claim + question.** Hierarchical video prediction has repeatedly produced nulls when
  evaluated on standard benchmarks (our own CW-VAE/DLH line: hierarchy null on MMNIST, temporal
  clock a capacity dump). Thesis: the null is a property of the *testbed*, not the mechanism — no
  standard synthetic benchmark has a regime that is hard *without* memory and computable *with* it.
- **Dissertation arc paragraph.** Article 1 (TempVerseFormer, Kulyk, Springer NCA 2025) built
  reversible transformers along time for deterministic dynamics; this article extends the program
  to *stochastic, hierarchical* video: (i) it measures the conditions under which hierarchy pays,
  and (ii) it extends time-axis reversibility to the stochastic case as a correctness/enabler
  theorem. Pre-written null-branch framing (survives any outcome): *article 1 proposed the
  machinery; this article measures the conditions under which it pays* — so even a null is a
  boundary-condition result, not a self-contradiction.
- **The three moves of the article.**
  1. A purpose-built **memory-horizon testbed** with exact hardness labels (Ch. 2): a momentum
     action walk over a toroidal world, $x_t = (x_{t-1} + 4a_t) \bmod 192$, loop period ≈ 48
     frames, 67% of the world off-screen at any instant.
  2. **Mechanism science at matched capacity** (Ch. 3): regression / diffusion / energy descent on
     one frozen latent interface and one harness; the EBT part is two-sided (landscape ordering ×
     descent reachability) and scale-scoped.
  3. **The system result** (Ch. 4): hierarchy vs. a token-budget-matched flat control over
     rollouts of 152 AR frames after a 48-frame GT warmup, stratified by horizon × analytic
     hardness — the win appears exactly where the memory horizon demands it, and nowhere else.
  4. Supporting theory (Ch. 5): the invertible stochastic recurrence — exact gradients at
     $O(1)$ activation memory in rollout length $K$ — plus the honest three-axis account of where
     reversibility cannot pay.
- **Contributions list (C1–C5).**
  - C1: the testbed + exact re-entry-stratified evaluation protocol (analytic hardness, single-GPU).
  - C2: matched-capacity mechanism comparison with the action-conditioning corrective; controlled
    replication of "regression averages / diffusion commits" (T1).
  - C3: a two-sided, official-code, fair-scale EBT capability study — ordering repaired under S1,
    reachability still failing; the config-gap table making extrapolation risk explicit (T2).
  - C4: the memory-horizon headline — long-coarse hierarchy beats token-matched flat exactly at
    hard re-entry beyond horizon ~24 (T3–T5); emergence of retrieval only after ~20k training steps.
  - C5: Ch. D — provably invertible stochastic recurrence (store-the-noise), grad-cos 1.000000 vs
    BPTT, measured memory frontier; invertibility ≠ stability (Lipschitz characterization); the
    three-axis reversibility taxonomy with the inner-loop no-go accounting (T6).
- **Reader's map + format note** (journal ~10–12 pp: Ch. 3 Part B and Ch. 5 compress to sections;
  dissertation: full chapters).

## 2. Chapter 2 — The memory-horizon testbed and evaluation protocol

**FULL DRAFT EXISTS — `docs/draft/CH2_TESTBED.md`. One-line pointers only:**
2.1 why purpose-built · 2.2 momentum looping world (torus, $p_{hold}≈0.10$, $p_{cont}≈0.90$,
loop ≈ 48) · 2.3 datasets (3000×100 train / 500×200 test) · 2.4 frozen two-level conv-AE latent
interface · 2.5 foreground-masked decoded metrics + exact hardness labels · 2.6 statistical
protocol (paired bootstrap, CRN, 2-seed rule) · 2.7 floors/ceilings (persistence, AE round-trip)
· 2.8 why aggregate pixel benchmarks are excluded (background dominance) · 2.9 limitations.

## 3. Chapter 3 — Prediction mechanisms at matched capacity (+ two-sided EBT study)

**FULL DRAFT EXISTS — `docs/draft/CH3_MECHANISM.md`. One-line pointers only:**
3.1 question/design · 3.2 shared harness (7.3–7.5M arms, best-of-N, NFE-matched) · 3.3 latent
standardisation corrective · 3.4 Part A: regression averages, diffusion commits (Table T1) ·
3.5 Part B: two-sided EBT study — official model imported unchanged; S2 = stably mis-ordered
landscape; S1 = ordering repaired, generation still unreachable; decomposition ordering ×
reachability; scope/non-claims + config-gap Table 3.3 (= T2b); diffusion control at same scale ·
3.6 limitations.

## 4. Chapter 4 — The system and the memory-horizon result (HEADLINE)

- **4.1 The model (RHVD-mini).** Two-level latent diffusion: coarse tokens 16×4×4 (1 token/frame
  ×16 dim… per Ch. 2 AE), fine 16×8×8; DiT denoiser d=384, depth 8, heads 6, 22.19M params;
  v-prediction, 16 DDIM steps; per-token roles/frame-ids/actions; 4-consecutive-frame prediction
  windows so train context layout == rollout context layout (design: `docs/STAGE2_DESIGN.md`).
  Training: 30k steps, B=16, bf16, EMA 0.999, constant LR + 2k warmup, atomic ckpt/resume.
- **4.2 Pre-test: the single-step memory gate (T3).** Does a long *coarse* context help at all,
  before building rollout machinery? Design: coarse-context 48 vs 6 at fixed fine context 6
  (1232 vs 560 tokens), stratified by analytic novel-content hardness. **Result: emergence with
  training** — clean null at 6k steps (both seeds), decisive hard-bin WIN at 20k steps (both
  seeds, CI excluding 0). Framing: retrieval-from-memory is a *late* capability; underscores why
  short-budget ablations in the literature can spuriously kill memory mechanisms.
- **4.3 Arms and controls (pre-registered, `docs/STAGE2_DESIGN.md`).** hier-long (48c+6f = 1152
  ctx tokens) — THE system; hier-short (6c+6f = 480) — memory ablation; flat-budget (18f = 1152) —
  the token-budget-matched flat control that makes the claim "hierarchy", not "long context";
  reg-long — regression-rollout control (H2; outcome: reversal, §4.6); reg-flat (18f = 1152,
  regression) — mechanism-agnosticism control (red-team P0, added post-registration);
  flat-long (48f = 3072 tokens ≈ 2.7×) — token-cost framing arm; persistence + AE round-trip
  floors/ceilings.
- **4.4 Headline result (T4).** 152 AR frames after a 48-frame GT warmup (38 steps × 4 frames),
  best-of-1, paired per-frame fg-MSE, horizon × hardness matrix, clip-level cluster-bootstrap CIs,
  empty-GT-fg records (21.5%) excluded from bins. hier-long vs flat-budget: **null at every bin
  below horizon 25; WIN with CI excluding zero from h25-60 onward on every easy/HARD bin**
  (hard re-entry h25-60: +0.0344 [+0.0162,+0.0531]; h61+ HARD: +0.0314 [+0.0202,+0.0431]; the
  sparse h25-60 mid cell, n=79, is null). Window ordering (ALL-row fg-MSE, final matrix):
  flat-long 0.1564 < reg-long 0.1682 < hier-long 0.1848 < reg-flat 0.2015 < flat-budget 0.2038 <
  hier-short 0.2260 < persistence 0.2552 — within each mechanism, hierarchy wins at matched
  budget; across mechanisms, regression wins the best-of-1 rollout; the full-resolution window
  wins outright at 2.7× tokens. This is the article's sentence: *at matched token budget, the
  advantage appears exactly where the memory horizon demands it, and nowhere else.* Gate G2c:
  **WIN (seed 0)**.
- **4.5 Per-horizon curves (T5)** + sharpness-vs-horizon; re-entry stratification; qualitative
  strips at re-entry events.
- **4.6 The mechanism controls (reg-long + reg-flat).** MEASURED — the pre-registered H2
  expectation (the regressor rolled out under the same scheduler fades toward the mean, per the
  E1 precedent) was **REFUTED**: reg-long beats hier-long-diffusion at every horizon band and
  hard bin at *equal sharpness* (0.0741 vs 0.0729; ALL Δ, direct cluster CI [+0.0110,+0.0227]).
  Reported as a reversal, not a confirmation. Protocol reconciliation (state it wherever T1 and
  the rollout result are juxtaposed): T1's diffusion win is single-step **best-of-8** (oracle
  coverage switched on); the rollout headline is **best-of-1** (coverage off) — diffusion wins
  the coverage regime, regression the deterministic rollout regime. Mechanism-agnosticism of the
  matched-budget hierarchy win (pre-registered gate, red-team P0): **CONFIRMED** — reg-long beats
  token-matched reg-flat in EVERY bin (ALL +0.0334 [+0.0248,+0.0423]; hard-bin wins at every
  horizon incl. h1-8 +0.0447; sharper 0.0741 vs 0.0512).
- **4.7 Honest limitations of the headline.** Single seed on the matrix so far (2-seed plan for
  hier-long/flat-budget); sharpness fades with horizon for *all* arms (hier-long 0.202 at h1-4 →
  0.032 at h105-152 — bounded drift, no divergence, but the G2b sharpness-retention clause
  (>0.5× single-step) is not met at deep horizons — stated plainly); best-of-1 headline with
  best-of-8 as coverage diagnostic only; blur ceiling owned by the frozen decoder (AE round-trip
  0.2412), not the dynamics.

## 5. Chapter 5 — Reversibility: the invertible stochastic recurrence (Ch. D) + three-axis taxonomy

- **5.1 Claim.** A hierarchical multi-rate recurrence whose per-level transition is RevNet additive
  coupling — extended to stochastic steps via *store-the-noise* — is exactly invertible, so a
  K-step free-running rollout backpropagates by reverse replay at recurrent-core activation memory
  **constant in K**, with gradients **provably equal to BPTT**. Formally, with coupling
  $(s_1, s_2) \mapsto (s_1 + F(s_2, \epsilon_t),\; s_2 + G(s_1 + F(s_2, \epsilon_t)))$ and stored
  noise $\epsilon_t$, the inverse is closed-form; activation memory is $O(1)$ in $K$ vs $O(K)$ for
  BPTT.
- **5.2 Verified properties (T6a).** T1 invertibility fp64 5.5e-15 (Lipschitz fp32 3.1e-5); T2
  stochastic store-the-noise exact iff ε stored (8.4e-13; wrong ε breaks, 10.9); T3 invertibility ≠
  stability (Lipschitz table — the interesting result: Lip(F)≤1 still explodes 2.7e4× because
  additive coupling is ≥1-expansive, step-Lipschitz ≈ 1+Lip(F); bounded only for Lip(F) ≲ 0.1);
  T4 reverse-replay grad-cos vs BPTT = 1.000000.
- **5.3 Measured memory frontier (T6b, Panel A).** BPTT spills at K=80 (44 s/step) and hard-OOMs
  at K=120; reversible flat ~1.1–1.2 GB through K=160; checkpointing 1.47–1.97 GB. Honest caption:
  constant-memory *capability* vs vanilla BPTT; **~Pareto with checkpointing — not a memory win**.
- **5.4 Delta over TempVerseFormer.** Deterministic → stochastic (store-the-noise ELBO step is the
  new part); invertibility ≠ stability = Behrmann-style analysis instantiated with a measured fade
  (explains the trained-RHCP fade honestly); no-forgetting framing per MacKay 2018.
- **5.5 Three-axis taxonomy (closing subsection).** Reversibility across depth (RevNet), time
  (TempVerseFormer/Ch. D), computation (inner optimization loop) — and why the computation axis
  cannot pay *under EBT's second-order truncated recipe*: the reconstructable descent state is
  (y,v) ≈ 10 KB/sample while the final-step `create_graph` second-order graph dominates the peak
  and must be rebuilt from a fresh forward regardless. **Measured (T6c, `logs_rhcp/mem_panel.md`):
  naive full-graph 18.1 GB (spilling, 17 s/step) at K=4 and OOM at K=8; truncate / checkpoint /
  reversible all flat at 5.282–5.296 GB through K=128; reversible saves 14 MB = 0.26% of peak vs
  checkpointing, at ~2× wall-clock (recompute).** The analytic states-vs-graph band is ~0.6–1.0%
  (bits-per-step volume-contraction accounting **attributed to Maclaurin 2015, applied**); the
  measured panel closes the earlier unmeasured-graph caveat. Qualification (red-team): the no-go
  is recipe-scoped — a first-order many-step store-the-noise sampler has a genuinely O(K)
  activation peak where reversibility saves real memory, but it ties checkpointing there too (the
  temporal Pareto tie). DOODL/MALI/EDICT preempt the diffusion-sampler variant.
  Applicability boundary sentence: *reversibility pays on recurrences whose transition is the
  coupling itself; diffusion rollouts are the checkpointing regime.*

## 6. Chapter 6 — Related work and honest scoping

- **6.1 Hierarchical latent video prediction:** CW-VAE, DLH (already = learned-clock CW-VAE — the
  pre-emption that killed our adaptive-clock direction), TECO, MSPred, NUWA-XL, MilliVid
  (architecture family this system miniaturizes); FDM (Harvey '22).
- **6.2 Memory in world models:** Memory Maze, TECO-Minecraft, LSSVWM (2505.20171), WorldPack
  (2512.02473), VidMem (2506.05284) — our testbed as the minimal fully-analytic member; why not
  those benchmarks (exact hardness labels, 16 GB feasibility).
- **6.3 Prediction-mechanism comparisons:** SVG/SAVP (regression blurs), DIAMOND (diffusion
  commits), EBT (2507.02092), EBT-Policy (2510.27545 — published small-EBT positive we do NOT
  contradict), mean-field GD entropy collapse (2403.08362) as the mechanism instantiated (not
  discovered) by our collapse study.
- **6.4 Rollout training + reversibility:** Self-Forcing (the stabilizer reversibility merely
  enables), MacKay 2018, CrevNet, i-ResNet/Behrmann, AlignProp/DRaFT (checkpointing chosen because
  state ≪ graph), DOODL, MALI, EDICT, Maclaurin 2015.
- **6.5 Explicit non-claims box:** no beats-checkpointing claim; no universal EBT refutation (scale
  scoped, §3.5.5); no FVD headline (I3D/Kinetics mismatch at 64×64 grayscale); no claim that
  reversibility stabilizes rollout; MMNIST aggregate metrics excluded by design (background
  dominance — Appendix A carries the bridge).
- Citation-verification debt (from `docs/FINAL_ARTICLE_REVIEW_2026-07-02.md` §5): all
  citation-mechanism claims to be verified against PDFs before submission; re-sweep prior art at
  submission (DSA/DySink collisions <60 days old).

## 7. Chapter 7 — Conclusion

- **One claim:** the coarse level of a hierarchical latent video model earns its token budget
  precisely when the prediction horizon exceeds what a flat context can carry — demonstrated with
  exact hardness labels, matched token budgets, and CIs.
- **Its boundary conditions:** emerges only under the larger optimization budget (null at 6k, win
  at 20k); holds for both mechanisms at matched budget (mechanism-agnostic — reg-flat control),
  while the best mechanism is regime-dependent (diffusion for single-step best-of-N coverage,
  regression for the best-of-1 rollout); requires the horizon regime (null below h25); an
  efficiency point, not a capability necessity (full-res flat context wins outright at 2.7×
  tokens); measured on a single synthetic world at 16 GB scale.
- **The two honest negatives that scope it:** (i) energy-based descent — even a correctly ordered
  landscape does not yield reachable generation at our scale (two-sided EBT study); (ii)
  reversibility saves memory vs BPTT but is Pareto-neutral vs checkpointing, and cannot pay on the
  inner optimization loop (0.26% measured, ~0.6–1.0% analytic band).
- **Future work:** paper-scale EBT (the 35% branch), Self-Forcing-style distribution-matching
  rollout objectives enabled by the Ch. D recurrence, natural-video instantiation of the testbed
  protocol, sharp renderer over the frozen latents (Step-1 adversarial renderer precedent).

## Appendices

- **A. OpenSTL MMNIST bridge (T7, QUEUED).** Our stack evaluated on the standard benchmark for
  connectivity: expected mid-pack aggregate MSE/SSIM (those metrics reward mean prediction —
  the 93%-black-background argument), competitive LPIPS; AE round-trip ceiling row decomposing
  decoder vs dynamics error.
- **B. Config-gap and arm-config tables.** T2b (ours-vs-paper EBT gap, = CH3 Table 3.3, DONE);
  full arm/hyperparameter table for Ch. 4 (from `hvp_rev/rhvd.py` + `docs/STAGE2_DESIGN.md`).
- **C. Extra probes.** EBT positive controls (copy/shift: correct landscape learned, 8/8, gap +26,
  yet cold descent still blank — the reachability dissociation in miniature); think-longer
  saturation (S2 recipe: 0.038→0.010 by 4 descent steps, flat to 32); optional energy-Spearman
  ranking probe (does E rank sample quality? pre-registered risk: E(mean)<E(tgt) ⇒ anti-ranking);
  CW-VAE-entropy ρ≈0 precedent.

---

# THE TABLE SET

Conventions: fg-MSE on decoded frames under the GT foreground mask (lower better); "sharp" =
foreground Laplacian sharpness (higher better; raw-GT 0.4785, AE-roundtrip ceiling 0.2412,
structure threshold ≈0.08); best-of-N over N samples; hardness = analytic novel-content labels
(Ch. 2). Δ = base − arm (positive = arm better); 95% CI = paired clip-level cluster bootstrap
(rollout bins exclude empty-GT-foreground records, 21.5%, reported as a bg-only row).

## T1 — Mechanism comparison at matched capacity (Ch. 3, Part A) — **DONE**

Source: `logs_rhcp/step1_mech_clean2.log` (corrected re-run: fixed hardness anchor + single
raw-GT reference; supersedes `step1_mech_clean.log`, whose 0.0489/0.0363 values must NOT be
cited; reg 6k steps, diff 8k, best-of-8, NFE 16, held-out test; params reg 7.27M / diff 7.53M /
ebt 7.28M).

| mech | fg-MSE all | easy | HARD | mean-of-N | sharp |
|---|---:|---:|---:|---:|---:|
| regression | 0.0664 | 0.0232 | 0.1808 | 0.0714 | 0.1591 |
| diffusion | **0.0512** | **0.0120** | **0.1549** | **0.0640** | **0.1886** |
| EBT (Part B) | collapsed — see T2 | | | | ≈0.009–0.035 |

References: raw-GT sharp 0.4785 | AE-roundtrip sharp 0.2412 (decoder ceiling).
Persistence baseline: fg-MSE 0.2367 | sharp 0.0515. Easy/hard split = novel_strip>0; frac_hard 0.27.

**T1b — matched-NFE (best-of-N fg-MSE):**

| NFE | reg | diff |
|---:|---:|---:|
| 4 | 0.0680 | 0.0546 |
| 8 | 0.0680 | 0.0514 |
| 16 | 0.0680 | **0.0503** |

Reading: diffusion beats regression on every column at matched capacity and NFE; regression flat
in N (deterministic), diffusion improves with samples — the "averages vs commits" signature.
Protocol scope: single-step, best-of-8 (oracle coverage regime); under the best-of-1 rollout
protocol the mechanism ranking reverses (§4.6, T4) — the two results are reconciled there.

## T2 — Two-sided EBT study, fair-scale runs (Ch. 3, Part B) — **DONE**

Source: `logs_rhcp/ebt_scaled_s2.log`, `logs_rhcp/ebt_scaled_s1.log` (xxs shape d384/6L/6H/FFN4×,
ctx 16, 14.66M, eff-batch 128, 20k steps / 5k warmup; probes on the final landscape; mean→target
distance reference 0.964 — d_tgt above it means descent output is farther from target than the
mean is; AE-roundtrip sharp ceiling ≈0.24).

**S2 recipe (α fixed 15.97, S2-calibrated; replay + Langevin 1):**

| step | E(target) | E(mean) | lowest | cold d_tgt | fg-sharp |
|---:|---:|---:|---|---:|---:|
| 0 | −1.004 | −1.414 | mean [BAD] | 1.570 | 0.0122 |
| 5,000 | −35.579 | −36.605 | mean [BAD] | 1.282 | 0.0108 |
| 10,000 | −42.136 | −43.232 | mean [BAD] | 1.282 | 0.0101 |
| 15,000 | −41.891 | −42.982 | mean [BAD] | 1.286 | 0.0097 |
| 20,000 | −41.761 | −42.951 | mean [BAD] | 1.292 | 0.0086 |

E(mean)<E(target) at **all 21 probes**; training SmoothL1 converged (≈0.597 from ~7k). Verdict:
stably mis-ordered landscape; the under-training objection is closed for this recipe.
Disclosure (norm confound): the batch-mean probe token has ~8× lower norm than an individual
target, so a smoothness/norm preference could carry part of the ordering; rebuttal in-probe —
E(randn) stays far above both throughout, and S1 flips the *same* probe to the correct ordering
(a norm effect alone cannot produce opposite orderings on identical probe inputs).

**S1 recipe (α learnable, init 23.95 → 33.27; mcmc 2, per-step losses, no replay):**

| step | E(target) | E(mean) | lowest | cold d_tgt | fg-sharp |
|---:|---:|---:|---|---:|---:|
| 0 | −0.337 | −0.601 | mean [BAD] | 1.642 | 0.0139 |
| 5,000 | −25.654 | −25.765 | mean [BAD] | 1.131 | 0.0165 |
| 10,000 | −35.557 | −32.595 | **target [GOOD]** | 1.139 | 0.0221 |
| 15,000 | −40.858 | −38.421 | **target [GOOD]** | 1.158 | 0.0259 |
| 20,000 | −44.097 | −41.961 | **target [GOOD]** | 1.148 | 0.0259 |

Target strictly lowest at all 15 probes from 6k→20k (ordering **repaired**; target-below-mean gap
+0.22…+3.09 across the 15 probes); cold-descent sharpness
peaks at 0.0345 (step 19k) — a real trend but ≈2.3× below the 0.08 structure threshold. Final
paper-inference probe (step 20k, best-of-4 min-energy, 4 steps, Langevin 1, both α readings):
α[cur/4]=8.316 → d_tgt 1.207, fg-sharp 0.0127; α[init/4]=5.988 → d_tgt 1.233, fg-sharp 0.0120.
**Decomposition: ordering (repaired under S1) × reachability (fails under both) — they dissociate.**

**T2b — config-gap block (ours vs paper)** — DONE; lives as CH3 Table 3.3. Sources:
`hvp_rev/ebt_ref_scaled.py` docstring + `C:/Users/leoni/Documents/projects/EBT/job_scripts/vid/pretrain/ebt_s{1,2}_vae.sh`.

| | This study (S2 / S1) | Paper S2 | Paper S1 |
|---|---|---|---|
| Model | xxs d384/6L/6H FFN4× (14.66M) | xxs | xxs |
| Context | 16 | 16 | 16 |
| Steps (warmup) | 20k (5k) | 60k (10k) | 100k (10k) |
| Eff. batch | 128 (+64 replay / plain) | 256 (32×4×2) + replay | 256 (16×4×4) |
| α | 15.97 fixed / 23.95 learnable (recalibrated) | 3,000 fixed | 30,000 learnable, 90,000× lr-mult |
| Data | 64×64 gray synthetic, 2,700×100 | SSv2 224×224 SD-VAE | SSv2 224×224 SD-VAE |
| Hardware | 1× RTX 4060 Ti 16 GB | 4 GPUs/node | 4 GPUs/node |

## T3 — Single-step memory gate: emergence with training (Ch. 4.2) — **DONE**

Sources: `logs_rhcp/mem_gate_full_s{0,1}.log` (6k steps), `logs_rhcp/mem_gate_full_s{0,1}_t20k.log`
(20k steps). Arms: coarse48 (48c+6f, 1232 tokens, 7.54M) vs nctx6 (6c+6f, 560 tokens, 7.53M),
B=16, best-of-8 fg-MSE, stratified by novel-fg-at-6; Δ = nctx6 − coarse48.

| training | seed | HARD-bin Δ | 95% CI | verdict | ALL Δ | 95% CI | verdict |
|---:|---:|---:|---|---|---:|---|---|
| 6k | 0 | +0.0014 | [−0.0004, +0.0033] | null | −0.0003 | [−0.0013, +0.0006] | null |
| 6k | 1 | +0.0008 | [−0.0009, +0.0025] | null | −0.0021 | [−0.0032, −0.0012] | LOSS |
| 20k | 0 | **+0.0135** | [+0.0080, +0.0196] | **WIN** | +0.0022 | [+0.0006, +0.0038] | WIN |
| 20k | 1 | **+0.0143** | [+0.0083, +0.0208] | **WIN** | +0.0024 | [+0.0007, +0.0041] | WIN |

Hard-bin absolute at 20k, seed 0: 0.1053 → 0.0918 (n=184). Sharpness tied (≈0.188 both arms) —
the win is retrieval, not sharpness. Reading: **retrieval-from-memory emerges late in training**;
6k-step ablations would have (wrongly) killed the memory headline.

## T4 — Rollout hierarchy matrix (Ch. 4.4) — **DONE (all arms)**

Source: `logs_rhcp/compare_clustered.log` (`venv/Scripts/python.exe -u -m hvp_rev.rhvd
--compare ...` over `logs_rhcp/rhvd_eval_{arm}_s0.npz`; 30k-step arms — flat-long 60k @ B=8,
sample-matched; 48-frame GT warmup + 152 AR frames, 100 test clips, best-of-1, paired per-frame
fg-MSE, clip-level CLUSTER bootstrap; empty-GT-fg records 3263/15200 (21.5%) excluded from bins
and reported as the bg-only row).

**Headline pairing — hier-long vs flat-budget (equal 1152 context tokens), Δ = flat − hier:**

| horizon | hardness | n | flat-budget | hier-long | Δ | 95% CI | verdict |
|---|---|---:|---:|---:|---:|---|---|
| h 1-8 | easy | 434 | 0.0440 | 0.0419 | +0.0022 | [−0.0040, +0.0089] | null |
| h 1-8 | HARD | 150 | 0.1435 | 0.1359 | +0.0075 | [−0.0164, +0.0327] | null |
| h 9-24 | easy | 919 | 0.1216 | 0.1189 | +0.0028 | [−0.0075, +0.0133] | null |
| h 9-24 | mid | 45 | 0.1661 | 0.1624 | +0.0038 | [−0.0121, +0.0182] | null |
| h 9-24 | HARD | 282 | 0.1800 | 0.1728 | +0.0071 | [−0.0110, +0.0260] | null |
| h 25-60 | easy | 2286 | 0.2210 | 0.1989 | +0.0221 | [+0.0080, +0.0368] | **WIN** |
| h 25-60 | mid | 79 | 0.2218 | 0.2036 | +0.0182 | [−0.0049, +0.0437] | null |
| h 25-60 | HARD | 550 | 0.2554 | 0.2209 | **+0.0344** | [+0.0162, +0.0531] | **WIN** |
| h 61+ | easy | 5610 | 0.3060 | 0.2788 | +0.0272 | [+0.0166, +0.0382] | **WIN** |
| h 61+ | mid | 209 | 0.3224 | 0.2842 | +0.0382 | [+0.0178, +0.0616] | **WIN** |
| h 61+ | HARD | 1353 | 0.3073 | 0.2759 | **+0.0314** | [+0.0202, +0.0431] | **WIN** |
| bg-only | | 3263 | 0.0067 | 0.0047 | +0.0020 | [+0.0013, +0.0026] | **WIN** |
| ALL | | 15200 | 0.2038 | 0.1848 | +0.0189 | [+0.0121, +0.0264] | **WIN** |

**The spine in one table: null in every bin below horizon 25; CI-excluding-zero win in every
easy/HARD bin from h25-60 onward** (the sparse mid cell, n=79, is null at h25-60 and WIN at
h61+ under the cluster bootstrap). (Both arms beat hier-short in every bin — long context helps;
the *hierarchical packaging* of it wins once the horizon exceeds what 18 flat frames carry.)

**ALL-row window ordering (fg-MSE over 48+152-frame rollouts, seed 0, final matrix):**

| arm | ctx tokens | ALL fg-MSE | sharp | status |
|---|---:|---:|---:|---|
| flat-long (48f) | 3072 (2.7×) | **0.1564** | **0.0996** | DONE (caveat: B=8×60k sample-matched — EMA horizon/grad-noise differ) |
| reg-long (48c+6f, regression) | 1152 | 0.1682 | 0.0741 | DONE |
| hier-long (48c+6f) | 1152 | 0.1848 | 0.0729 | DONE |
| reg-flat (18f, regression) | 1152 | 0.2015 | 0.0512 | DONE |
| flat-budget (18f) | 1152 | 0.2038 | 0.0582 | DONE |
| hier-short (6c+6f) | 480 | 0.2260 | 0.0392 | DONE |
| persistence | — | 0.2552 | — | DONE (floor) |

Measured outcomes (supersede the pre-registered expectations): **reg-long** — the fade
expectation (E1 precedent) was **REFUTED (reversal)**: reg-long beats hier-long at every horizon
band and hard bin at equal sharpness (ALL Δ, direct cluster CI [+0.0110,+0.0227]); protocol note —
T1 is best-of-8 coverage, the rollout is best-of-1 (§4.6 reconciles). **flat-long** — beats every
arm on both error and sharpness: the Pareto framing holds (hierarchy recovers most of the
long-memory benefit at ~37% of the context tokens; the full-res window is best outright at 2.7×).
**reg-flat** — mechanism-agnosticism gate PASSED: reg-long beats token-matched reg-flat in EVERY
bin (ALL +0.0334 [+0.0248,+0.0423]; h1-8 HARD +0.0447 — for regression the win starts at h1-8,
no null prelude, unlike diffusion). `logs_rhcp/rhvd_eval_reg-flat_s0.npz`.

## T5 — Per-horizon rollout curve, hier-long (Ch. 4.5) — **DONE**

Source: `logs_rhcp/rhvd_eval_hier-long_s0.npz` (numpy re-computation, 2026-07-03; 15,200
frame-records, horizons 1–152 past the 48-frame warmup).

| horizon | n | fg-MSE | sharp | persistence fg-MSE |
|---|---:|---:|---:|---:|
| h 1-4 | 400 | 0.0376 | 0.2023 | 0.2133 |
| h 5-8 | 400 | 0.0625 | 0.1825 | 0.2376 |
| h 9-16 | 800 | 0.0861 | 0.1591 | 0.2393 |
| h 17-24 | 800 | 0.1209 | 0.1304 | 0.2462 |
| h 25-40 | 1600 | 0.1481 | 0.1072 | 0.2656 |
| h 41-60 | 2000 | 0.1787 | 0.0851 | 0.2660 |
| h 61-80 | 2000 | 0.2066 | 0.0606 | 0.2598 |
| h 81-104 | 2400 | 0.2213 | 0.0436 | 0.2586 |
| h 105-152 | 4800 | 0.2219 | 0.0323 | 0.2528 |
| ALL | 15200 | 0.1848 | 0.0729 | 0.2552 |

Reading: below persistence at *every* horizon out to 152 frames (>3 loop periods); error saturates
(0.221–0.222 over the last 70 frames) rather than diverging; sharpness fades monotonically —
reported honestly as the deep-horizon limitation (§4.7).

## T6 — Reversibility panels (Ch. 5)

**T6a — Ch. D verified properties — DONE** (source: `docs/THEORY_CHAPTER_D.md`, measured by
`hvp_rev/theory.py`):

| # | property | result |
|---|---|---|
| T1 | block invertibility (unconstrained / Lipschitz) | fp64 **5.5e-15** / fp32 3.1e-5 |
| T2 | stochastic store-the-noise invertibility | exact iff ε stored (**8.4e-13**); wrong ε breaks (10.9) |
| T3 | invertibility ≠ stability, ‖δ₁₀₀‖/‖δ₀‖ | unconstrained 3.3e2; Lip(F)≤1 **2.7e4 (still explodes)**; Lip≤0.1 → 1.5; Lip≤0.02 → 1.0 |
| T4 | reverse-replay gradients vs BPTT | **grad-cos 1.000000** |

**T6b — Panel A memory frontier — DONE** (source: `docs/SESSION_JOURNEY_AND_EBT_PLAN.md` §2.2,
`hvp_rev/mem_frontier.py`, real train loop + Adam, 3 levels, batch 32):

| K | BPTT (MB) | checkpoint | reversible | BPTT ms/step |
|---:|---:|---:|---:|---:|
| 40 | 12,162 | 1,471 | 1,091 | 740 |
| 80 | 23,305* | 1,637 | 1,125 | 43,797* |
| 120 | **OOM** | 1,801 | 1,152 | — |
| 160 | **OOM** | 1,967 | 1,176 | — |

*K=80 BPTT spills to shared RAM (~60× slower); hard OOM at K=120. Caption (pre-committed):
constant-memory capability vs vanilla BPTT; **~Pareto with checkpointing — not claimed as a win**.

**T6c — inner-loop 4-way memory panel — MEASURED** (source: `logs_rhcp/mem_panel.md`; naive /
truncate_mcmc / checkpoint / reversible, K ∈ {4, 8, 32, 128}, B=12, fp32, spill-guarded; CPU
`--verify` grad-cos(reversible, checkpoint) > 0.999999):

| arm | K=4 | K=8 | K=32 | K=128 |
|---|---|---|---|---|
| naive (full graph) | 18.141 GB, 17.1 s/step (spill) | **OOM** (29.76 GB attempted) | OOM (implied) | OOM (implied) |
| truncate | 5.282 GB | 5.282 GB | 5.282 GB | 5.282 GB, 7.85 s/step |
| checkpoint | 5.282 GB | 5.283 GB | 5.286 GB | 5.296 GB, 7.85 s/step |
| reversible | 5.282 GB | 5.282 GB | 5.282 GB | 5.282 GB, 15.37 s/step |

Outcome matches the pre-registration: naive OOMs by K=8; truncate/checkpoint/reversible flat
through K=128; **reversible saves 14 MB = 0.26% of peak vs checkpointing, at ~2× wall-clock
(recompute)**. States-vs-graph accounting: per-step (y,v) state 0.117 MB; checkpoint's K-state
total at K=128 is 15.0 MB = 0.28% of its measured peak — the most reversibility could save on
this axis. Quote **0.26% (measured)** alongside the ~0.6–1.0% analytic band.

## T7 — OpenSTL MMNIST bridge (Appendix A) — **QUEUED**

Pre-registered expectation: mid-pack aggregate MSE/SSIM vs the OpenSTL leaderboard — those
aggregate metrics reward mean prediction on a 93%-black benchmark (our own measurement: all-black
beats CW-VAE on aggregate SSIM/PSNR); competitive LPIPS; an AE round-trip ceiling row that
decomposes decoder error vs dynamics error. Purpose: connectivity to the standard suite, not a
leaderboard claim.

---

# FIGURE LIST (one-line captions)

| # | figure | caption | status |
|---|---|---|---|
| F1 | testbed strip (ch2-1) | 64 px viewport pans a 192 px toroidal world; 67% off-screen, ~48-frame loop; a digit exits and re-enters. | QUEUED (placeholder in CH2 draft) |
| F2 | mechanism grids (ch3-1) | last-context / prediction / GT for regression vs diffusion: averaging vs commitment, same harness. | DONE (assets: `logs_rhcp/step1_mech_clean_{reg,diff}.png`; compose) |
| F3 | energy ordering vs step (ch3-2) | S2 stays mis-ordered (mean below target) through 20k; S1 flips at ~6k and the gap grows — yet neither generates. | data DONE (T2); plot QUEUED |
| F4 | cold-descent sharpness vs step (ch3-3) | descent output stays in the collapse regime under both recipes; S1's small trend stays ~2.3× below the structure threshold. | data DONE (T2); plot QUEUED |
| F5 | memory-gate emergence | hard-bin Δ with CIs at 6k vs 20k × 2 seeds: the retrieval capability emerges late. | data DONE (T3); plot QUEUED |
| F6 | per-horizon rollout curves | fg-MSE vs horizon for hier-long / flat-budget / hier-short / persistence; the curves separate exactly past h≈24. | data DONE (T4/T5); plot QUEUED |
| F7 | horizon × hardness win map | Δ(flat − hier) heat/CI map: null below h25, wins beyond — the claim in one picture. | data DONE (T4); plot QUEUED |
| F8 | rollout strips at re-entry | qualitative 48+152-frame rollout: hier-long re-places the returning digit, flat-budget hallucinates. | ckpts DONE; grid QUEUED (interim: `logs_rhcp/rhvd_*_train30000.png`) |
| F9 | Panel A frontier | activation memory vs K: BPTT OOMs at 120; reversible flat ~1.2 GB; checkpointing ~parity. | data DONE (T6b); plot QUEUED |
| F10 | Lipschitz stability panel | four exactly-invertible recurrences, perturbation growth 3.3e2× → 1.0×: Lip(F), not invertibility, is the lever. | data DONE (T6a); plot QUEUED |
| F11 | three-axis taxonomy diagram | reversibility across depth / time / computation, with the state-vs-graph accounting on the third axis. | QUEUED (drawing) |
| F12 | 4-way inner-loop memory panel | naive / truncate / checkpoint / reversible vs K: the 0.26% regime, measured. | data DONE (`logs_rhcp/mem_panel.md`); plot QUEUED |

---

# §S. STATUS LEDGER

## Tables and figures

| item | status | eta / next action |
|---|---|---|
| T1 mechanism comparison (+NFE, refs) | **DONE** (verified from `step1_mech_clean2.log`; corrected hardness anchor — `clean.log` superseded) | — |
| T2 EBT two-sided (S1+S2 probes, paper-inference) | **DONE** (verified from `ebt_scaled_s{1,2}.log`) | — |
| T2b config-gap block | **DONE** (CH3 Table 3.3; scripts cross-checked) | — |
| T3 memory gate emergence (2 seeds × 2 budgets) | **DONE** (verified from 4 `mem_gate_full_*` logs) | — |
| T4 rollout matrix — hier-long/flat-budget/hier-short/persistence | **DONE** (cluster CIs, `compare_clustered.log`, seed 0) | 2-seed replication of headline pairing QUEUED |
| T4 reg-long row (H2) | **DONE** — REVERSAL: reg-long beats hier-long everywhere at equal sharpness (ALL cluster CI [+0.0110,+0.0227]) | fade expectation REFUTED; best-of-8 vs best-of-1 protocol note propagated |
| T4 flat-long row | **DONE** — ALL 0.1564, sharp 0.0996; best arm outright | B=8×60k sample-matched caveat disclosed wherever quoted |
| T4 reg-flat row (mechanism-agnosticism control) | **DONE** — reg-long beats reg-flat in EVERY bin (ALL +0.0334 [+0.0248,+0.0423]); pre-registered gate PASSED | mechanism-agnosticism CONFIRMED |
| T5 per-horizon curve (hier-long) | **DONE** (recomputed from npz) | flat-budget/hier-short columns: same npz, trivial |
| T6a Ch. D properties | **DONE** (`THEORY_CHAPTER_D.md`) | — |
| T6b Panel A frontier | **DONE** (journey doc §2.2) | — |
| T6c inner-loop 4-way panel | **MEASURED** (`logs_rhcp/mem_panel.md`) | reversible saves 14 MB = 0.26% of peak vs checkpointing at ~2× wall-clock; naive 18.1 GB spill @K4, OOM @K8 |
| T7 OpenSTL bridge | **QUEUED** | expectation pre-registered above |
| F2 | DONE (compose) | F3–F10: plots from existing data (CPU, ~1 day) |
| F1, F11 | QUEUED (drawing) | F8: after grid render; F12: plot from `mem_panel.md` |

## Pre-registered gates — outcomes on the record

- **G0a (memory gate)** — text: *"the coarseL arm must WIN the HARD re-entry bin (CI excluding 0).
  If null/LOSS on seed 0, run --seed 1 before declaring the memory-horizon headline dead (2-seed
  rule)."* Outcome: null at 6k (both seeds) → **WIN at 20k on BOTH seeds** (+0.0135
  [+0.0080,+0.0196]; +0.0143 [+0.0083,+0.0208]). **PASSED, 2-seed.**
- **G0b (diffusion commits, standardized latents)** — outcome per `docs/STAGE2_DESIGN.md`: *"G0b
  PASSED (diffusion commits with standardized latents and beats regression on every column)."*
  **PASSED** (T1).
- **G2a (smoke)** — passed implicitly: all four 30k arms trained without NaN, digit structure by
  2k (train grids on disk).
- **G2b (rollout stability)** — bounded-drift clause holds (error saturates ≈0.222, never crosses
  persistence); sharpness-retention clause (>0.5× single-step at depth) **not met** at h≥81 —
  reported as the honest deep-horizon limitation, no context-noise iteration spent.
- **G2c (headline)** — text: *"hier-long beats flat-budget on the hard re-entry bin at horizon
  >= 24 with CI excluding 0 (paired, CRN). If flat-budget matches hier-long, the honest conclusion
  is 'context length, not hierarchy' — write it."* Outcome: **WIN, seed 0** (cluster CIs: h25-60
  HARD +0.0344 [+0.0162,+0.0531]; h61+ HARD +0.0314 [+0.0202,+0.0431]). Seed-1 replication queued
  before the word "headline" appears in print.
- **H2 (mechanism-at-rollout expectation)** — pre-registered: the regressor rolled out fades
  toward the mean (E1 precedent) while diffusion holds sharpness. Outcome: **REFUTED — reversal
  on the record**: reg-long beats hier-long-diffusion everywhere at equal sharpness (ALL cluster
  CI [+0.0110,+0.0227]); reconciled with T1 via the best-of-8 vs best-of-1 protocol difference.
- **Mechanism-agnosticism gate (red-team P0, pre-registered before reg-flat completed)** — text:
  *"hierarchy is mechanism-agnostic iff reg-long beats reg-flat with CI excluding 0 in the
  hard/high-horizon bins."* Outcome: **PASSED** — reg-long beats reg-flat in every bin (ALL
  +0.0334 [+0.0248,+0.0423]; hard-bin wins at every horizon incl. h1-8 +0.0447).
- **EBT scoping gate** — text (from `hvp_rev/ebt_ref_scaled.py`): *"if by 10k post-warmup steps
  there is NO E(target)<E(mean) flip AND no sharpness trend, write the hardened negative; a
  flip/trend => scale-sensitivity finding."* Outcome: **S2 = hardened negative** (no flip in 21/21
  probes, converged loss); **S1 = the two-sided finding** (ordering flips at ~6k and holds 15/15;
  sharpness trend real but 2.3× below structure; paper-inference recipe also collapsed at both α
  readings). Both branches written into CH3 §3.5.2–3.5.4.

## Verification notes (what this skeleton could not verify)

- reg-long/flat-long rows verified from `compare_clustered.log`; reg-flat row from
  `logs_rhcp/rhvd_eval_reg-flat_s0.npz` (final 2026-07-03 updates, handoff §12).
- T7 has no data — expectation pre-registered only; T6c is measured (`logs_rhcp/mem_panel.md`).
- Citation-mechanism claims in Ch. 6 remain a verification debt (review doc §5) — to be checked
  against PDFs at write-up.
- All other numbers in this file were re-read from the named logs/npz/scripts on 2026-07-03/04.

---

## CORRECTIONS LEDGER (2026-07-03, validity audit) — supersedes numbers above where they conflict

- **T1 REPLACED** by the clean2 recompute (corrected hardness anchor + single raw-GT reference,
  logs_rhcp/step1_mech_clean2.log): reg 0.0664/0.0232/0.1808/sharp 0.1591; diff
  0.0512/0.0120/0.1549/sharp 0.1886; frac_hard 0.27; NFE diff 0.0546->0.0503, reg 0.0680 flat.
  Diffusion > regression on every column (conclusion unchanged; absolute values now include the
  decoder floor).
- **T4/T5 CIs REPLACED** by cluster-bootstrap versions (logs_rhcp/compare_clustered.log):
  headline verdicts survive; early-horizon flat-budget cells now null (cleaner horizon-gated
  story); empty-fg records (21.5%) excluded from bins, reported as bg-only row.
- **flat-long (DONE)**: rollout ALL 0.1564, sharp 0.0996 — beats every arm; fixed-window corner
  confirms the Pareto framing (full-res memory best at 2.7x tokens; hierarchy = efficiency point).
  Caveat: B=8 x 60k sample-matched protocol.
- **reg-flat (DONE, 2026-07-03 final update)**: reg-long beats token-matched reg-flat in EVERY
  bin (ALL +0.0334 [+0.0248,+0.0423]; hard-bin wins at every horizon incl. h1-8 +0.0447; sharper
  0.0741 vs 0.0512) — mechanism-agnosticism of the matched-budget hierarchy win CONFIRMED
  (pre-registered gate PASSED). `logs_rhcp/rhvd_eval_reg-flat_s0.npz`.
- **T6c (MEASURED, 2026-07-03, `logs_rhcp/mem_panel.md`)**: naive full-graph 18.1 GB/17 s-step
  (spill) at K=4, OOM at K=8; truncate/checkpoint/reversible all 5.282–5.296 GB flat through
  K=128; reversible saves 14 MB = 0.26% of peak vs checkpointing at ~2× wall-clock — quote
  0.26% (measured) alongside the ~0.6–1.0% analytic band (replaces all "~0.2%" and the
  "1.5–2.3 GB unmeasured estimate" phrasing).
- Wording: "152 AR frames after 48-frame GT warmup" everywhere (not "200-frame rollout").
- Full audit + disclosures: docs/draft/VALIDITY_AUDIT.md.
