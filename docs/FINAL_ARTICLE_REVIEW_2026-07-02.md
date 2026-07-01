# Final-article direction review (2026-07-02)

**How this was produced:** 14-agent structured review — 5 evidence/forensics readers (master doc, EBT-negative
forensics, minimal-MilliVid gap analysis, reversibility math/memory accounting, web prior-art sweep), a 4-direction
assessment panel, a synthesis judge, and 3 adversarial attackers (feasibility, novelty, dissertation-coherence).
This document is the corrected synthesis after the attacks.

---

## 1. Direct answers to the standing questions

### Q1. "Should I drop EBT and switch to Ch. D?"
**Neither.** The panel converged on a third option: **keep EBT as a scoped, two-sided mechanism chapter, hardened by
exactly one overnight run** — not as the article's spine, and not dropped.

The forensic audit found a real hole in the "EBT is settled dead" record:

- Every landscape probe (`E(mean) < E(target)`) ran on models trained **≤ 2,000 steps = 3.3% of the paper's 60k steps,
  entirely inside the paper's own 10k-step warmup window**, at 0.4% of its sample budget, FFN 1× vs the paper's 4×,
  batch 32 vs 256, **with the loss still falling at stop** (~1.8× above even the predict-the-mean SmoothL1 floor).
  The mechanistic claim "the trained energy's minimum is the mean" is **not scale-defensible as recorded** — a reviewer
  lands the undertraining punch immediately.
- What DOES survive scale-independently: the **descent-reachability failure** (copy/shift positive controls learned a
  CORRECT landscape — target = global min, 8/8, gap +26 — yet cold descent still landed blank; inference sweeps
  steps 2→128 flat, α 0.3→3000 flees the target), and the **~90%-safe comparative claim** (diffusion commits sharply
  where EBT collapses, at matched small compute, official-code control).
- Forensics odds: **P(collapse artifact disappears at paper-faithful scale) ≈ 35%; P(EBT matches diffusion sharpness
  at any 16GB-reachable scale) ≈ 10%.**

**The fix costs one overnight** (~5–16 h, after a 10-min 100-step timing probe): reference control at
d384 / FFN 4× / ctx 16, eff-batch 128 via 4× accum, 20k steps, 5k warmup, cosine, with `E(target)`-vs-`E(mean)` +
fg-sharpness probes every 1k — **plus the never-run S1-recipe control** (learnable α=30000 with 90000× lr-mult, mcmc 2,
best-of-N min-energy — the config behind the paper's *actual* video generations; closes the most likely reviewer
objection). Both branches are writable: no trend by 10k post-warmup → hardened mechanistic negative; trend → honest
scale-sensitivity finding ("EBT needs ≥N× diffusion's compute to escape mean-collapse; diffusion commits at 2k steps").

Framing rules (novelty attack): never "reproduction failure" — EBT §5.2 never claimed sharp video generation; scope
against **EBT-Policy (2510.27545)**, a published small-EBT positive in robotics; cite mean-field microcanonical GD
entropy collapse (2403.08362) as the mechanism being instantiated, not discovered.

### Q2. "Stable minimal MilliVid, then swap in EBT?"
**The base system: yes — buildable and worth it (~4–6 person-days + 2–4 GPU-days).** But "swap the denoiser" is a
comparison study, not the novelty; the **memory-horizon result is the headline**, the mechanism comparison is a chapter.

What "stable training" concretely requires (gap analysis, in priority order):
1. **Latent standardization — the single biggest silent quality killer.** `step1_mech.py`'s Diffusion noises raw tanh
   latents (std coarse 0.139 / fine 0.242) at noise std 1.0. `ebt_reference_control.py` already standardizes; the
   diffusion path never did. Half-day fix; **must precede every other run** (it changes all diffusion numbers).
2. **Checkpoint save/resume** — no current script ever `torch.save`s the denoiser; overnight runs on Windows without
   resume are a coin-flip. Plus **EMA 0.999**, constant LR (restartable), per-level loss logging.
3. **Universal multi-layout trainer** (MilliVid-style: per-token roles, per-token t, one target level per step) and the
   **coarse-to-fine AR rollout scheduler** ([4,16]-style) + long-horizon eval glue — none exist; all evals so far are
   single-step. Budget 2–3 days for the layout/rollout code (the fiddliest part), plus async batch prefetch (current
   Python-loop batch assembly can silently halve throughput at multi-frame windows).
4. Model: DiT d=384 / depth 8 (~20–25M) via the verified `faithful_nets.py` path, bf16, seq ≤ ~1150 tokens, B=32–48.
   Realistic wall-clock (feasibility attack): **0.6–1.2 s/step → 30k steps = 5–10 h** (tight overnight). Run a 100-step
   timing+VRAM probe first; on this machine a 10× slowdown means WDDM spill, not load — kill and shrink.

### Q3. "Reversibility on the optimization steps — worth investigating?"
**Investigated in depth: NO as a method (predetermined null), YES as a theory subsection.** Three independent kills:

1. **The memory accounting is fatal.** Backprop-through-descent needs, per reverse step, an HVP + mixed VJP that
   requires **rebuilding the `create_graph` graph of ∇E from a fresh forward regardless of how the state was obtained**.
   Reversibility eliminates only trajectory storage: the descent variable is 1280 floats = 5.1 KB/sample, while the
   per-step second-order graph is ~1.5–2.3 GB at micro-batch 12 (this graph, not the states, is what pegged 16GB in
   `step1_mech.py`). Saving at K=128: **~7.8 MB of a 3–4 GB peak ≈ 0.2%.** Crossover where states matter: K ≈
   5,000–30,000 — unreachable and pointless.
2. **Deep K has no demonstrated value here**: S2 think-longer saturates at ~4 descent steps (0.038→0.010, flat to 32).
3. **The diffusion-sampler variant is preempted**: DOODL (ICCV 2023) already published constant-memory backprop through
   sampling via EDICT couplings; MALI (ICLR 2021) covers reversible-solver exact gradients; AlignProp/DRaFT chose
   checkpointing/truncation because state ≪ graph. Also plain GD (the actual EBT sampler) is not closed-form
   invertible — you'd have to switch to Maclaurin momentum + fp64 state buffers, whose 10 KB triviality itself proves
   states were never the problem.

**Salvage (genuinely good):** a closing Ch. D subsection — *"Reversibility across three axes: depth (RevNet), time
(TempVerseFormer / Ch. D), computation (inner loop) — and why the computation axis cannot pay"* — with the
bits-per-step volume-contraction accounting (**attributed to Maclaurin 2015, applied** — not claimed as a new theorem)
and a measured 4-way memory panel (naive / truncate_mcmc / checkpoint / reversible, K ∈ {4,8,32,128}, ~0.5 GPU-day;
guard the naive arm against WDDM spill: abort the arm if step > 10 s). This completes the TempVerseFormer arc, answers
the question on the record, and inoculates against the "why didn't you reverse the inner loop?" reviewer question.
A claimed win here would contradict Ch. D's own "do NOT claim it beats checkpointing" clause.

---

## 2. Panel verdicts (summary)

| Direction | Verdict | Confidence | One-line |
|---|---|---|---|
| A — drop EBT; diffusion + testbed + Ch. D + negative section | **GO** | 0.80 | Highest-EV package; EBT negative publishable if rescoped + hardened with one overnight |
| B — user's plan: minimal MilliVid base → EBT swap | **CONDITIONAL** | 0.75 | Base yes; one budget-capped fair-scale EBT arm yes; "swap" alone is not the novelty; needs Ch. D attached for reversibility coherence |
| C — EBT re-homed as verifier (energy ranks samples / allocates compute) | **CONDITIONAL** | — | Premise unverified and contradicted by 2 measurements (E(mean)<E(target) ⇒ plausibly anti-ranks; argmin-of-16 stayed collapsed) + CW-VAE-entropy ρ≈0 precedent → 1-day Spearman ranking probe as optional appendix only, nothing load-bearing |
| D — reversible inner-loop optimization | **NO-GO** | 0.92 | ~0.2% saving vs checkpointing; DOODL/MALI preemption; fold into Ch. D as honest-negative subsection |

**Integrated recommendation** = A's shape + B's single fair-scale EBT run + D's salvage subsection + C's probe as
optional appendix. This satisfies the stated preference (EBT stays in the article, honestly) without betting the
article on P≈10%.

## 3. Staged plan with kill-gates

**Budget honesty (feasibility attack): ~15–21 person-days, 8–14 GPU-days (not 6–10), ~3–4 weeks calendar, all runs
overnight-sized on the 4060 Ti.**

- **STAGE 0 — hygiene, then cheap decisive gates (1–2 days, ~0.5 GPU-day).**
  Order is mandatory: (a) latent standardization + save/resume + EMA **first**; (b) then patch and run
  `hvp_rev/mem_gate.py` — **the decisive, never-run pre-test of the memory-horizon headline**. Patches needed before
  trusting it: bootstrap CIs (the hard re-entry bin has only ~60–80 samples — as coded the kill decision is
  unpowered), force bf16 + smoke the 3920-token long arm (spill risk), add a **long-coarse-ONLY third arm** (48 coarse
  + 6 fine ≈ 984 tokens — otherwise the gate tests "long context helps", not the headline mechanism "long coarse
  context at a token budget flat can't afford"), 2 seeds before any kill verdict. (c) Clean logged re-run of the
  action-conditioned reg-vs-diff table (~2–3 h; `step1_mech.log` truncates at EBT step 1500 — the citable table
  currently does not exist). (d) 100-step timing probes for both the Stage-2 DiT config and the EBT config.
  **G0a:** no long-context re-entry advantage clearing the decoder-blur floor → memory-horizon headline dead →
  reframe as mechanism-study + negative + Ch. D (still a defensible article).
- **STAGE 1 — the EBT fair-scale shot (0.5–1 GPU-day, 1.5–2.5 person-days** — the reference-control script needs
  real mods: cosine, accum, FFN-4× plumbing, ctx-16, save/resume; the S1 control needs a learnable-α param group +
  best-of-N glue; run α both transplanted (30000) and movement-recalibrated, report both).
  **Scoping gate:** no E-gap flip and no sharpness trend by 10k post-warmup → write the hardened negative, do NOT
  spend the 60k paper-faithful run (27–67 h — only if Stage 1 trends positive).
- **STAGE 2 — minimal MilliVid base (4–6 person-days + 2–4 GPU-days).** Universal DiT, rollout scheduler, long-horizon
  eval on `loopmom_test_500x200` with per-horizon curves + re-entry stratification + fg-LPIPS/centroid (FVD demoted:
  I3D/Kinetics-RGB vs 64×64 grayscale). **Controls a reviewer will demand:** (i) **token-budget-matched flat arm** —
  same total context tokens as the hierarchical model, else any win is attributable to context length and the headline
  is a CW-VAE rerun; (ii) **regression-AR-rollout arm** — regression wins single-step-with-actions (fg-MSE 0.0014), so
  "generative modeling earns its keep at rollout/re-entry" is unsupported without rolling the regressor out
  (pre-registered hypothesis: it fades to the mean per E1; diffusion holds sharpness).
  **G2:** AR rollout stable (≤1 context-noise-augmentation iteration, +2 GPU-day cap — historically this eats weeks,
  so the cap is a hard scope fence) AND hierarchical beats persistence + flat on re-entry with bootstrap CIs; if
  hier ≤ flat, drop the hierarchy-wins headline, keep the long-context memory result, say so honestly.
- **STAGE 3 — reversibility module, hard 5-day box.** Guaranteed deliverable: the Ch. D three-axis subsection + 4-way
  memory panel. Upside bet (rollout-unroll at K beyond BPTT's OOM point, distribution-matching objective per E1):
  **blocked on a half-page architecture memo first** — the Stage-2 DiT rollout is a ~16-step sampling chain with no
  coupling structure; if the memo shows the sampler graph dominates (likely), pre-decided fallback: Ch. 5 becomes the
  explicit bridge-from-TempVerseFormer theory chapter with a stated applicability boundary ("reversibility pays on
  recurrences whose transition is the coupling itself; diffusion rollouts are the checkpointing regime") — that
  boundary statement itself completes the taxonomy and keeps Ch. 5 load-bearing.
- **STAGE 4 — seeds + writing (6–10 person-days + 2.5–4 GPU-days).** ≥3 seeds on ONE headline table (hier-vs-flat,
  6×30k runs ≈ 1.5–2.5 GPU-days), 2 seeds or CI-on-one-seed for the second; rewrite the stale master-doc sections
  **before** drafting; re-sweep prior art at submission time (DSA/DySink near-collisions are ≤60 days old).

## 4. Article skeleton (single-spine version, post-coherence-attack)

**Spine: "hierarchy earns its keep only where the memory horizon demands it."** One headline, not three.

1. **Intro / dissertation arc** — hierarchy (article 1) + reversibility (article 2) converge on resource-honest
   hierarchical video prediction. Pre-written null-branch framing: *article 1 proposed the hierarchy; this article
   measures the conditions under which it pays* (so even a null is a boundary-condition result, not a self-contradiction).
2. **The memory-horizon testbed + evaluation protocol** — looping/momentum dataset (67% off-screen, ~48-frame loop),
   re-entry-stratified foreground metrics; **"why not Memory Maze / TECO-Minecraft" subsection** (analytic loop period,
   exact re-entry ground truth, 16GB feasibility) to defuse benchmark-shopping.
3. **Mechanism study at matched compute** — fair 3-way (regression / diffusion / EBT, shared backbone); the
   action-conditioning correction; framed as *controlled replication* of regression-blurs / diffusion-commits
   (DIAMOND, SVG/SAVP cited) **with the EBT cell as the genuinely new part**: two-sided study (generation +
   reachability), official-code control, fair-scale run, scale scoping in the claim text.
4. **Hierarchical latent diffusion system + the memory-horizon result** — delta over CW-VAE/TECO/DLH stated explicitly:
   action-conditioned off-screen **re-entry** (not mere identity retention), sharp commitment (vs L2 blur), token-budget
   argument, with the token-matched flat control.
5. **Reversibility (Ch. D)** — with an explicit delta-over-TempVerseFormer paragraph (deterministic→stochastic
   store-the-noise; invertibility≠stability = Behrmann 2021 instantiated with measured fade; measured frontier with
   checkpointing-parity caption) + the **three-axis closing subsection** (bits-per-step = Maclaurin 2015 applied).
6. **Related work / honest scoping** — CW-VAE, TECO, DLH, NUWA-XL, Harvey'22 FDM, DIAMOND, Self-Forcing lineage,
   LSSVWM/WorldPack, DOODL/MALI/EDICT, DSA/VFScale, EBT-Policy, MilliVid. Explicit non-claims: no beats-checkpointing,
   no universal EBT refutation, no FVD headline.
7. **Conclusion** — one claim, its boundary conditions, and the two honest negatives that scope it.

**Format decision needed from the user:** journal article (Springer NCA shape, ~10–12 pp main → cut hard) vs
dissertation chapter (roomier). The plans differ; decide before writing.

## 5. Verification debts the review surfaced (do not skip)

- `mem_gate.py` was never run; `step1_mech.log` is truncated at EBT step 1500 (the 3-way table needs a clean re-run);
  every headline number is single-seed; master doc §2.6/§7/§10 contain the refuted "blur is task-inherent" conclusion.
- Citation-mechanism claims taken from training knowledge (DOODL, MALI, EDICT, AlignProp, DRaFT, NUWA-XL, FDM,
  DIAMOND, EBT-Policy, Self-Forcing ablations, 2506.05284) must be verified against the actual PDFs before appearing
  in the article.
- Check the local MilliVid LaTeX for any rollout/self-generated-context training component before claiming the
  Self-Forcing×hierarchical combination is unclaimed.
- Retry the EBT OpenReview forum before printing any "no published negative exists" sentence; if unreachable, drop
  the sentence.
