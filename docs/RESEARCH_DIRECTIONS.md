# Research Directions — forward plan

> Forward-looking options after the findings in `docs/SUMMARY.md`. One PRIMARY direction (A), two reviewer-proof
> FALLBACKS (B, C) that need no reversibility, and a THEORY thread (D) that can pair with any. All scoped to a single
> RTX 4060 Ti 16 GB. Each direction lists the claim, the key experiments, novelty, risks, and the decision gate.

---

## Cross-cutting prerequisites (needed by A/B/C)

### P1 — The dataset (gates everything)
MMNIST/colour-ball are **too simple** (short, no off-screen memory, free-to-persist black bg) and reproduce our null.
Build **ONE long looping action-conditioned world**:
- A viewport **panning** over a wide drifting-texture canvas with colored sprites at *fixed world positions*.
- Ternary action `{pan-left, pan-right, hold}` scrolls sprites off-edge → they **must reappear** on pan-back.
- Knobs: canvas size, sprite count, off-screen-return-rate, background drift speed (slow = a real coarse-level job).
- ≥256 frames/clip, 64×64, action+pose metadata. This satisfies MilliVid's 4 dataset criteria (long; many clips;
  content exits+re-enters; fine-grained conditioning) on a tiny compute budget.

### P2 — The scaled-down MilliVid (the substrate)
64×64, patch 8, **3 levels**, `num_frames_per_level=[4,8,16]`, `extra_budget=64` → `S≈1216`. AE batch 16-32, DiT batch
4-8 + grad-accum. Reuse MilliVid's tokenizer, DiT, rollout strategy, and **the full metrics suite** (DINOv2,
LightGlue keypoints, FVD/FID/LPIPS) + baselines (FramePack, full-res). First **reproduce learned-pyramid > cascaded**
on our data — if the hierarchy nulls here, the *task* (P1) is mis-designed.

### P3 — The evaluation (already solved by MilliVid)
Consistency: PSNR/SSIM/LPIPS + DINOv2 cosine + keypoint-match-count, conditioning on context then comparing the rollout
to ground truth. Quality: FID/FVD. **LPIPS is the most robust consistency metric** under slight drift. Never headline
aggregate SSIM.

---

## Direction A (PRIMARY) — Reversible rollout-aware training

**Claim.** MilliVid trains each rollout step independently on **clean** ground-truth context → exposure bias; at
inference, context is the model's own (erroneous) output → drift. We train through a **K-step self-conditioned
unroll** with a **distribution-matching** objective; **reversible coupling along the rollout-step axis** makes the long
unroll constant-memory, so we afford a large K on one GPU.

**Math.** `ĉ_0 = real context; for k=1..K: ĉ_k = Sample_DiT(context=ĉ_{<k}); L = DistMatch({ĉ_k}, GT)`. BPTT through
K·T transformer evals (T = inner diffusion steps) → memory O(K·T) (OOMs). Reversible step
`u_k=u_{k-1}+F(v_{k-1},ĉ_k); v_k=v_{k-1}+G(u_k,ĉ_k)` (exact inverse) reconstructs states on backward → ~constant in K;
store the per-step diffusion noise ε_k (the only non-invertible quantity — store-the-noise).

**Novelty.** "Stochastic reversible recurrence over a learned hierarchical latent enabling longer rollout-aware
training windows" is not in the literature; it extends TempVerseFormer (deterministic, single-scale, time-axis) to the
stochastic, hierarchical, rollout-step axis.

**Key experiments / the money plot.**
1. Reproduce single-step MilliVid on P1/P2 (baseline).
2. K-step unroll with **ordinary gradient checkpointing** — does it beat single-step on long-range consistency?
3. **Money plot:** K vs consistency under fixed 16 GB, **reversible-unroll vs checkpointed-unroll** — reversibility
   must buy a clearly larger K (and better consistency) at the same VRAM.
4. Gradient-equivalence: reversible grads == BPTT (cosine ~1, drift control via spectral-norm + 32-step anchors).

**Risks.** (i) MSE unroll → mean-collapse (E1) → objective MUST be distribution-matching/diffusion-forcing. *Highest.*
(ii) Reversibility may match checkpointing at small K (the DLH null) → the money plot must separate clearly. (iii)
Stochastic multimodality on a reversible (deterministic-coupling) state — handled by store-the-noise + a stochastic
latent head; this is the research depth.

**GO/NO-GO gate.** Experiment 2 (checkpointed unroll beats single-step) **before** any reversibility code. If it
doesn't help → fall back to B/C (≈2 weeks spent, not a semester).

---

## Direction B (FALLBACK 1) — Adaptive / learned token-budget allocation

**Claim.** MilliVid's `num_frames_per_level` is a **fixed** hand-set tuple. The paper's thesis is "given fixed S,
decide what to fill it with" — but it never makes that decision *content-driven*. We learn a per-clip (or per-step)
**adaptive budget**: spend more fine tokens when the scene is detail-rich / changing, more coarse frames when it's
static, under a fixed S.

**Why it's safe.** Tiny models; no reversibility; uses MilliVid's substrate directly; a clean ablation against the
fixed budget. **Note:** this is *budget* allocation (how many tokens per level), NOT the dead uncertainty-clock (which
gates *whether a latent updates*) — the failure mode there (entropy doesn't track error) is about a different signal;
here the signal is reconstruction-error / token-utility, which is measurable.

**Key experiments.** Learned budget vs fixed `[4,8,16]` vs other fixed splits, on consistency+quality+token-efficiency;
what the learned policy does on static vs dynamic clips; a Pareto frontier (consistency vs token budget).

**Risk.** May only tie the fixed budget on simple data → needs the P1 task to have *heterogeneous* detail/motion so
adaptivity has something to exploit.

---

## Direction C (FALLBACK 2) — Controlled science-of-hierarchy study

**Claim.** *When* does a learned token pyramid beat a cascaded (mean-pooled) or flat latent? MilliVid shows it does on
Minecraft but doesn't characterize *why/when* (their Q3 surprise — "mirrored FramePack beats FramePack" — is left as a
hypothesis). We build a **parametric synthetic generator** (controllable off-screen-memory rate, coarse-resolvable
structure, texture complexity, sequence length) and **map the regime** where the learned hierarchy / coarse-to-fine
rollout helps vs nulls.

**Why it's the most reviewer-proof.** Turns our *documented nulls* (the clock/hierarchy capacity-dumps unless the data
has multi-scale structure) into the **central finding**: a precise account of the conditions under which multi-scale
video generation pays off. Runs on tiny models; no reversibility; an analysis paper the big labs can't afford to do.

**Key experiments.** Sweep the generator knobs; for each regime measure learned-pyramid vs cascaded vs flat on
consistency/quality; derive the boundary (e.g. "hierarchy helps once off-screen-return-rate > x AND coarse-resolvable
structure carries > y bits"). Explain MilliVid's Q3.

**Risk.** Lower "wow" than A; strongest as a companion to A or B (the analysis chapter).

---

## Direction D (THEORY thread, pairs with A) — Stochastic reversible recurrence

**Claim.** A correctness contribution independent of any resource win: the **first provably-invertible stochastic-ELBO
hierarchical recurrence** — resolving the MacKay no-forgetting collision (a stochastic draw breaks bijectivity) via
**store-the-noise + Lipschitz coupling**, with reverse-replay gradients **proven exact** vs BPTT. Non-nullable (it's a
theorem + a gradient-equivalence figure), and it's the formal backbone that justifies the reversibility in Direction A.

**Key deliverable.** Panel-A figure: gradient cosine vs BPTT + fp32 reverse-drift, vs sequence length, with/without
spectral-norm+anchors (cos>0.9999 under the Lipschitz bound where naive explodes ~1e3). Plus the dissertation arc:
TempVerseFormer (done, deterministic single-scale) → this (stochastic + hierarchical, where no-forgetting first bites).

**Detail:** `docs/RHVP_ARCHITECTURE.md` (the standalone reversible-core version) + `hvp_rev/rhvp_m0_memory.py`,
`scripts/reversible_temporal_proto.py`.

---

## Recommended sequencing

```
P1 (dataset) ─► P2 (scaled MilliVid, reproduce pyramid>cascaded)
                    │
                    ├─► A.exp2 checkpointed K-unroll beats single-step?  ──┐
                    │        YES → A (reversible unroll, + D theory)        │ the signature paper
                    │         NO →────────────────────────────────────────┘
                    │
                    └─► (any time) B (adaptive budget) and/or C (science-of-hierarchy)  ← reviewer-proof fallback/companion
```

**Bottom line.** Pursue **A** with **D** as its formal backbone; gate hard on P1 (real off-screen memory) and A.exp2
(unroll must help before reversibility). Keep **B+C** as the safety net that turns our nulls into a publishable
analysis using no reversibility. Either branch finishes the dissertation without another pivot.
