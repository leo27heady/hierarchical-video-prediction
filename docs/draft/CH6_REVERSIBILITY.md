# Chapter 6 — Reversibility: the exactly invertible stochastic recurrence and its limits

> **Status (2026-07-07):** journal-register prose, content-complete; all numbers measured
> (sources: `docs/THEORY_CHAPTER_D.md` / `hvp_rev/theory.py` for T6a; `hvp_rev/mem_frontier.py`
> for T6b; `logs_rhcp/mem_panel.md` for T6c). Figures F9–F12 of the skeleton list are the
> placeholders here. Numbering: Chapter 6 per the canonical order fixed in CH1_INTRO.md.
> **Triage (2026-07-10, approved):** NON-CORE for the article's central claim (long-span context
> packaging); retained in the working manuscript for the dissertation's [A4] mapping.
> Appendix-vs-separate-paper decided at venue constraints.

## 6.1 Claim

Long-horizon training is memory-bound before it is compute-bound: backpropagation through a
$K$-step rollout stores $O(K)$ activations, and on a 16 GB device this is the binding constraint
long before wall-clock is. TempVerseFormer [REF: A2] removed this constraint for deterministic
dynamics by making the temporal transition itself invertible. This chapter extends the result to
the stochastic, multi-level recurrences that video generation actually uses, and then delimits —
by measurement, not argument — where the extension can and cannot pay.

**Theorem (informal; store-the-noise).** Let the per-step transition be additive two-branch
coupling with the step's noise injected as a conditioning argument,
$$(y_1, y_2) = \bigl(s_1 + F(s_2, \epsilon_t),\; s_2 + G(y_1)\bigr),$$
and let the noise samples $\epsilon_t$ drawn on the forward pass be stored. Then the inverse is
closed-form — $s_2 = y_2 - G(y_1)$, $s_1 = y_1 - F(s_2, \epsilon_t)$ — for *arbitrary* coupling
functions and arbitrary stored conditioning, so a $K$-step free-running rollout can be
backpropagated by reverse replay: states are reconstructed backwards, local graphs are rebuilt
one step at a time, and recurrent-core activation memory is $O(1)$ in $K$ (the stored noise
costs $O(K \cdot D_\epsilon)$ — one tensor per step, no layer factor, no graph factor). The
gradients so obtained are exactly those of BPTT.

## 6.2 Verified properties (Table 6.1)

| # | property | measured result |
|---|---|---|
| P1 | block invertibility (unconstrained / Lipschitz-constrained) | fp64 **5.5e-15** / fp32 3.1e-5 |
| P2 | stochastic store-the-noise invertibility | exact iff ε stored (**8.4e-13**); wrong ε breaks it (error ~10.9) |
| P3 | invertibility ≠ stability: ‖δ₁₀₀‖/‖δ₀‖ under perturbation | unconstrained 3.3e2; **Lip(F) ≤ 1 still explodes (2.7e4)**; Lip ≤ 0.1 → 1.5; Lip ≤ 0.02 → 1.0 |
| P4 | reverse-replay gradients vs BPTT | **cosine similarity 1.000000** |

P2 makes the information accounting exact: the stored noise is precisely the price of inverting
a stochastic step — replay the wrong noise and reconstruction is destroyed. P3 is the
theoretically interesting entry: **invertibility does not imply stability.** Additive coupling
is ≥1-expansive (per-step Lipschitz ≈ 1 + Lip(F)), so even a coupling with Lip(F) ≤ 1 amplifies
perturbations by 2.7e4× over 100 steps; bounded propagation requires Lip(F) ≲ 0.1. This
instantiates the invertible-network stability analysis [REF: Behrmann et al., i-ResNet line] on
temporal recurrences and honestly explains the fade observed in trained reversible predictors:
exactness of the *inverse* says nothing about conditioning of the *forward* trajectory, and
trajectory stability must be engineered separately. [FIG F10 — PLACEHOLDER: perturbation growth
vs Lipschitz bound, four exactly-invertible recurrences, 3.3e2× → 1.0×.]

## 6.3 The measured memory frontier (Table 6.2)

Measured on the real training loop (Adam, three-level recurrent core, batch 32, single RTX 4060
Ti 16 GB):

| K | BPTT (MB) | checkpointing | reversible | BPTT ms/step |
|---:|---:|---:|---:|---:|
| 40 | 12,162 | 1,471 | 1,091 | 740 |
| 80 | 23,305* | 1,637 | 1,125 | 43,797* |
| 120 | **OOM** | 1,801 | 1,152 | — |
| 160 | **OOM** | 1,967 | 1,176 | — |

*At K = 80 BPTT spills into shared memory (~60× slower); it hard-OOMs at K = 120. The reversible
implementation stays at ≈1.1–1.2 GB through K = 160. The caption is pre-committed and carried
everywhere this table is quoted: **a constant-memory capability relative to vanilla BPTT, and an
approximate Pareto equivalence with gradient checkpointing — not a memory victory over
checkpointing.** [FIG F9 — PLACEHOLDER: activation memory vs K, three curves.]

## 6.4 Delta over TempVerseFormer

Three items are new relative to [REF: A2]: (i) the deterministic → stochastic extension itself —
the store-the-noise construction and its exactness proof/measurement (P2), which is what a
*generative* video core requires; (ii) the stability characterization (P3), which explains
honestly why exact invertibility did not deliver stable long rollouts for free; and (iii) the
no-forgetting information accounting in the sense of [REF: MacKay et al. 2018]: an invertible
transition forgets nothing, so everything BPTT would store is reconstructable — the theorem that
converts memory from a stored resource into a recomputed one.

## 6.5 The three-axis taxonomy and the inner-loop no-go

Reversibility can be exploited along three orthogonal axes of the training computation:

1. **Depth** — across layers within a step (RevNet and descendants): the literature baseline.
2. **Time** — along the physical time of the modeled process (this chapter): here reversibility
   **pays**, because what BPTT stores is the $O(K \cdot L \cdot D)$ activation trajectory, which
   is exactly what the inverse coupling reconstructs (Table 6.2).
3. **Computation** — along the *inner optimization loop* ($K$ pseudo-time steps of gradient
   descent, Langevin sampling, or a diffusion sampler inside one generation step): here
   reversibility **provably cannot pay against checkpointing**, and we close the axis by direct
   measurement rather than argument.

The four-way measurement (naive full-graph / truncated / checkpointed / reversible
differentiation through the inner loop; $K \in \{4, 8, 32, 128\}$, micro-batch 12, fp32):

| variant | K=4 | K=8 | K=32 | K=128 |
|---|---|---|---|---|
| naive (full graph) | 18.14 GB, 17.1 s/step (spill) | **OOM** | OOM | OOM |
| truncated | 5.282 GB | 5.282 | 5.282 | 5.282 GB, 7.85 s/step |
| checkpointed | 5.282 GB | 5.283 | 5.286 | 5.296 GB, 7.85 s/step |
| reversible | 5.282 GB | 5.282 | 5.282 | 5.282 GB, 15.37 s/step |

The reverse step of a descent loop must rebuild the differentiable graph of $\nabla E$ (a
second-order, `create_graph`-class object) from a fresh forward pass *regardless of how the
trajectory state was obtained*; reversibility can eliminate only the trajectory-state storage.
Measured: the reversible variant saves **14 MB = 0.26% of peak** versus checkpointing, at ~2×
wall-clock. The per-step state $(y, v)$ is 0.117 MB, so checkpointing's total state storage at
$K = 128$ is 15.0 MB = 0.28% of its peak — the ceiling on what this axis can ever reclaim; the
analytic accounting band (bits-per-step vs volume contraction, attributed to [REF: Maclaurin et
al. 2015] and *applied*, not re-proven) is ≈0.6–1.0%, with the state-dominates-graph crossover
at $K \approx 5{,}000$–$30{,}000$ inner steps — far beyond practical refinement depths.
Qualification (recipe-scoped): a first-order, many-step store-the-noise sampler has a genuinely
$O(K)$ activation peak where reversibility saves real memory — but checkpointing ties it there
too (the temporal Pareto tie), and reversible-sampler variants for diffusion loops are already
published [REF: EDICT; DOODL; MALI]. [FIG F11 — PLACEHOLDER: three-axis diagram. FIG F12 —
PLACEHOLDER: the 4-way panel.]

**Applicability boundary (the chapter's closing sentence).** *Reversibility pays on recurrences
whose transition operator is the invertible coupling itself — the temporal rollout of the
modeled process — and cannot pay on inner optimization loops, whose reverse step is dominated by
per-step derivative-graph reconstruction; that regime belongs to gradient checkpointing.* In
this article the theorem plays the enabler role: it makes constant-memory rollout-through-time
training *affordable* for the distribution-matching objectives of future work (Ch. 8); the
Chapter 4 system itself is trained single-window and does not presuppose it.
