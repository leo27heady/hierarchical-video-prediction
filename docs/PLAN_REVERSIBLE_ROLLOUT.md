# Reversible temporal rollout training on the hierarchical CW-VAE backbone — plan

> Dissertation thread: extend **TempVerseFormer** (Kulyk, *Temporal Modeling with Reversible
> Transformers*, Research Square rs-6293520 / Springer NCA 2025 — reversible temporal transformer,
> memory independent of time-to-predict, validated on deterministic 2D-shape rotation) to
> **hierarchical, stochastic video prediction** (CW-VAE/DLH), using reversibility to enable
> **on-the-fly long-rollout training** that fixes autoregressive drift. Grounded by E1/E2 probes
> and the `reversible_temporal_proto.py` correctness demonstration.

## The thesis in one line
On-the-fly rollout training (BPTT through free-running prior steps) is the principled fix for AR drift
(E2 proved no *inference* trick helps); **reversibility is the memory enabler** that lets the rollout
window scale; the hierarchical-stochastic setting forces a clean treatment of the **no-forgetting
theorem** that the deterministic TempVerseFormer never had to face. That treatment is the novel core.

## Two axes — never conflate (the #1 rigor risk)
- **Depth-reversibility** (RevNet/RevViT): O(1) activation memory in network *depth*. Buys this
  backbone **nothing today** — the per-level transition is one GRUCell + small MLPs, no deep stack.
  It only pays off once a deep **reversible temporal transformer** (TempVerseFormer block) replaces
  the transition.
- **Time-reversibility** (TempVerseFormer / reversible RNN, MacKay 2018): reconstruct past states from
  the final state → activation memory independent of *time/rollout length*. **This** is what enables
  long-rollout BPTT. It is the valuable axis here.

## The no-forgetting tension (the intellectual core) — and its resolution, demonstrated
**Theorem (MacKay 2018):** a perfectly reversible recurrence is bijective → **cannot forget**. This
collides with CW-VAE/DLH, which forget *on purpose* (clockwork 0/1 gating `s=e·s_change+(1−e)·s_prev`,
abstraction depth K = discarding fast detail). `reversible_temporal_proto.py` makes all of this concrete:
- **P1** additive-coupling block inverse is exact (4e-16, float64). ✓
- **P2** the chain reconstructs all states from the final state + stored per-step conditioning — but
  **float32 drift explodes to ~1e3 over T=64** ⇒ time-reversal **needs Lipschitz/spectral-constrained
  coupling** (else reverse-mode gradients are wrong). A first-class finding.
- **P3** a stochastic transition is invertible **iff the per-step noise ε is stored** (store-the-noise,
  invert-the-deterministic-core); wrong ε → reconstruction breaks. ✓
- **P4** a hard clock gate is **non-invertible**; **reversal-with-forgetting** (store the 1-bit mask +
  pre-gate state) reproduces the chain exactly. ✓
So: keep DLH's forgetting in the recurrence; pay a *constant* O(rollout) cost of stored {ε, gate-bits}
(not the full activation stack); Lipschitz-constrain the coupling for numerical correctness.

## Honest framing (for the committee)
- **Claim strongly (novel):** on-the-fly rollout training for *hierarchical stochastic* video, with the
  depth-vs-time distinction made precise and the no-forgetting collision resolved (decouple axes /
  store-the-noise / reversal-with-forgetting / Lipschitz coupling). Stands even with vanilla checkpointing.
- **Claim modestly (enabler, measured):** reversibility = a larger BPTT window at fixed memory. Never
  "what makes it work," only "what makes it scale" — show a memory-vs-window curve.
- **Pre-register the null (speculative):** that longer windows / reversibility improve long-horizon
  *quality/stability*. Tie any such claim to measured fg-LPIPS/FVD/drift. (E1 showed the adaptive-clock
  signal is weak on MMNIST; be ready for a clean null and report it.)

## Staged plan (cheapest decisive first)
- **E0 — exposure gap (≈1 h, no training).** Measure teacher-forced vs open-loop fg-MSE/fg-LPIPS@25/50
  on CW-VAE. If the gap is small the rollout thesis is weak — find out first.
- **E1 — short-window rollout training (M=4–8), NO reversibility.** Add a training mode: after ctx
  posterior steps, run M prior steps *with grad*, decode, add rollout recon loss (+ scheduled sampling);
  plain gradient checkpointing. **The actual contribution, decoupled from reversibility.** Decisive: does
  it close the E0 gap (fg-LPIPS@50)?
- **E2 — memory-vs-window curve.** Profile peak memory vs M for {full storage, checkpointing,
  ε-cached reversible transition}. The *only* place reversibility's provable benefit is shown; no quality
  claim needed.
- **E3 — the theorem, as a figure.** A perfectly-invertible (no-gating) recurrence fails to forget /
  fails the adaptive clock; reversal-with-forgetting recovers it. Turns the tension into a result.
- **E4 — does the longer window buy quality? (only if E1 positive).** Scale M=4→16→32 via E2 savings;
  plot long-horizon drift + fg-LPIPS/FVD vs M. Pre-register the null. Foreground/LPIPS/FVD only.

## Where the code touches
- New reversible temporal block: `scripts/reversible_temporal_proto.py` (→ module). Add **spectral norm**
  on F/G (the float32 finding).
- Rollout-training path parallel to the teacher-forced loss, reusing the prior-mode step; ε-caching at the
  sampling site; checkpoint/reversible wrapper around the conv enc/dec (dimension-changing → not invertible).
- Eval on fg-MSE/fg-LPIPS/FVD (never aggregate SSIM/PSNR — background-dominated on MMNIST).

## Risks
- Axis conflation (depth↔time) — the likeliest reviewer kill-shot. Name the axis on every claim.
- Stability is a *memory* result, not a *stability* result — measure it, expect a possible null.
- float32 time-reversal is numerically unstable without Lipschitz control (P2) — spectral-normalize.
- Depth-reversibility buys nothing until a deep transformer transition exists.
- MMNIST is background-dominated + hierarchy-null — use fg/LPIPS/FVD; consider a real-events dataset
  (KTH/TaxiBJ, already in the TempVerseFormer paper) where the clock and long-horizon actually bite.
