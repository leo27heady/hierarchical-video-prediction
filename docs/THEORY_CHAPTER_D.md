# Chapter D (theory): the invertible stochastic recurrence

The honest, non-nullable backbone of the dissertation. Reversibility is homed here as a **correctness/enabler
theorem**, NOT as a stabilizer (the mechanism review settled that invertibility ≠ stability). Everything below is
**measured on the real conv core** (`hvp_rev/theory.py`, `hvp_rev/rollout_train.py`), not asserted.

## The claim

> A hierarchical multi-rate recurrence whose per-level transition is RevNet additive coupling — extended to a
> **stochastic** step via store-the-noise — is exactly invertible, so a K-step free-running rollout can be
> backpropagated by reverse-replay at **recurrent-core activation memory constant in K**, with gradients **provably
> equal to BPTT**. Reversibility is the constant-memory *enabler* of long-horizon rollout training; any *stability* of
> the rollout comes from a separate **Lipschitz constraint on the coupling**, not from invertibility.

## Verified properties (measured)

| # | Property | Result |
|---|---|---|
| T1 | Block invertibility (unconstrained **and** Lipschitz coupling) | fp64 **5.5e-15**, fp32-Lipschitz **3.1e-5** |
| T2 | **Stochastic** store-the-noise invertibility | exact iff eps stored (**8.4e-13**); wrong eps breaks (**10.9**) |
| T3 | Invertibility ≠ stability (perturbation growth over K=100) | see below |
| T4 | Reverse-replay gradients == BPTT (Lipschitz + stochastic) | **grad-cos 1.000000** |
| — | Constant-memory frontier (real training loop) | BPTT **OOMs at K=120**; reversible flat ~1.2 GB (Panel A) |

### T3 — the stability characterization (the interesting result)
Perturbation growth ‖δ_100‖/‖δ_0‖ of a free-run reversible recurrence, all **exactly invertible** (det-Jacobian ≡ 1):

| coupling | ‖δ_100‖/‖δ_0‖ | state-norm ratio | regime |
|---|---|---|---|
| unconstrained | 3.3e2 | 2.6e2 | explodes |
| spectral-norm, Lip(F)≤1 | 2.7e4 | 1.9e5 | **still explodes** |
| spectral-norm, Lip(F)≤0.1 | 1.5 | 8.9 | bounded |
| spectral-norm, Lip(F)≤0.02 | 1.0 | 1.7 | tightly bounded |

**Finding:** invertibility holds for *all* of these, yet the dynamics span explosive→bounded purely by the Lipschitz
bound on F. Crucially, **spectral-norming F to ≤1 is NOT enough** — additive coupling `s+F(s)` is inherently
≥1-expansive per step (step-Lipschitz ≈ 1+Lip(F)), so bounded rollout requires *small* Lip(F) (≲0.1) or a different
transition. This is the precise, honest reason the trained RHCP **faded** (centroid-63): an unconstrained coupling has
uncontrolled singular values. Invertibility contributes nothing to stability; Lip(F) is the only lever.

## Honest scope (what to claim, what not to)
- **Claim:** exact-gradient, constant-memory long-horizon rollout backprop via a *provably-invertible stochastic*
  recurrence (extends the deterministic TempVerseFormer; the stochastic-ELBO version is the novel step). Plus a clean
  characterization of when such a recurrence is dynamically stable (small Lip(F)).
- **Do NOT claim:** that reversibility stabilizes rollout (it doesn't — T3), or that it beats gradient checkpointing on
  memory (Pareto/throughput point, not a memory win — AlignProp + this repo's own measurement).
- **Prior art to cite:** MacKay et al. 2018 (reversible RNN, memory framing + no-forgetting); CrevNet (reversible
  *deterministic* video predictor); i-ResNet / Behrmann 2019-2021 (Lipschitz for invertible-net stability);
  Self-Forcing (the rollout-training stabilizer reversibility merely *enables*).

## The role in the dissertation
Chapter D is the theorem + figures (T1–T4 + the memory frontier). It is non-nullable because it is a *correctness*
result, not a benchmark. It underwrites Chapter C: reversibility lets the hierarchy study train longer rollouts at
constant memory, and the Lipschitz result tells us to spectral-norm the coupling (small Lip(F)) so the study's models
don't fade. Run: `venv/Scripts/python.exe -m hvp_rev.theory`.
