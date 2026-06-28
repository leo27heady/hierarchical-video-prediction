# Improving the CW-VAE backbone — investigation & gated plan

> Pivot: CW-VAE (`../cwvae/cwvae_torch.py`) is the chosen backbone (sharper, more stable, faster than DLH).
> This is the output of a 10-agent web-grounded + code-grounded + prior-art-aware investigation of four
> directions: transformer/BLT, adaptive clock (DLH-like), per-pixel/spatial, and sharp-renderer fusion.
> Grounded in established project findings (memories + `compare_curves.json` + `c0_headroom.json`).

---

## The one-line conclusion

**The honest lever is the RENDERER, not the dynamics.** The three directions you proposed (transformer,
adaptive clock, per-pixel) all restructure the *latent dynamics* — but every piece of evidence in this
project says the dynamics/hierarchy is the **marginal-to-preempted** axis, while the **renderer/likelihood
(sharpness)** is the unsolved, high-value, *non-preempted* one you actually care about.

## What the evidence already settles (so we don't relitigate it)

| Finding | Implication for your directions |
|---|---|
| **A.** Temporal hierarchy is a ~null on MMNIST (flat ≈ hierarchical; built transformer found cross-stage conditioning **+0.013**) | More hierarchy depth / a transformer core **won't** move quality on this data. |
| **B.** CW-VAE prior **entropy doesn't track error** (ρ≈0, anti-correlates with horizon) | A **BLT entropy-adaptive clock is dead** here. Any clock must be grounded on real predictive *error*, not entropy. |
| **C.** **DLH already is** "CW-VAE + learned content-adaptive clock" and beats CW-VAE at 7M vs 12M params | The adaptive-clock direction is **mostly built**; only a differentiable/budgeted error-grounded variant that *beats DLH* is novel. |
| **D.** A transformer/BLT cascade was **already built** (`feat/continuous-joint-blt`); teacher-forced works, the real bottleneck is **AR rollout drift** (anti-baking didn't fix it) | The transformer surgery is **reinvention**; the only live fragment is a **sequence-level rollout objective**. |
| **E.** Blur = the **Gaussian-L2 likelihood**, not the latents (C0: a perceptual decoder on *frozen* codes hit **GT-level lap-var**, LPIPS −0.085). DLH improves dynamics, **not the renderer**. | A **sharp renderer is genuinely novel** vs DLH and directly fixes sharpness. |

## Ranked directions (honest verdicts)

1. **Sharp-renderer fusion — PURSUE (the only confound-controlled win).** Perceptual/adversarial or short-step
   diffusion decoder on CW-VAE's bottom `[s,det]` output. Headroom already proven (C0). The catch is the
   **perception–distortion frontier**: a naive sharp decoder cost +45% fg-MSE in C0b, and fg-MSE is the *only*
   axis the stochastic backbones beat alternatives on (long-horizon identity). → must be **fg-MSE-floored and
   FVD-gated** (lap-var/LPIPS are gameable into incoherent texture).
2. **Sequence-level rollout objective (Diffusion-Forcing / Self-Forcing) — CHEAP PROBE.** The one live fragment of
   the transformer direction: attack AR drift with a rollout-distribution objective on the **stochastic latent**
   (not the architecture). Neither DLH (teacher-forced) nor the prior BLT branch (scheduled-sampling, *failed*)
   did this. High-risk on a 20-dim KL-regularized latent; **probe before building**.
3. **Adaptive clock (differentiable + budgeted, error-grounded) — LOW PRIORITY / must beat DLH.** A PonderNet-style
   *soft, budgeted* tick gated on the **post‖prior KL** CW-VAE already computes (the differentiability DLH's hard
   argmin lacks), keeping CW-VAE's sharp renderer = "DLH done right." Novel sliver, but Finding A makes a **null
   the modal outcome on MMNIST**, and it must be benchmarked head-to-head vs the ported DLH.
4. **Per-pixel / spatial adaptive — DEPRIORITIZE.** Skip-Conv/DeltaCNN/SACT are **FLOP-efficiency** mechanisms
   (3–7× speed), not sharpness/stability wins — orthogonal to your stated goal. The only quality-relevant variant
   (object-slot, SAVi/SlotFormer) is a **full backbone replacement**, not a CW-VAE improvement.

## The benchmark problem (cross-cutting, important)

MMNIST is a **bad testbed** for *every* dynamics direction: it's background-dominated (black scores SSIM 0.76)
**and** hierarchy-null (constant-velocity motion, no real multi-timescale events). If you want adaptive
allocation / a clock to *show value*, it needs a dataset with genuine events (collisions, KTH actions, or the
SO(3) tumbling data). On MMNIST, the renderer is the only thing that can visibly move. **Decide the dataset
before investing in any dynamics direction.**

## Staged path (cheapest decisive first; all on the CW-VAE backbone)

- **E0. Baseline/FVD hygiene (once, reused).** Lock black / persist / mean-context baselines + an fg-mask + **FVD
  (I3D)** harness. Every later decision = fg-MSE / LPIPS-fg / **FVD**, never aggregate SSIM/PSNR.
- **E1. Clock-premise pre-check (zero training, ~1–2 h).** Instrument the *existing* CW-VAE checkpoint: emit
  per-level one-step `g = KL(post‖prior)` over ≥64 clips; test whether `g` **spikes at events** (correlate with
  per-frame foreground-change |fgₜ−fgₜ₋₁|), with a bootstrap CI. **Kill rule:** if `g` shows no event structure
  (like the dead entropy signal in Finding B), the adaptive-clock direction is dead *before any training*.
- **E2. Rollout-drift probe (zero training, ~15 lines in `CWVAE.open_loop`).** Add residual-noise carry on the
  free-run prior latent; sweep {0, 0.1, 0.25, 0.5}; measure fg-MSE + fg-LPIPS at +50/+100/+150 best-of-N matched.
  **Go rule:** if it flattens the drift curve → justify a Diffusion-Forcing *training* term; else drop the whole
  rollout/transformer direction.
- **E3. dec_stddev knob (1-line retrain).** `dec_stddev ∈ {0.3, 0.5}` (paper leaves it at 1.0) — the one
  architecture-free absolute-sharpness lever; measure lap-var + fg-MSE + FVD.
- **E4. THE renderer gate (the real investment).** Train a perceptual+light-adversarial (or short-step diffusion)
  decoder on CW-VAE's frozen `[s,det]` codes, sweeping λ and adversarial weight, with a **hard fg-MSE floor**
  (reject any config beyond baseline fg-MSE by >X%). **Decision metric = FVD** on 36→50 open-loop vs CW-VAE/DLH;
  lap-var/LPIPS reported but not the gate. Answers: "CW-VAE-or-better sharpness without erasing the long-horizon
  identity edge?"
- **E5 (only if E2 positive).** Promote residual-noise to a Diffusion-Forcing training term + Self-Forcing
  fine-tune on the **unchanged** GRU backbone (never the transformer surgery).

## Honest risks
- **Perception–distortion is the central risk** (E4): sharpness is not free; it trades fg-MSE, the one axis the
  backbone leads. fg-MSE floor + FVD gate are mandatory.
- **lap-var & LPIPS are gameable** (incoherent high-freq texture) → FVD is the gate, and FVD at sequence scale on
  16 GB is non-trivial to compute reliably.
- The long-horizon edge being protected is **small (~0.01 fg-MSE) and has no CI yet** — quantify it first.
- **Reinvention risk:** the transformer/clock work duplicates DLH (ICLR 2023) and the prior internal BLT branch;
  the win, where it exists, is in the **renderer** and the **rollout objective**, not the temporal architecture.
- Diffusion-Forcing on a 20-dim KL-regularized latent may be **redundant** (the prior already regularizes it).

## Bottom line
Improve CW-VAE by making its **renderer sharp** (E4), not by adding a fourth dynamics variant. Give your three
proposed directions a **fair, cheap shot** (E1/E2 are zero-training kill-tests), but expect the dynamics axis to
be null/preempted on MMNIST. If you want the *dynamics* directions to matter, **change the dataset** to one with
real events first.
