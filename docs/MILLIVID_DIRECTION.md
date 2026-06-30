# Direction: build on MilliVid — rollout-aware training, reversibility as the enabler

> Verdict after deep paper+code review (6-agent workflow) of MilliVid (MIT+TRI, arXiv 2606.09056, "Hierarchical Latents
> for Long-Range Consistency in Video Generation") as a dissertation base. Branch `explore/blt-decoder-gating`.

## 1. MilliVid as a base — CONDITIONAL YES (reuse, don't fine-tune)
- **What it is:** a learned hierarchical token *pyramid* per frame (coarse=layout/semantics, fine=texture, each level
  independently decodable) + a coarse-to-fine latent **diffusion transformer** (DiT-B) that rolls out coarsest-first
  then upsamples, with a context scheme that prevents super-resolution/temporal "stop-go" inconsistency. Beats FramePack
  on long-range consistency + quality on Loopcraft (200k Minecraft videos). The modern SOTA version of this thesis area.
- **REUSE (clean, dataset-agnostic):** the hierarchical **tokenizer** (`model_autoencoder.py`); the **DiT + per-token
  adaptive conditioning** + flow-matching V-prediction loss; the **coarse-to-fine rollout strategy**
  (`strategy_millivid.py`); the **full metrics suite** (DINOv2, LightGlue keypoints, FVD/FID/LPIPS/SSIM) + baselines
  (FramePack, full-res rollout) — this is your dissertation's eval harness for free.
- **Sizes / hardware:** AE ~214M + DiT ~130M (both "B"). Inference on 16 GB is trivial (~0.6 GB, but slow). Fine-tuning
  is BLOCKED (training latents not released; Minecraft AE/DiT doesn't transfer; action-conditioning hardwired to 3
  Minecraft actions). **Scaled-down from-scratch on your own data fits 16 GB** (batch 4–8 + grad-accum). Hardware is NOT
  the binding constraint — TASK DESIGN is. (4 levels at 64×64 hits a literal ZeroDivisionError → use patch 8, 3 levels.)

## 2. Reversibility — NO home as published; ONE honest home
**Stated plainly:** MilliVid trains by **independent per-step denoising on CLEAN context** (`model_video_universal.py:
115-148`) — no recurrence, no BPTT through the rollout (the only recurrence is the inference schedule under `no_grad`,
already O(1) memory). So your TempVerseFormer mechanism (constant-memory BPTT through a recurrence) has **nothing to
attach to**. Depth-reversible DiT = RevViT exists (the DLH checkpointing-null again); reversible tokenizer = re-spends
the token budget the paper exists to save.

**The one home:** change the OBJECTIVE to a **K-step rollout unroll** — chain K rollout steps so each step's *self-
generated* coarse output is the next step's context, and supervise the chain. THAT creates a real recurrence whose
activation memory grows ~linearly in K. **Reversible additive coupling along the rollout-STEP axis** (your mechanism,
retargeted from time to rollout-step) gives constant-memory BPTT → a larger K under 16 GB than checkpointing alone.
Reversibility = the **enabler** (memory), the objective is the **headline**.

## 3. The contribution (one sentence)
*"Rollout-aware training of a coarse-to-fine hierarchical latent video model: by unrolling K self-conditioned rollout
steps with a distribution-matching objective — made affordable on one GPU by reversible coupling along the rollout-step
axis — we cut the long-range drift that MilliVid's independent-per-step denoising leaves uncorrected."*
This attacks the real train/test gap (training sees clean context; inference compounds self-generated error = exposure
bias), uses the distribution-matching direction you already proved on CW-VAE (E1), and makes reversibility a genuine
memory enabler rather than a bolt-on.

## 4. Plan (cheapest validating experiment first)
- **Phase 0 — plumbing (wk 1):** run `demo.py`; write the `.frames`/`index.json`/dummy-action data packer (none ships);
  grayscale→3ch; `torchrun --standalone --nproc-per-node=1` (DDP mandatory); disable wandb + torch.compile (Windows).
- **Phase 1 — DESIGN THE TASK FIRST (wks 2–3, GATES everything):** ONE long (≥256-frame), looping, action-conditioned
  world — a panning viewport over a wide drifting-texture canvas with colored sprites at fixed world positions; ternary
  action {pan-left/right/hold} scrolls sprites off-edge so they MUST reappear on pan-back. Real off-screen memory +
  a real job for the coarse level. **Plain bouncing MMNIST fails 3 of MilliVid's 4 dataset criteria → do not use it as a result.**
- **Phase 2 — scaled-down from-scratch (wks 3–4):** AE then DiT at 64×64, patch 8, 3 levels, `num_frames_per_level
  [4,8,16]`, `extra_budget 64` → S≈1216; AE batch 16–32, DiT batch 4–8 + grad-accum. Reproduce **learned-pyramid >
  cascaded/flat** on YOUR data with the bundled metrics. If the hierarchy nulls here, the task is mis-designed.
- **THE GO/NO-GO GATE (~2-wk spike, BEFORE any reversibility code):** implement the K-step unroll with **ordinary
  gradient checkpointing** (no reversibility yet), largest K that 16 GB allows; show it **beats** the single-step
  objective on long-range consistency (DINOv2/keypoints/FVD). **Only if the unroll helps** add reversible coupling and
  show it buys a larger K at the same VRAM. Money plot = *reversible-unroll vs checkpointed-unroll, K vs consistency
  under fixed 16 GB*.

## 5. Risks + the fallback (a real paper on its own, no reversibility)
- **Risks:** MSE unroll fades to the mean (E1 → must use distribution-matching/diffusion-forcing, not latent-MSE); task
  without off-screen memory nulls (Phase-1 gate); RevNet coupling is deterministic but video is multimodal (need a
  stochastic latent on the reversible state — the dissertation crux); reversibility may match checkpointing at small K
  (the DLH null — the money plot must show a clear K-vs-VRAM frontier).
- **Fallback (no reversibility, reviewer-proof, tiny models):** (a) **adaptive/learned token-budget allocation** — make
  MilliVid's *fixed* `num_frames_per_level` content-driven (the paper says "decide what to fill S with" but never makes
  it adaptive); (b) a **controlled science-of-hierarchy study** — derive *when* the learned pyramid beats cascaded/flat
  (off-screen memory + coarse-resolvable structure), explaining the paper's own untested Q3 surprise. Converts your
  prior nulls into the central finding.
