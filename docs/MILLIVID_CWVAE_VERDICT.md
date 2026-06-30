---

# Is combining MilliVid (spatial hierarchy) + CW-VAE (clock temporal downsampler) possible and a good idea?

## 1. Verdict: CONDITIONAL — leaning NO on the headline

**Possible to build: yes. A good idea as a *novel-contribution* play: no, unless one specific empirical gate passes first.**

The honest one-line summary: the cleanest faithful version is **MilliVid's existing token-budget schedule with the temporal clock made explicit** — and that clock is **largely redundant with a mechanism MilliVid already has**. The part of the combination that would actually beat DLH is the **diffusion decoder + learned token pyramid** (the tokenizer/decoder), which your own repo already identified as the real lever (`dlh-blur-is-decoder-bottleneck`, `step1-adversarial-renderer-works`). The clock — the thing CW-VAE contributes — is the part most likely to be a measured null (`hierarchy-is-null-on-mmnist`, `cwvae-entropy-does-not-track-error`).

So: buildable as a small-scale science artifact; not honestly sellable as a new architecture without first proving the clock earns its place.

## 2. The combined architecture (if you build it)

**Two-stage, two variants.** Reuse map and scale to 16GB:

**Stage 1 — frozen spatial codec (reuse MilliVid wholesale):**
- MilliVid transformer AE: per-frame, no time (`frames.unbind(dim=1)`), produces a spatial token pyramid `[C, H/2^l, W/2^l]`, each level independently decodable by the DiT.
- Scale: ViT/DiT-B shrunk to ~384–512-d, depth 8, **128×128** input → 3 levels (S0 8×8=64, S1 4×4=16, S2 2×2=4 tokens), `latent_channels≈16–32`, bf16 + grad-checkpoint.
- Train once, freeze, dump `.pyramid` files to disk (`format_pyramid.py`, `4**level` stride). All temporal training then reads cheap disk latents.

**Stage 2 — temporal model, pick ONE variant:**

- **Variant A (recommended first, cheap):** keep MilliVid's stateless DiT + v-prediction diffusion. Add the clock **only as a `Selection`-matrix constraint**: a level-`l` token is denoised once every `factor^l` frames, then held as clean context and broadcast to faster levels. Reuses `scatter_by_index_` and `posemb_flhw(frame,level,h,w)` verbatim — no new indexing. Single-step training, no BPTT. This is a config/strategy change, not a rewrite.
- **Variant B (only one with a real complementary story, but a from-scratch retrain):** replace held-context with a **spatially-structured recurrent RSSM cell per level** (ConvGRU on the `[C, H/2^l, W/2^l]` grid — CW-VAE's flat GRU must be rewritten, so only the ~15-line clock schedule survives) + MilliVid DiT decoder + BPTT over 16–32 frames. Justified solely by constant-memory history beyond MilliVid's attention window.

Clock config: `factor=2`, levels matched to spatial levels (3). CW-VAE `factor=6` is too aggressive for a 3-level pyramid.

## 3. How it differs from DLH and RHCP

- **vs DLH:** DLH and CW-VAE are **flat-vector, temporal-only** models — each "level" is a D-dim vector with no spatial structure (verified: `cwvae_torch.py:75` collapses each frame to one 1024-d conv vector; `hvp_dlh/model.py:46-48` levels are feature MLPs). The combined model adds a genuine **second axis** — a spatial token pyramid (`adaptivity.py:37` `num_tokens_at_level = base//4**level`). So it is **not literally DLH**. BUT: the 2D (space×time) factorization is **MSPred (BMVC 2022)** — that, not DLH, is the work this most resembles and must cite/beat. And the part that beats DLH is the better tokenizer/decoder, not the clock.
- **vs your built RHCP/RHVP:** RHVP is a reversible *temporal* latent core (still single-axis, reversibility-as-enabler). This proposal sets reversibility aside and adds the **spatial** axis MilliVid provides, which RHVP lacks. Different axis, different lever.

## 4. The single defensible contribution

Not "spatial + temporal hierarchy" (that's MSPred). The only non-redundant, non-decoder sliver: **replace MilliVid's hand-designed rollout schedule with a learned, content-adaptive clock baked into the tokenizer** (a learned *when*-to-refresh on top of the learned *where*-to-allocate). No prior work combines learned 2D-factored tokens + adaptive clock + diffusion rollout. **Caveat:** "learned adaptive clock" is exactly DLH on the temporal axis, and your repo found that clock to be a null. Keep it **fixed** to stay distinct from DLH; if you learn it, you must own the DLH comparison head-on.

## 5. Cheapest first experiment (go/no-go gate)

On a **long looping dataset** (not MMNIST — your memory says MMNIST is a bad dynamics testbed and is background-dominated):

> **With the decoder held fixed, does a fixed-clock variant (coarse tokens held `factor^l` frames) beat plain MilliVid rollout at equal token budget on FVD / long-horizon consistency?**

- **No →** the clock is the established null; contribution collapses to "small-scale MilliVid reproduction." Stop.
- **Yes →** real result; frame it at the mechanism level (§4).

This is cheap: Variant A on frozen Stage-1 disk latents is a strategy-mask change. **Ablation discipline is the whole game** — the DiT-vs-L2 decoder upgrade is a massive confound; any gain must be attributed to the clock, not the decoder.

## 6. Honest risks

1. **Redundancy (highest):** MilliVid's `num_frames_per_level=[3,12,48,192]` IS a factor-4 clock on the same index that sets spatial resolution (`strategy.py:123` and `adaptivity.py:37` share the `4**level` divisor). "Coarse-spatial = slow-temporal" is already one index in MilliVid. Variant A's clock re-implements it with uniform hardcoded periods — strictly *less* flexible than MilliVid's attention-conditioned schedule.
2. **"Combination preserving both codes" is misleading:** the faithful spatial version forces a full rewrite of CW-VAE's flat GRU into a spatial cell. Honest label: "MilliVid + a newly written spatial RSSM," not "MilliVid + CW-VAE."
3. **Compute/quality gap:** MilliVid's recipe is ~576 H200-GPU-hours on LoopCraft (200k × 1024-frame 256×256 videos, not in hand). Matching it on one 4060 Ti is ~3 orders of magnitude off. The only quality bar reachable on this hardware is CW-VAE/MMNIST — which is near-trivial (all-black beats CW-VAE on aggregate SSIM) and validates the decoder, not the architecture. **Fits in 16GB ≠ reaches meaningful quality.**
4. **Capacity of the coarse level:** S2 at 2×2×32 (smaller scaled down) may be too thin to carry long-horizon scene memory at all, making the slow-clock state irrelevant. The diagonal coupling is a hypothesis to ablate, not a given.

## Should you build DLH-with-MilliVid-tokenizer instead?

If the goal is a *result that works* rather than a *novel architecture*: **yes, that's the higher-EV build** — it directly applies your proven lever (diffusion decoder fixes the L2 blur) to a stronger learned-token codec, and you can honestly report it as "the decoder is the lever, demonstrated at MSPred-family scale." Just don't dress it up as the clock contribution; the clock must pass §5's gate to earn a sentence.

Key files: `millivid/source/strategy/strategy_millivid.py` (the budget clock the CW-VAE clock competes with), `millivid/source/components/adaptivity.py:37,133` (`4**level` divisor + flhw index), `millivid/source/model/model_video_universal.py` (`generate()` rollout seam), `cwvae/cwvae_torch.py:75,167-169` (flat conv + sum-merge clock + tile-by-factor), `hvp_dlh/model.py:46-48` (DLH = temporal-only flat clock, the reinvention risk).