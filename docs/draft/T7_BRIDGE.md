# T7 — OpenSTL Moving-MNIST bridge (10→10, canonical test set, n=10000) — MEASURED 2026-07-06

Harness: `hvp_rev/mmnist_bridge.py` (bit-verified OpenSTL metric adapters; faithful generator
port, GEN_SEED=0; fresh AE `hier_ae_mmnist.pt`; unmodified WindowDiT 22M, B16×15k bf16, best-of-1,
3 AR window steps 4+4+2; log `logs_rhcp/mmnist_bridge_chain.log`).

| row | MSE (pixel-sum) | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| mm-flat (10 fine, 640 tok) | 180.91 | 229.94 | 0.7625 | 13.55 | 0.1320 |
| mm-hier (10 coarse + 6 fine, 544 tok) | 181.29 | 230.45 | 0.7607 | 13.54 | 0.1389 |
| AE round-trip ceiling | 3.30 | 20.79 | 0.9917 | 30.93 | 0.0062 |
| persistence | 286.28 | 337.89 | 0.6528 | 11.56 | 0.2533 |

Leaderboard reference (OpenSTL local docs, 200-epoch specialists): ConvLSTM-S 29.80, PredRNN
23.97, SimVP+gSTA-S 26.69 MSE.

## Honest framing (appendix text spine)

1. **Protocol anchor achieved:** the harness reproduces the standard benchmark protocol exactly
   (canonical test sequences, bit-verified metric conventions), and the AE ceiling row (3.30)
   shows the tokenizer is not the limiter.
2. **Absolute numbers are modest, with disclosed reasons:** our arm is a 22M stochastic latent-
   diffusion generalist at ~15k steps evaluated best-of-1; the leaderboard entries are
   deterministic pixel-regressors trained ~10× longer and optimized exactly for pixel-sum MSE,
   which structurally punishes sampling variance. The model halves persistence's LPIPS
   (0.132 vs 0.253) and beats it on every metric.
3. **The flat ≈ hier null (Δ 0.4 MSE ≈ 0) is the PREDICTED boundary condition, not a failure:**
   Moving-MNIST has no off-screen re-entry — nothing ever leaves the frame — so there is no
   memory horizon for a coarse level to carry. The bridge therefore closes the article's
   argument ON the standard benchmark: where the memory horizon is absent, hierarchy is a null
   (consistent with the project's founding MMNIST observations); where the horizon exists
   (testbed A), the matched-budget hierarchy win is two-seed CI-clean. "When does hierarchy
   pay" gets its negative control from the field's own benchmark.

Fills: front-matter [AUTHOR: OpenSTL bridge] marker → measured; Розділ 4 §4.1 [PENDING — T7].
