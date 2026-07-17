# CW-VAE same-population all-black diagnostic

The archived PyTorch CW-VAE run used 100 clips, 36 context frames, a 964-frame open-loop horizon and best-of-5 selection by clip-average SSIM.

| Horizon window | CW-VAE SSIM | Black SSIM | CW-VAE PSNR | Black PSNR |
|---|---:|---:|---:|---:|
| +1–48 | 0.6338 | 0.7679 | 13.944 | 13.943 |
| +49–192 | 0.5665 | 0.7679 | 12.625 | 13.942 |
| +193–964 | 0.5677 | 0.7682 | 12.909 | 13.941 |

## Interpretation boundary

This is a metric-sanity diagnostic, not a model-ranking experiment. It uses the same clips, target frames and `skimage` SSIM semantics as the archived evaluator, but the CW-VAE score is best-of-stochastic-samples whereas black is deterministic. Architectures, training budgets and inference budgets are not matched.

Only the CW-VAE mean curves survived, so paired record- or clip-level confidence intervals cannot be reconstructed. High black-frame scores on sparse Moving-MNIST mainly expose background dominance in full-frame SSIM/PSNR; they do not imply useful motion or object prediction.

## Provenance

- dataset SHA-256: `dff1f471b4ca4324975089d9217ae2c3382f12ac24d3985be5f82bb95e8ed141`
- archived curve SHA-256: `7d6c8cfd6f3e19a6c70f72b8ec2e0ab985b2a93d21a90258acea43e4cfb782a9`
- archived evaluator SHA-256: `6b4bc5dfc64bddded69aebcb3826503c19345eab569a67b66cdd3c73ffb13535`
