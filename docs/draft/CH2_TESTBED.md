# Chapter 2 — A memory-horizon testbed and evaluation protocol

> **Detailed fourth-article source chapter; not the binding submission narrative.** Assemble the
> focused paper through the binding reach–fidelity specification and claim ledger. Geometric
> observability replaces claims of information-theoretic necessity, and complete
> false-positive-sensitive metrics are required.

## 2.1 Motivation: why a purpose-built testbed

The central question of this work — how a bounded context-token budget should be packaged to cover
task-relevant memory across temporal rate and spatial resolution — cannot be tested on the standard
synthetic video-prediction suites. On
Moving-MNIST-class data, nothing ever leaves the frame and returns: every future frame is either
fully determined by a short context window or permanently unknowable, so there is no
*hard-but-computable-with-memory* regime for a long-memory mechanism to win in. Our earlier
experiments on such data produced clean nulls for every dynamics-side mechanism we tried, for
exactly this reason. What is needed is a testbed in which (i) content genuinely exits the viewport
and must later re-enter at a position the model can only know by remembering it, (ii) the ground
truth of "what was knowable from this context" is *exactly computable*, so that hardness is an
analytic label rather than a heuristic, and (iii) the whole benchmark is tractable on a single
16 GB consumer GPU.

Existing memory-oriented benchmarks satisfy (i) but not (ii) and (iii). Memory Maze [MemMaze22] and
the TECO Minecraft benchmark [TECO23] pose genuine long-horizon memory problems, but their re-entry
events arise from 3-D egomotion in a rendered environment: identifying *which* pixels of a target
frame were geometrically observable from a given context requires geometry processing at
best and is ill-defined at worst, and the associated model scales are beyond a single-GPU budget.
Recent evaluation protocols for spatial memory in world models — long-term spatial-memory
benchmarks for video world models [LSSVWM25, arXiv 2505.20171], memory-augmented world-model
evaluation [WorldPack25, arXiv 2512.02473], and geometry-grounded long-video consistency probes
[VidMem25, arXiv 2506.05284] — are the adjacent line of work: they too stratify evaluation by
whether content re-enters the field of view. Our testbed can be read as a minimal, fully analytic
member of this family: hardness labels are computed *exactly* from the recorded action track, and
every experiment in this article trains overnight on one RTX 4060 Ti.

## 2.2 The momentum looping world

The generator (`scripts/gen_looping.py`) constructs, per clip, a static *world*: a 192×64 texture
on a horizontal torus, composed of a smooth background (a sum of four sinusoids with integer
wavenumbers, so the texture wraps seamlessly; intensities in [0.15, 0.5]) and three MNIST digits
max-composited at fixed world positions. A 64×64 viewport pans horizontally over this world; the
observed clip is the sequence of viewport crops. With a 192-column world and a 64-column viewport,
two thirds of the world (128 columns, 67%) is off-screen at any instant.

The pan is a *momentum action walk*. At each frame the camera holds with probability
p_hold, otherwise it continues its current heading with probability p_cont and reverses
otherwise; the executed action a_t ∈ {−1, 0, +1} is recorded, and the viewport moves by
4 pixels × a_t (all parameters verified against the shipped data; empirically p_hold ≈ 0.10 and
p_cont ≈ 0.90 from the recorded action tracks). The momentum is essential. A symmetric random walk
diffuses too slowly to circumnavigate the torus within a clip, so off-screen content is either
still inside a short context window (determined) or never seen again (unknowable) — recreating
exactly the degenerate regime of Moving MNIST. Directed motion cycles the torus with a loop period
of approximately 192/4 = 48 frames, so a sprite that exits the viewport re-enters roughly 48 frames
later *at a world position fixed since frame 0*. A model with a long enough memory can predict the
re-entry exactly; a short-context model cannot. The sequence is also exactly periodic under a full
loop, giving a clean long-horizon consistency probe.

Because the viewport trajectory is a deterministic function of the recorded actions
(x_t = (x_{t−1} + 4·a_t) mod 192), the set of world columns covered by any context window is
computable in closed form. This is what makes the testbed's hardness labels exact rather than
approximate (§2.5), and it is why all models in this article are action-conditioned: withholding
the action forces every mechanism to hedge over three pan directions and manufactures a spurious
blur verdict, a failure mode we document in Chapter 3.

[FIG ch2-1: one world strip with the panning viewport marked, plus a row of consecutive observed
frames showing a digit exiting and re-entering. Caption: the momentum looping testbed — a 64 px
viewport pans a 192 px toroidal world; ~67% of the world is off-screen at any time and re-enters
with a ~48-frame loop period.]

## 2.3 Datasets

Two fixed datasets are generated once and shared by every experiment: `loopmom_train_3000x100`
(3,000 clips × 100 frames) and `loopmom_test_500x200` (500 clips × 200 frames), each stored as
uint8 video with the aligned action track. The training file is split by clip into 2,700 training
and 300 validation clips; the test file provides a held-out pool of 250 evaluation clips whose
longer 200-frame duration supports multi-loop rollout evaluation in Chapter 4. Test-set latents
are always standardised with the *training* set's statistics (§2.4).

## 2.4 Latent interface: a frozen two-level autoencoder

All predictive models operate in the latent space of a single frozen, two-level convolutional
autoencoder (1.85 M parameters), trained once on the training clips and never fine-tuned. A shared
encoder taps a *coarse* latent (16 channels at 4×4) and a *fine* latent (16 channels at 8×8), each
independently decodable to the full 64×64 frame — the per-level-decodability property of
hierarchical latent video models in the MilliVid mould [MilliVid26], in minimal form. The coarse
level is a compact layout/pan code that makes long spatially compressed history affordable; the
fine level retains higher-resolution detail. Whether that spatial package is preferable to a
temporally subsampled full-resolution package is an experimental question, not an architectural
premise. Final validation reconstruction error is 0.0020 (MSE, fine level).

Both latents pass through a tanh and are therefore small in scale (per-channel standard deviations
average 0.121 coarse and 0.162 fine). The evaluation and training harness z-scores each latent
channel to unit variance, per level, using training-set statistics only. This *latent
standardisation* is part of the protocol, not a detail: Chapter 3 shows that omitting it silently
mis-scales a diffusion noise schedule and materially changes mechanism-level conclusions.

Downstream predictors consume these latents as an arbitrary set of token *streams*, each
parameterized by spatial resolution, temporal stride, and frame count, joined in one attention
sequence — so hierarchy **depth is backbone-capacity neutral**: the attention backbone is
unchanged, and a third level adds only per-level input/output parameters (+0.30 M on a 22.19 M
predictor, +1.4%) plus a frozen autoencoder tap (1.25 M) — measured counts, disclosed rather
than rounded to zero. This matters for the shape
comparisons of §4.4: unlike recurrent hierarchies (where each level adds a full recurrent cell,
and published depth comparisons are capacity-confounded), our depth axis holds the backbone
fixed by construction.
For the three-stage arm the autoencoder gains a third, 2×2 latent tap trained on the *frozen*
trunk, so the two levels above remain bit-identical and every earlier measurement stays valid.

## 2.5 Metrics: foreground-masked, per-image, decoded

Aggregate pixel metrics are unreliable on sparse video (§2.7), so the protocol measures only
decoded pixels under a foreground mask, and always at the per-image level.

**Foreground MSE (fg-MSE).** Predictions and ground truth are decoded to [0, 1] pixels; error is
averaged over pixels where the ground-truth frame exceeds a threshold of 0.5. The threshold is
testbed-specific and verified: the background texture occupies ≈0.18–0.41 while sprite strokes
reach ≈0.99, so thr = 0.5 isolates the digits. (A conventional threshold of 0.1 would classify the
entire panning background as foreground.)

**Foreground sharpness (fg-sharp).** Per-image Laplacian energy on that image's foreground mask:
sum(∇²x)² weighted by the mask, normalised by mask area, then averaged over images. This is
deliberately *not* a batch-variance statistic — batch variance rewards inter-sample diversity,
whereas per-image Laplacian energy rewards per-frame detail, which is what distinguishes a
committed prediction from a mean-hedged blur.

**Re-entry stratification.** For each evaluation sample, the world columns covered by the context
window are reconstructed exactly from the action track, and the *novel-foreground fraction* is the
fraction of the target frame's sprite columns never covered by that context. Stratification is
always computed against a fixed 6-frame reference window ("novel-fg-at-6"), so that all arms —
whatever their actual context length — are binned on the same hardness axis. Three bins are used:
**easy** (novel-fg = 0: the sprite was fully visible in the short window), **mid** (0 < novel-fg
≤ 0.1) and **hard re-entry** (novel-fg > 0.1: a substantial part of the sprite was not
geometrically visible in the short window but was visible to the compared longer context). Shared
digit priors can still support statistical guesses, so this is an observability label rather than
a necessity theorem. The
single-step mechanism study of Chapter 3 uses the analogous *novel-strip* label, restricted to the
newly revealed leading strip of the pan — the only genuinely generative region of a single
predicted frame.

**Decoded-sample review.** Every evaluation saves panels of decoded frames (last context frame |
prediction | ground truth) drawn from the *same* samples that produced the scores. We treat this
as a first-class protocol element: on two occasions during this project, scalar metrics concealed
a qualitative failure (a slow fade to the mean, and a blank-output collapse) that one glance at
the panels exposed.

## 2.6 Statistical protocol

All arm-versus-arm comparisons use identical (clip, target, horizon) records. Stochastic samplers
also use common random numbers when the compared target tensors have the same shape. Cross-shape
comparisons such as B1/E-320 retain exact record pairing but cannot share elementwise sampler
noise; this does not bias the delta, but it gives less variance reduction. Headline deltas are reported
with 95% confidence intervals from a paired bootstrap over per-sample deltas (10,000 resamples,
percentile intervals), computed within each hardness bin; evaluation pools are sized at ≈768
samples so that the hard re-entry bin retains adequate power. Where a best-of-N statistic is
reported, N samples are drawn per input and the candidate with minimum foreground MSE against the
ground truth is selected *in pixel space*; this is an oracle coverage measure (does the mechanism
ever place probability mass correctly?) and is always reported alongside the mean over all N
samples (precision) and, in later chapters, best-of-1.

## 2.7 Reference floors and ceilings

Every table in this article is bracketed by the same references, measured under the identical
pipeline. The **persistence baseline** (copy the last context frame's fine latent, decode) scores
fg-MSE 0.2367 and fg-sharp 0.0515 at the 4-frame prediction offset used in Chapter 3 — any model
below that error has learnt at least the action-conditioned warp. The **sharpness ceiling is set
by the frozen decoder, not by the data**: raw ground-truth frames measure fg-sharp 0.4785, but the
autoencoder round-trip of those same frames measures 0.2412. All model sharpness values must
therefore be read against ≈0.24, not ≈0.48; roughly half the raw sharpness is spent in the
autoencoder before any prediction model acts. We state this ceiling explicitly wherever sharpness
is reported, and we do not claim sharpness beyond it. Conversely a decoded output with fg-sharp
below ≈0.08 contains no digit structure at all (an empirically calibrated collapse threshold used
throughout Chapter 3).

Table 2.1 summarises the testbed and protocol.

**Table 2.1 — Testbed and evaluation protocol summary.**

| Component | Value | Notes |
|---|---|---|
| World / viewport | 192×64 torus / 64×64 crop | 128 cols (67%) off-screen at any time |
| Content | background texture + 3 MNIST digits | fixed world positions; max-composited |
| Pan process | momentum walk, speed 4 px/frame | p_hold ≈ 0.10, p_cont ≈ 0.90 (measured); actions {−1, 0, +1} recorded |
| Loop period | ≈ 48 frames | 192/4; sets the memory horizon |
| Train / test | 3,000×100 / 500×200 frames | clip-disjoint; test standardised with train stats |
| Latent interface | frozen 2-level conv-AE, 1.85 M | coarse 16 ch @ 4×4, fine 16 ch @ 8×8, per-level decodable |
| Standardisation | per-channel z-score, per level | train-set statistics only |
| Error metric | fg-MSE, threshold 0.5 | decoded pixels; bg ≈ 0.18–0.41, sprite ≈ 0.99 |
| Sharpness metric | per-image masked Laplacian energy | GT 0.4785; AE round-trip ceiling 0.2412; collapse < ≈0.08 |
| Hardness axis | novel-fg vs 6-frame window (exact, from actions) | easy = 0; mid (0, 0.1]; hard > 0.1 |
| Statistics | paired bootstrap 95% CIs; CRN where target shapes match | 10,000 resamples; ≈768-sample eval pools |
| Baselines | persistence fg-MSE 0.2367, fg-sharp 0.0515 | k = 4 offset |

## 2.8 Why aggregate pixel benchmarks are excluded

We report no aggregate SSIM, PSNR or full-frame MSE in the main chapters. On sparse-foreground
video these metrics are dominated by the background: in this project's own measurements on
Moving-MNIST-class data (≈93% black background), a constant all-black prediction outperformed a
trained CW-VAE [CWVAE21] on aggregate SSIM and PSNR, and the real differences between models were
visible only on foreground pixels. A benchmark on which the empty image wins is not evidence about
dynamics. The looping testbed's panning background makes aggregate metrics less degenerate than on
Moving MNIST but still foreground-diluted, so the main text uses only the masked metrics of §2.5.
For continuity with the literature, an appendix reports our bridge to the OpenSTL benchmark suite
using bit-verified reimplementations of its metric definitions; those numbers are provided for
calibration, not as claims. One further exception of the same kind: for the standard-Moving-MNIST
long-horizon control (§5.3.1) we *display* an SSIM-versus-horizon curve for comparability with
the CW-VAE-era literature, always with the persistence and all-black floors drawn on the same
axes — it is calibration, not evidence, for exactly the reason this section establishes.

## 2.9 Limitations

The testbed is deliberately minimal, and its scope should be stated plainly. Motion is
one-dimensional and piecewise-deterministic given the action; the only stochasticity a predictor
faces is the action-marginal future and the identity/position of off-screen content, so the
generative burden is concentrated in the re-entry events by design. World content is static —
nothing moves in world coordinates — so the testbed cannot probe memory of *dynamic* off-screen
state. All conclusions are at 64×64 grayscale and single-digit-scale complexity; the analytic
hardness labels that justify the design would not transfer unchanged to 3-D egomotion benchmarks,
which we position as the natural next rung. Finally, the sharpness ceiling of the frozen
autoencoder (0.2412 vs 0.4785 raw) caps every downstream sharpness comparison; a stronger decoder
would raise the ceiling but was not needed to separate the mechanisms under study.

---

*Draft-only provenance notes (remove before submission).* World/viewport/off-screen/loop-period:
`logs_rhcp/gen_mom_train.log`, `logs_rhcp/gen_mom_test.log`, `scripts/gen_looping.py`. p_hold/p_cont/digit
count: measured from `data/loopmom_train_3000x100(_actions).npy` (scratchpad script, 2026-07-02; digit
components 2–3 per reconstructed world, consistent with 3 placements with occasional overlap). AE:
`hvp_rev/hier_convae.py`, `logs_rhcp/hier_ae_mom.log` (1.85 M, fine val recon 0.0020). Latent stds
0.1213/0.1618, references (raw sharp 0.4785 / AE 0.2412), persistence (0.2367/0.0515):
`logs_rhcp/step1_mech_clean.log`. Thresholds/metric definitions: `hvp_rev/step1_mech.py` docstrings
([RT-fgthr], [RT-sharp], [RT-pxbest]). Bins/bootstrap/CRN/768: `hvp_rev/mem_gate.py`. All-black-beats-CW-VAE
on aggregate SSIM: project memory `mmnist-ssim-is-background-dominated` (needs a citable table in the
appendix before submission). OpenSTL bridge: `hvp_rev/eval_adapters.py`.
