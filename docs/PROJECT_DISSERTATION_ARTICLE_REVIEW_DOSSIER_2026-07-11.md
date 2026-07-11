# Project, dissertation, and article review dossier

**Snapshot:** 2026-07-11, during the article-critical post-Phase-B experiment queue  
**Project root:** `C:\Users\leoni\Documents\projects\hierarchical-video-prediction`  
**Candidate:** Kulyk Leonid Ruslanovych  
**Specialty:** 124 — System Analysis  
**Supervisor:** Prof. O. B. Mokin, VNTU  
**Purpose of this document:** a cold-start, self-contained handoff for an independent reviewer.

This dossier records the project's scientific evolution, measured results, code and protocol,
current interpretation, dissertation/article organization, unresolved experiments, editorial
work, risks, and the exact files needed to verify it. It deliberately separates four categories:

1. **MEASURED** — results backed by local logs/NPZ/checkpoints.
2. **INTERPRETATION** — the current explanation of those measurements.
3. **PENDING** — pre-registered experiments already in the execution queue.
4. **FUTURE** — valuable work that is not required to close the current article/dissertation.

The project has changed substantially as evidence accumulated. Older documents may describe the
earlier coarse-spatial hierarchy thesis; the current scientific authority is the context-packaging
account in this dossier, `docs/CLAIM_LEDGER.md`, and the chronological evidence trail in
`docs/SPATIAL_FACTORIAL.md`.

---

## 1. Executive summary

The project began as an attempt to build and defend a hierarchical long-memory video-generation
system, initially with a strong interest in energy-based generation, reversible optimization, and
a coarse latent level that would carry long-range state. The work produced several publishable
technical strands: entropy-based adaptive decomposition, multidimensional positional encoding,
reversible long-horizon optimization, a controlled comparison of regression/diffusion/energy
heads, a hierarchical latent video predictor, and a purpose-built memory benchmark.

The experiments did not support a universal claim that a coarse spatial hierarchy is intrinsically
better. They supported something more precise and, in the present assessment, scientifically
stronger:

> **Long-horizon prediction under a bounded context-token budget is a context-packaging problem.**
> Performance depends on whether the budget covers the task-relevant memory span and on how it is
> allocated across temporal rate, spatial resolution, recent detail, depth, representation, and
> prediction head.

The central measured picture is:

- A contiguous flat window improves steadily as tokens buy more full-resolution frames.
- A spatially coarse/fine package quickly reaches the required memory span, then plateaus because
  extra tokens mostly add redundant recent detail rather than more useful span.
- This creates a bounded budget region where compressed long history beats a contiguous flat
  window, followed by an inversion once the flat window can afford the same span at full resolution.
- The lower edge depends on the packaging family: temporally striding the coarse stream makes the
  loop-covering package affordable at 512 rather than 896 tokens.
- A still more direct control — temporally subsampled **full-resolution** history — beats the
  spatially coarse package under the diffusion head on both measured seeds.
- Under regression, the temporal-versus-spatial ordering flips across seeds and is tied in the
  two-seed mean. Thus span coverage is robust, while the best representation of that span is
  head- and optimization-basin-sensitive.
- Adding a third spatial level at fixed budget hurts in the measured single-scale testbed, but the
  experiment confounds depth, loss allocation, optimization load, and a low-information 2×2 top
  stream. It is correctly reported as a **depth-plus-objective-allocation probe**, not “depth
  hurts.”
- Blinding confirms that the coarse stream contains retrieval-relevant information, even though
  coarse spatial packaging is not the best quality configuration in every condition.
- On a much longer loop, single-step retrieval can be learned, but its advantage collapses during
  self-fed rollout after approximately 24 frames. Memory span is insufficient if the remembered
  contents are no longer trustworthy.

The article's most defensible identity is therefore a rigorous causal measurement study of
long-span context packaging, not a universal architecture paper. The dissertation retains the
broader systems-analysis programme: consistency criteria, adaptive decomposition, positional
coherence, reversible resource optimization, hierarchical/multi-stream generation, measurement
instrumentation, and empirical decision rules.

---

## 2. How the project evolved

### 2.1 Initial dissertation direction

The early dissertation was organized around “systems analysis and information technologies for
consistent iterative video generation,” with a proposed hierarchical energy-based video model
(HVEBT). Its intended conceptual chain was approximately:

1. Video generation is a complex multi-level system with conceptual, hierarchical, and trajectory
   consistency requirements.
2. Entropy and adaptive decomposition identify where representation resources should be spent.
3. RoPE-ND aligns positions across spatial/abstraction levels.
4. Reversible computation makes long iterative processes trainable within limited memory.
5. A hierarchical energy model supplies global consistency and long-range generation.

The early energy-centric Chapter 3 could not be defended by the eventual measurements. Rather
than hiding that outcome, the dissertation was restructured. Energy-based descent remains a
carefully characterized negative result; it is no longer the system's generation core.

### 2.2 The actual implemented predictor

The implemented article system became a continuous-latent, multi-stream transformer over a frozen
hierarchical convolutional autoencoder. A shared WindowDiT backbone consumes arbitrary context
streams defined by:

`stream = (spatial resolution, temporal stride, number of frames)`.

Generation is performed using either:

- latent diffusion / iterative refinement, or
- direct latent regression.

The core capacity is approximately 22.19M parameters. Changing from flat to two-level context is
backbone-capacity neutral; a third level adds only about 0.30M parameters (~1.4%) for additional
level-specific interfaces.

### 2.3 The original hierarchy claim

The first article-level hypothesis was that a cheap coarse latent stream could retain a much
longer temporal history than a full-resolution flat window at the same context-token budget. The
coarse stream costs 16 tokens/frame; the fine stream costs 64 tokens/frame. A 48-frame coarse
history therefore costs only 768 tokens, while 48 full-resolution fine frames cost 3,072 tokens.

The first matched-budget results supported the coarse package at 1,152 tokens. This initially led
to the story “hierarchy pays when memory is required.” Subsequent frontier experiments showed that
the claim had two boundaries: flat wins when neither package covers useful memory, and flat wins
again when it can afford the whole horizon at full resolution.

### 2.4 The CW-VAE-inspired expansion

The author then asked why the hierarchical curve did not improve monotonically with more tokens,
unlike the flat curve, and proposed a much larger factorial inspired by Clockwork VAE.

Reference material supplied for comparison:

- CW-VAE article: `C:\Users\leoni\Downloads\arXiv-2102.09532v2`
- CW-VAE code: `C:\Users\leoni\Documents\projects\cwvae`

The original proposed factorial crossed temporal downsampling, spatial downsampling, depth,
prediction head, and token budget. Review showed that many cells were degenerate, impossible, or
scientifically predetermined. The programme was reduced to claim-bearing controls:

- lower/upper budget edges;
- temporal stride at fixed spatial representation;
- spatial abstraction at fixed window;
- temporal full-resolution versus spatially coarse packaging;
- depth at fixed budget;
- prediction-head twins;
- stream blinding;
- seed and optimization-budget robustness;
- a long-horizon external boundary test.

### 2.5 The decisive conceptual shift

The temporally subsampled full-resolution control (`strided-flat`) beat the coarse spatial
hierarchy. This result made the old story untenable but made a stronger story available.

Current terminology:

- **Context packaging** is the load-bearing scientific object.
- **Spatial packaging** means long coarse-resolution history plus recent fine detail.
- **Temporal packaging** means sparse full-resolution anchors plus recent dense detail.
- **Depth** means additional nested streams/levels.
- **Flat** is a one-stream special case, not the only alternative to hierarchy.
- “Hierarchy” is retained as a taxonomy of multi-rate and/or multi-resolution organization, but
  it is not allowed to make the thesis true by definition.

The falsifiable content is in span thresholds, saturation/inversion, family-specific costs,
head/seed interactions, and the trustworthy-memory boundary.

---

## 3. Dissertation-wide research programme

The dissertation is broader than the final article. Its intended systems-analysis contribution is
the integration of several methods and their applicability boundaries.

### 3.1 Consistency criteria

The dissertation distinguishes:

- **Conceptual consistency:** ability to generate coherent unseen combinations.
- **Hierarchical consistency:** alignment between abstraction/representation levels.
- **Trajectory consistency:** stability and recoverability across autoregressive rollout.

Trajectory consistency is operationalized using decoded rollout metrics stratified by an exactly
computable re-entry-hardness label. This is important for specialty-124 positioning: the work is
not only a neural architecture; it is a criteria-based system analysis and decision procedure.

### 3.2 Adaptive decomposition / V-BLT strand

The V-BLT work uses byte/pixel entropy to identify high-information regions and motivate adaptive
allocation. It is the first-stage hierarchical/adaptive decomposition strand of the publication
arc. It contributes uncertainty estimation and resource allocation rather than the article's
long-memory result.

### 3.3 RoPE-ND strand

RoPE-ND provides a mathematical instrument for multidimensional positional coherence and, where
appropriate, coordinate alignment between abstraction levels. A prior draft incorrectly called it
the implementation basis of the Chapter 3 predictor. That contradiction has been corrected: the
measured predictor uses learned level/role/frame embeddings in joint attention. RoPE-ND and the
article predictor are complementary instruments, not the same implementation.

### 3.4 Reversibility / TempVerseFormer strand

The reversible method targets activation-memory growth in long multi-step processes.

Measured/proven receipts include:

- deterministic reversible-coupling reconstruction error ~`5.5e-15` in fp64;
- store-the-noise stochastic reconstruction error ~`8.4e-13`;
- reverse-replay gradients matching BPTT (cosine similarity `1.000000`);
- constant activation-memory scaling with temporal horizon;
- a three-axis applicability taxonomy: reversibility across depth, time, and computation.

The honest boundary is as important as the positive result. Reversibility is useful against
vanilla BPTT on long temporal rollouts and approximately Pareto-comparable with checkpointing. It
does **not** materially solve the inner optimization-loop problem: the measured achievable saving
there is only about 0.26% of peak memory (analytic estimate ~0.6–1.0%). Reversibility is therefore
an enabling method for future long-horizon rollout training, not the cause of the current Chapter
4 prediction gains.

### 3.5 Multi-stream video-generation strand

The current model supports multiple context streams at different temporal rates and spatial
resolutions over a shared attention backbone. The coarse/fine autoencoder is the measured spatial
implementation. Strided full-resolution history is the measured temporal implementation. The
dissertation must defend the method family and measurement procedure, not claim that its first
spatial implementation is universally optimal.

### 3.6 Measurement-instrument strand

The looping memory-horizon benchmark and its evaluation harness are arguably the most original
joint contribution:

- exact hard/easy labels from action/world geometry;
- matched context-token budgets;
- explicit floors and decoder ceilings;
- decoded qualitative review;
- paired clip-cluster bootstrap intervals;
- pre-registered decision branches;
- seed-visible rather than best-seed reporting;
- negative controls and boundary conditions.

This instrument is what turns “the model looks better” into a systems-analysis result about which
configuration is accepted under which operating conditions.

---

## 4. Model and code architecture

### 4.1 Frozen latent representation

For 64×64 grayscale loopmom frames, the frozen hierarchical convolutional autoencoder produces:

- coarse latent: 16 channels at 4×4 = **16 tokens/frame**;
- fine latent: 16 channels at 8×8 = **64 tokens/frame**;
- optional top latent: 2×2 = **4 tokens/frame**.

The coarse and fine streams are independently decodable. Latents are standardized per channel
using training-set statistics only. Reusing an autoencoder on a differently scaled spatial world
was found to destroy absolute-position landmarks; the standing rule is a fresh autoencoder per
dataset/testbed, shared within that campaign.

### 4.2 Shared predictor

`hvp_rev/rhvd.py` contains the main experimental stack:

- arm registry and exact token accounting;
- `WindowDiT`, a shared-attention multi-stream transformer;
- `WindowDiffusion` and `WindowRegression` heads;
- stream extraction and temporal-stride phasing;
- autoregressive ring-buffer rollout;
- oracle GO gates;
- coarse/fine/top blinding;
- evaluation artifact writing;
- paired comparison and clip-cluster bootstrap tables.

Other important modules:

- `hvp_rev/hier_convae.py` — frozen two-level autoencoder;
- `hvp_rev/hier3.py` — frozen-trunk 2×2 top extension;
- `hvp_rev/faithful_nets.py` — tokenizers, DiT blocks, EBT networks;
- `hvp_rev/step1_mech.py` — data encoding/standardization and decoded metrics;
- `hvp_rev/phasec_eval.py` — Phase-C identity/trajectory/count evaluation;
- `hvp_rev/phasec_summary.py` — seed-visible equivalence summary.

### 4.3 Important implementation safeguards

- Marked-prediction canaries found no ground-truth leakage in open-loop rollouts.
- Target and context tokens are accounted frame-exactly.
- Old two-level checkpoints remain bit-compatible after adding optional top-stream support.
- Evaluation always shares clip/target/horizon records across arms.
- Common random numbers are used only when target tensor shapes match. Cross-shape comparisons
  remain record-paired but do not receive elementwise noise cancellation; this affects precision,
  not bias.
- Confidence intervals resample clips, not individual frames, to respect rollout correlation.
- Checkpoint names encode the target number of steps, not necessarily current progress.
- The current queue is lock-protected and single-owner.

### 4.4 Known code/artifact risks

- Evidence NPZ/checkpoints are intentionally untracked and occupy roughly 18 GB. They require an
  external backup plus checksum manifest before defense/submission.
- W&B runs are useful provenance but are not a substitute for local artifact backup.
- The working tree may show `docs/CWSTYLE_TABLE.md` and `docs/figures/fig12_cwstyle_curves.png`
  modified while the live queue regenerates them. These are expected generated outputs and must
  not be discarded.

---

## 5. Primary testbeds and evaluation protocol

### 5.1 Operating point A: loopmom

The primary testbed is a 192×64 toroidal world viewed through a 64×64 panning viewport.

- Approximately 67% of the world is off-screen at any instant.
- Directed motion is approximately 4 px/frame with momentum and ~10% hold probability.
- Nominal re-entry loop is approximately 48 frames.
- Train data: 3,000 clips × 100 frames.
- Test data: 500 clips × 200 frames.
- Evaluation uses a 48-frame ground-truth warmup followed by 152 autoregressive frames.
- The action track makes re-entry hardness analytically computable.

The benchmark is intentionally static in world coordinates. This makes it maximally favorable to
temporal subsampling because content does not evolve between anchors. Any claim comparing temporal
and spatial packaging must retain this qualifier.

### 5.2 Operating point B: hgrid3 / long loop

The second testbed widens the world to produce an effective loop of approximately 196–213 frames.
It tests whether memory that works when the loop is within the ground-truth warmup remains useful
when it must persist through a long self-fed rollout.

Several iterations were scientifically informative:

- A scaled-background version accidentally made positions exactly aliased and produced a loss.
- A fresh position-unique variant restored single-step retrieval.
- A span-224 arm produced a strong single-step threshold signature.
- The decisive matched pair failed during open-loop rollout: advantage at h1–24, null around
  h25–60, then small losses at later horizons.

The correct conclusion is not “long memory fails.” It is:

> A sufficient addressable span can learn retrieval, but memory contents degrade under self-fed
> rollout; trustworthy memory is a separate capability from nominal reach.

### 5.3 External/boundary datasets

- Standard Moving MNIST 10→10: measured boundary/null for the memory-horizon claim.
- TaxiBJ 4→4: measured real-data competence/economy anchor, not positive transfer of the
  hierarchy result.
- Long Moving MNIST Phase C: pending three-seed boundary test over ~300-frame rollouts with
  identity, trajectory, count, and fg-MSE equivalence criteria.
- 2D rotations, 3D polycubes, MMNIST-CIFAR, KTH: publication/dissertation receipt programme;
  mostly not part of the focused article's core evidence.

### 5.4 Metrics and statistics

Primary article metrics:

- decoded foreground-masked MSE (lower is better);
- decoded foreground sharpness / masked Laplacian energy;
- horizon × hardness stratification;
- `dt` since last visibility and answerability (`seen`) gates;
- persistence floor and autoencoder round-trip ceiling.

Headline comparisons use paired clip-cluster bootstrap confidence intervals, normally 10,000
resamples. Seed-level effects are shown individually; frames are never pooled across seeds as if
they were independent replications. Aggregate SSIM/PSNR on mostly black synthetic data are not
used as primary evidence because all-black output can score misleadingly well.

---

## 6. Experimental results: mechanism studies

### 6.1 Regression versus diffusion

Single-step / best-of-N coverage favors diffusion at matched capacity/NFE:

- diffusion ALL foreground error ~0.0512;
- regression ~0.0664;
- diffusion improves with additional NFE and produces sharper committed samples.

Deterministic best-of-1 rollout favors regression:

- `reg-long`: 0.1682 (seed 0), 0.1503 (seed 1), mean ~0.1592;
- `hier-long` diffusion: 0.1848 (seed 0), 0.1690 (seed 1), mean ~0.1769.

There is no universal “best head.” Diffusion is a coverage/perceptual mechanism; regression is
better for the deterministic pixel-accuracy rollout regime used by most benchmark rows.

### 6.2 Energy-based descent

Energy-based transformers were tested using both the project's implementation and a reference
repository control.

Two complementary failures were measured at reachable scale:

1. One recipe learns a mis-ordered landscape where a mean/collapsed prediction has lower energy
   than the true target.
2. A corrected recipe orders the target properly, but gradient-based inference still cannot reach
   structured outputs from the prescribed initialization; more inference computation does not
   recover the capability.

This is a characterized negative, not a universal refutation of EBTs. It tests a capability and
scale that the source work did not claim. EBT and reversibility are non-core for the focused
article and likely appendix/separate-paper material.

---

## 7. Experimental results: token-budget frontier

All values below are decoded ALL foreground MSE; lower is better. Seed means are shown where two
seeds exist. Positive paired delta means the long-span/spatial arm beats its listed flat partner.

| Context tokens | Flat | Spatial coarse/fine | Paired result | Interpretation |
|---:|---:|---:|---|---|
| ~512 | flat-short 0.2215 | hier-short 0.2258 | flat wins −0.0036 / −0.0051 | stride-1 coarse span is too short; coarse tokens are redundant |
| 896 | flat-min 0.2104 | hier-min 0.1725 | spatial wins +0.0399 / +0.0358 | measured lower edge of dense coarse family |
| 1,152 | flat-budget 0.2041 | hier-long 0.1769 | spatial wins +0.0189 / +0.0354 | original matched-budget headline |
| 1,792 | flat-c1 0.1876 | hier-c1 0.1799 | spatial wins +0.0077 [0.0016, 0.0142], seed 0 | approaches upper crossing |
| 2,432 | flat-c2 0.1703 | hier-c2 0.1820 | flat wins −0.0116 [−0.0162, −0.0073], seed 0 | 30k upper crossing bracket |
| 3,072 | flat-long 0.1572 | hier-rich 0.1691 | flat wins −0.0131 / −0.0107 | full-resolution saturation/information advantage |

At B16×30k, the spatial package wins a bounded band with lower edge between 512 and 896 and an
upper crossing between 1,792 and 2,432. However, the upper bracket is **optimization-budget
conditional**. At 1,152 tokens, increasing both arms to B8×60k changed flat only 0.0006 but
improved the spatial arm by 0.0182. A matched B8×60k c2 pair is therefore pending before any
asymptotic upper-edge language is allowed.

### 7.1 Why the spatial curve plateaus

The spatial arms at and above 896 tokens already contain the approximately 48-frame coarse span.
Additional tokens mostly buy more recent fine frames, which are highly redundant for this static
world. Their quality therefore oscillates around a family plateau rather than declining
monotonically with tokens. The apparent irregularity is further explained by:

- seed variance in memory-bearing arms;
- fixed-step under-convergence that worsens with sequence length;
- mixing B16×30k and B8×60k regimes in early figures.

The flat curve continues improving because each token increment extends full-resolution history.
This plateau is expected under the present testbed; it is not evidence of a broken hierarchy.

---

## 8. Experimental results: shapes of context packaging

### 8.1 B1 — stride-2 spatial package at 512 tokens

Configuration: `hier-s2-min`, 24 coarse frames at stride 2 plus 2 recent fine frames. Span ~48;
budget 512.

- seed 0: ALL 0.1893; +0.0330 `[0.0246, 0.0419]` versus flat-short;
- seed 1: ALL 0.1869; +0.0338 `[0.0251, 0.0430]`;
- two-seed mean: 0.1881;
- every horizon × hardness cell wins on both seeds.

This confirms that the lower edge is family-dependent: temporal stride allows the spatial family
to cover the loop at 512 tokens. The effect is extremely stable across seeds. One wording
refinement is required: seed 0 was null at very long `dt`, while seed 1 retained a small positive
effect. The robust statement is a benefit strongest near the memory-retrieval range that decays
with distance, not a universal exact cutoff.

The stride-2 family plateaus approximately 0.017 above the dense stride-1 family. Sparser anchors
therefore buy span more cheaply but impose an inter-anchor/family cost.

### 8.2 B2 — temporally subsampled full-resolution package

Configuration: `strided-flat`, 11 full-resolution history anchors at stride 4 plus 6 recent fine
frames, no coarse spatial stream. Span ~48; budget 1,088 (64 fewer tokens than hier-long).

- seed 0: 0.1652, sharpness 0.0919;
  - beats hier-long by +0.0197 `[0.0143, 0.0252]`;
  - beats flat-budget by +0.0386.
- seed 1: 0.1644, sharpness 0.0948;
  - beats hier-long by +0.0046 `[0.0007, 0.0084]`;
  - beats flat-budget by +0.0400 `[0.0296, 0.0508]`.
- two-seed mean: 0.1648.

The aggregate temporal-over-spatial ordering replicates under diffusion. The effect magnitude is
heterogeneous: seed 1's advantage over the coarse package is small and concentrated through the
short/mid rollout and `dt<=48`, while later bins are null. The large advantage over contiguous
flat context is stable throughout the rollout.

Correct current statement:

> Under diffusion on this static-world testbed, temporally subsampled full-resolution context is
> the best measured aggregate packaging on both seeds, but its margin over spatial coarsening is
> seed- and horizon-dependent.

This does not prove that temporal subsampling dominates on dynamic off-screen content. Static
world coordinates are maximally favorable to sparse anchors.

### 8.3 Regression-head twins and head × seed interaction

`strided-flat-reg` results:

- seed 0: 0.1574;
  - beats `reg-long` 0.1682 by +0.0107 `[0.0062, 0.0158]`;
  - beats `reg-flat` 0.2015 by +0.0441 `[0.0338, 0.0551]`;
  - also beats diffusion strided-flat by +0.0077 `[0.0013, 0.0147]`.
- seed 1: 0.1606;
  - **loses** to `reg-long` 0.1503 by −0.0104 `[−0.0146, −0.0062]`;
  - beats `reg-flat` 0.2004 by +0.0397 `[0.0293, 0.0509]`.
- two-seed means:
  - temporal regression ~0.1590;
  - spatial regression ~0.1592;

This is a genuine prediction-head × seed/optimization-basin interaction. Regression does not
yield a robust family winner: the signs flip nearly symmetrically and the means tie. Both
long-span packages decisively beat contiguous regression-flat context.

Correct current statement:

> Span coverage is robustly valuable. Diffusion consistently prefers temporally sparse
> full-resolution anchors in this testbed, whereas regression makes temporal and spatial
> packages approximately tied across seeds with strong seed-level sign variation.

### 8.4 B3 — three spatial levels at fixed budget

Configuration: `hier3`, 48 top 2×2 frames + 24 coarse 4×4 frames + 9 fine 8×8 frames, exactly
1,152 context tokens.

- ALL 0.2044; sharpness 0.0564 (class-lowest);
- loses to hier-long by −0.0196 `[−0.0248, −0.0146]`;
- approximately null versus flat-budget.

This was outside the expected null range. Three mechanisms cannot be separated:

1. target/loss dilution across more decoded levels;
2. additional optimization burden;
3. the 2×2 top stream may be below the useful information floor for this single-scale task.

It is appendix-grade unless a cleaner weight-matched/control design is later justified.

### 8.5 Stream blinding

Eval-only stream-zeroing results:

| Probe | ALL | Main observation |
|---|---:|---|
| hier-long unblinded | 0.1848 | reference |
| blind coarse | 0.2236 | hard re-entry cost much larger than recency cost; coarse is retrieval-specific |
| blind fine | 0.2695 | renderer collapse, sharpness 0.0086; severe OOD brittleness |
| hier3 unblinded → blind top | 0.2044 → 0.2079 | removing top costs only ~0.0035 |

The coarse stream genuinely carries retrieval information. The fine stream is necessary for
rendering, not merely short-term detail. The near-null top blinding result converges with B3's
interpretation that the 2×2 stream contributes little useful information while still imposing
training/allocation costs.

Blinding is an out-of-distribution intervention. It is evidence of attribution plus brittleness,
not a perfectly isolated causal ablation.

---

## 9. Experimental results: trustworthy-memory boundary

The long-loop hgrid3 campaign separated nominal memory reach from usable self-fed memory.

Single-step / widened evaluation showed:

- beyond-window retrieval exists;
- it is reach-causal;
- span 224 > effective loop produces a strong hard/reachable signature;
- easy/mid and beyond-reach cells remain null.

The decisive self-fed rollout pair returned RED under its pre-committed gate:

- wins at h1–8 and h9–24 while memory contents are still near ground truth;
- null around h25–60;
- small systematic losses at h61+;
- all long-`dt` reachable-hard gate cells null.

This result prevents overclaiming. The current model has learned retrieval, but it has not learned
to preserve memory contents through a 213-frame purely self-fed loop. Future distribution-matching
rollout objectives are a more plausible remedy than merely increasing context depth.

---

## 10. External and transfer evidence

### 10.1 Standard Moving MNIST bridge

The canonical 10→10 bridge is a measured negative control for the long-memory claim:

- flat ~180.91 aggregate pixel-sum MSE;
- hierarchical ~181.29;
- effectively null at this short protocol;
- autoencoder ceiling ~3.30, so tokenization is not the limiting factor.

This benchmark contains persistent identity and velocity but no off-screen re-entry under the
short bridge. It cannot prove that long-horizon state is irrelevant, which is why Phase C exists.

### 10.2 TaxiBJ

The unchanged ~22M generalist stack was evaluated on TaxiBJ with disclosed ×2 upsampling and
native-resolution scoring.

Measured regression results include approximately:

- MAE 14.96;
- PSNR 39.83;
- MSE 0.3642;
- SSIM below the quoted leaderboard band.

This is used as a competence/economy anchor: the system is not a toy, and regression remains the
better deterministic head on real flow fields. It is not positive transfer of the hierarchy
claim because the 4→4 task has no meaningful memory horizon; hierarchy is economy-only there.

### 10.3 Phase C

Phase C will generate distinct train/test Moving-MNIST-L sets and evaluate flat versus spatial
hierarchy over roughly 300-frame rollouts using three training seeds. Metrics are:

- foreground MSE by horizon;
- identity retention;
- trajectory error;
- digit count;
- seed-visible equivalence margins.

A previous script bug would have generated train and test from the same RNG seed, making each test
trajectory a deterministic continuation of a training trajectory. This was caught before data
generation. The fixed script uses train seed 1234, test seed 5678, and aborts on any frame-0 hash
collision.

Interpretation is pre-registered:

- replicated equivalence supports the boundary/null account;
- a sound hierarchical win is reported as a real synthetic cross-dataset finding, not presumed
  to be an error;
- it is not presented as a CW-VAE reproduction or natural-video transfer.

---

## 11. Evidence discipline and audits

The project's strongest methodological feature is its evidence discipline.

- Major experimental branches were written before launch.
- Unexpected outcomes were preserved and used to change the theory.
- Symmetric pairs are run; best-of-seed cherry-picking is forbidden.
- Seed effects are reported individually and as means/ranges.
- Raw artifacts have been repeatedly recomputed against text values.
- A five-lens independent review recomputed 30/30 headline values to four decimals.
- Git history confirms sampled pre-registrations preceded their results.
- No published/current headline result was found to be fabricated or unsupported.

Important review files:

- `docs/REVIEW_FIVE_LENS_2026-07-10.md`
- `docs/SESSION_SUMMARY_2026-07-09.md`
- `docs/SESSION_SUMMARY_2026-07-10.md`
- `docs/SPATIAL_FACTORIAL.md`

The five-lens review found two genuinely critical issues:

1. Phase-C train/test contamination risk — fixed before launch.
2. The 1,792–2,432 upper bracket was presented too generally despite optimization-budget evidence
   that hierarchical arms converge later — wording budget-qualified and a 60k c2 control added.

Other accepted corrections include:

- context packaging rather than coarse hierarchy as the central claim;
- RoPE-ND/learned-embedding separation;
- sparse/keyframe conditioning acknowledged as established practice;
- no global “cheapest strong” superlative;
- B3 loss rather than stale expected null;
- CRN claims scoped to shape-compatible comparisons;
- seed 2 included in generated tables;
- “laws” softened toward empirical regularities under fixed conditions.

---

## 12. Current scientific vision

### 12.1 Main article claim

A concise current formulation is:

> At bounded context budget, long-horizon performance depends on how context is packaged across
> span, temporal rate, spatial resolution, and depth. Covering the relevant memory span is the
> robust requirement; the least lossy package depends on the data, representation, prediction
> head, and optimization basin.

For the current static-world testbed:

- contiguous flat history is token-expensive but wins once it covers the horizon at full
  resolution;
- dense coarse spatial history is an economical loop-covering package;
- sparse full-resolution anchors are the best diffusion package measured;
- regression leaves temporal and spatial packages tied across two seeds;
- extra spatial depth without an additional scale to own imposes a measured allocation cost;
- memory must remain trustworthy through rollout, not merely be addressable.

### 12.2 What is novel

The novelty is **not** inventing keyframes, temporal subsampling, coarse latent levels, or
hierarchical video models. The defensible novelty is the combination of:

- an analytic re-entry benchmark with exact memory-dependent labels;
- matched context-token-budget causal comparisons;
- a measured budget frontier with both inversions;
- decomposition of packaging into rate, resolution, allocation, depth, and head axes;
- attribution/blinding and trustworthy-memory boundary tests;
- pre-registered, seed-visible statistical discipline;
- integration into a broader systems-analysis methodology.

### 12.3 Scope limitations

- One synthetic static-world testbed carries most core packaging results.
- Effects are measured at 64×64 grayscale with one frozen representation.
- Static content favors temporal subsampling.
- The 30k upper edge may change at 60k.
- Two seeds are evidence of replication, not population-level seed inference.
- Depth is not cleanly isolated in B3.
- Natural-video re-entry transfer is untested.
- Long self-fed memory is not yet trustworthy at the 213-frame operating point.

---

## 13. Article state and recommended final form

### 13.1 Current article materials

Article drafts live in `docs/draft/`:

- `CH1_INTRO.md`
- `CH2_TESTBED.md`
- `CH3_MECHANISM.md`
- `CH4_RESULTS.md`
- `CH5_GENERALIZATION.md`
- `CH6_REVERSIBILITY.md`
- `CH7_RELATED.md`
- `CH8_CONCLUSION.md`

All chapters exist as prose, but the manuscript is still dissertation-shaped. Chapter 4 is the
scientific spine and is too operationally detailed for a final article. Commit hashes,
pre-registration quotations, queue language, and evidence-ledger mechanics should move to an
appendix/supplement.

### 13.2 Recommended six-section article

1. **Introduction and contribution** — context-packaging question, exact benchmark, bounded
   frontier, main shape result.
2. **Protocol and model family** — testbed, token definition, streams, heads, metrics, statistics.
3. **Matched-budget frontier** — lower edge, interior, saturation, optimization caveat.
4. **What packaging works and why** — B1/B2, regression interaction, blinding, B3 as appendix-grade.
5. **Boundary and external controls** — trustworthy-memory failure, Moving MNIST/TaxiBJ, Phase C.
6. **Discussion, limitations, related work, conclusion**.

EBT and reversibility should be compressed to appendices or separate work unless the chosen venue
requires the full dissertation-publication mapping. The article's identity is protected by focus,
not merely by shortening prose.

### 13.3 Article-essential versus non-blocking evidence

Article-essential:

- matched-budget frontier;
- B1/S1 family edge;
- B2/S1 temporal package;
- regression twins;
- seed-2 trio;
- 60k c2 optimization control;
- Phase C boundary test;
- limitations and negative controls.

Non-blocking/appendix-grade:

- B3 and top blinding;
- hier3 regression;
- E-320 third family point;
- full reversibility exposition;
- full EBT exposition;
- broad unmeasured transfer shells.

---

## 14. Dissertation state and chapter role

Dissertation rewrites live in `docs/dissertation/rewrite/`:

- `00_FRONT_MATTER.md`
- `01_VSTUP.md`
- `02_ROZDIL1.md`
- `03_ROZDIL2.md`
- `04_ROZDIL3.md`
- `05_ROZDIL4.md`
- `06_VYSNOVKY.md`
- `07_REFERENCES.md`

The current conceptual mapping is:

- **Introduction:** systems problem, tasks, novelty, defended propositions, practical value.
- **Chapter 1:** systems analysis of uncertainty/consistency and model families.
- **Chapter 2:** adaptive decomposition, RoPE-ND, reversible resource optimization and its
  applicability boundary.
- **Chapter 3:** multi-stream hierarchical latent generation model plus measurement instrument.
- **Chapter 4:** controlled experiments, frontier, packaging shapes, boundary studies, external
  controls, practical receipts.
- **Conclusions:** task-by-task results with explicit scope.

The load-bearing Task 5, defended proposition, front matter, Chapter 3 synthesis, and conclusions
have been reframed around multi-stream/context packaging. Some old wording remains in working
blueprints/skeletons and must not be treated as current authority.

### 14.1 Dissertation work that is not waiting on experiments

- Full Ukrainian authoring/translation pass: current rewrite files are English working text.
- Final front-matter/TOC synchronization; the TOC still carries an older Chapter 3 title.
- Resolve `[AUTHOR:]` fields: publication list, [A4] venue/status, [A2] tables, NDR/UDC,
  implementation acts, supervisor/council conventions.
- Verify `[REF:]` and `[VERIFY]` citations, especially MilliVid.
- Unify table/figure numbering and complete missing drawings/renders.
- Produce the all-black-baseline appendix table.
- Assemble TOC, abbreviations, appendices, counts, and final DOCX/PDF.
- Back up raw evidence artifacts.

The author-input cluster and Ukrainian pass are likely the actual defense critical path; no GPU
experiment can substitute for them.

---

## 15. Live execution state at this snapshot

The GPU queue is controlled by:

- `scripts/run_newvision_queue.ps1`
- `scripts/run_newvision_queue.sh`
- log: `logs_rhcp/newvision_queue.log`
- lock: `logs_rhcp/newvision_queue.lock`

The queue was launched in `core-only` mode. The optional hier3-reg/E-320 tail is not authorized to
auto-launch.

Completed in this queue:

1. S1-a — B1 seed 1: confirmed.
2. S1-b — B2 seed 1: confirmed direction with smaller margin.
3. `strided-flat-reg` seeds 0 and 1: head × seed sign flip, two-seed tie.

Running at document creation:

- `flat-short-reg`, seed 0, approximately 29.5k/30k; healthy.

Remaining automatic core order:

1. finish `flat-short-reg`;
2. train/evaluate `hier-s2-min-reg` and compare the 512 regression edge;
3. diffusion seed-2 trio: `hier-long`, `flat-budget`, `strided-flat`;
4. B8×60k `hier-c2` versus `flat-c2`, tagged `_60k`;
5. three-seed Phase C.

Do **not** start another queue controller. The existing process is lock-protected and writes
checkpoints incrementally.

---

## 16. Remaining experiments and their decision value

### 16.1 Required for current closure

#### 512-token regression pair

Asks whether the stride-2 family edge survives under direct regression. It expands L3's head
scope; it is not needed to prove the diffusion edge already replicated in B1.

#### Seed-2 critical trio

Adds a third visible training seed for the central contiguous-flat, spatial-long, and
temporal-strided diffusion configurations. It is especially important because hierarchy-bearing
arms are much more seed-variable than flat arms.

#### 60k c2 upper-edge control

Asks whether the 2,432-token flat win is a true saturation boundary or a 30k convergence artifact.

Pre-registered outcomes:

- flat still wins: the `(1,792, 2,432)` bracket is robust across 30k/60k at seed 0;
- null or spatial win: the 30k edge was optimization-conditional, and the 60k upper edge is
  unlocated at or above 2,432;
- mixed horizons: report unresolved; do not interpolate.

#### Phase C

Asks whether persistent identity/velocity become memory-relevant over long Moving-MNIST rollout.
It closes the external boundary column, not natural-video transfer.

### 16.2 Optional after core results

- `hier3-reg`: useful only if the B3 allocation/depth mechanism needs head resolution.
- E-320: third family edge point, scientifically weaker than varying the actual memory horizon.
- representation sensitivity: more valuable if spatial/temporal ordering remains tied or
  head-sensitive.

### 16.3 Highest-value future experiment

A controlled memory-horizon manipulation is the strongest next study. Current family edges are all
measured at approximately loop 48. A loop-24/48/96 factorial comparing contiguous flat, spatially
coarse, and temporally sparse full-resolution packages would test whether the edge scales with the
variable the claimed regularity is about.

A single loop-96 point is useful, but a small horizon-by-family dose response is preferable. It
must avoid confounding horizon with background identifiability, motion difficulty, training clip
coverage, or autoencoder reuse.

### 16.4 Flagship-level future requirement

Natural-video or geometry-grounded re-entry evaluation remains the main external-validity gap. It
is not necessary for a focused measurement venue but may determine acceptance at a flagship.

---

## 17. Text status and marker inventory

At the snapshot immediately before this dossier:

- the critical strategic reframe is integrated;
- B3 and blinding are integrated;
- the newly landed B1 seed-1, B2 seed-1, and regression-twin results are **not yet fully swapped
  into every article/dissertation site**;
- approximately 41 `[EXPECTED]` occurrences remain;
- approximately 301 lines contain some combination of `[AUTHOR:]`, `[REF:]`, figure placeholder,
  `[PENDING]`, or `[EXPECTED]` markers.

The remaining markers are not all experiment blockers. They fall into four classes:

1. stale result placeholders to replace now (S1/regression twins);
2. genuine core experiment placeholders (512-reg, seed 2, c2-60k, Phase C);
3. dissertation/future-programme shells that must be filled, moved, or struck before defense;
4. author/citation/figure/translation work unrelated to GPU evidence.

Final assembly must pass:

`venv\Scripts\python.exe scripts\check_text_consistency.py --gate`

The gate intentionally fails until all `[EXPECTED]`/`[PENDING]` content is resolved.

---

## 18. Honest strengths

1. **Evidence integrity.** Numbers trace to artifacts; independent recomputation has repeatedly
   matched the texts.
2. **Pre-registration discipline.** Surprises changed the theory rather than being hidden.
3. **Falsifiable boundary map.** Flat wins below and above the spatial hierarchy's useful band.
4. **B2 is a real discovery.** The project's nominal architecture lost to a simpler temporal
   packaging, and the project improved because this was reported honestly.
5. **Mechanistic instrumentation.** Hardness, horizon, `dt`, answerability, floors, ceilings, and
   blinding provide more than an aggregate benchmark score.
6. **Strong negative results.** EBT reachability, inner-loop reversibility, B3, and long-loop
   rollout failure are characterized rather than discarded.
7. **Consumer-GPU reproducibility.** The core programme is designed around one 16 GB GPU rather
   than inaccessible scale.
8. **Systems-analysis fit.** The dissertation can defend criteria, uncertainty, resource limits,
   decision rules, and applicability boundaries rather than only model accuracy.

---

## 19. Honest weaknesses and likely reviewer attacks

1. **One synthetic world carries most central regularities.** Static world coordinates are a
   substantial scope restriction.
2. **Seed inference remains thin.** Two or three visible seeds are honest but do not estimate a
   broad training-seed population.
3. **The title may promise hierarchy while the winner is sparse full-resolution history.** The
   article must make context packaging primary and state clearly what “hierarchical” means.
4. **Keyframe/strided history is standard practice.** Novelty lies in controlled measurement, not
   invention.
5. **B3 is confounded.** Depth, target allocation, parameter interfaces, and optimization burden
   are not cleanly separated.
6. **Blinding is OOD.** It supports attribution but also measures brittleness.
7. **Upper frontier may be training-budget conditional.** The 60k c2 run is essential.
8. **Long-memory contents are not self-sustaining.** The 213-frame rollout boundary is a real
   capability limitation.
9. **Article breadth is excessive.** EBT, reversibility, frontier, packaging shapes, boundary
   studies, and external suites cannot all receive equal main-text weight.
10. **Dissertation is not final Ukrainian prose.** This is substantial work, not formatting.
11. **Artifact preservation is fragile.** Local untracked evidence needs external backup.
12. **Author-dependent administrative fields remain.** These can block defense regardless of
   scientific completeness.

---

## 20. Recommended questions for an independent reviewer

Please review the project adversarially, with special attention to:

1. Is “context packaging” sufficiently falsifiable, or has hierarchy merely been broadened until
   the thesis becomes tautological?
2. Is the article's novel contribution clear relative to CW-VAE, keyframe conditioning, sparse
   attention, and hierarchical latent video models?
3. Does the head × seed sign flip require a different statistical treatment or more seeds before
   publication?
4. Does the current diffusion result justify “temporal packaging preferred,” or only “best in the
   two measured seeds on this static testbed”?
5. Is the planned 60k c2 pair sufficient to resolve the optimization-budget criticism?
6. Are token budgets, sequence lengths, target-token overhead, parameters, step time, and VRAM
   disclosed fairly enough to avoid “matched compute” overclaiming?
7. Are the clip-cluster intervals and seed-visible reporting appropriate for every headline?
8. Does B3 support any claim about depth, or should it be purely appendix diagnostic evidence?
9. Does the dissertation's systems-analysis spine genuinely integrate V-BLT, RoPE-ND,
   reversibility, and context packaging, or merely collect separate papers?
10. Which article content should be removed or moved to appendices?
11. Which dissertation novelty/proposition wording would be most vulnerable during defense?
12. What is the minimum additional external validation required for the intended venue?

---

## 21. File map for review

### 21.1 Read first

1. `docs/PROJECT_DISSERTATION_ARTICLE_REVIEW_DOSSIER_2026-07-11.md` — this document.
2. `docs/CLAIM_LEDGER.md` — claim status and scope (note: S1/reg rows need the next live swap).
3. `docs/SPATIAL_FACTORIAL.md` — chronological pre-registrations and measured branches.
4. `docs/REVIEW_FIVE_LENS_2026-07-10.md` — independent adversarial review.
5. `docs/CWSTYLE_TABLE.md` — generated measured shape table.
6. `docs/HGRID_DESIGN.md` — long-loop trustworthy-memory campaign.

### 21.2 Article

- `docs/draft/CH1_INTRO.md`
- `docs/draft/CH2_TESTBED.md`
- `docs/draft/CH3_MECHANISM.md`
- `docs/draft/CH4_RESULTS.md`
- `docs/draft/CH5_GENERALIZATION.md`
- `docs/draft/CH6_REVERSIBILITY.md`
- `docs/draft/CH7_RELATED.md`
- `docs/draft/CH8_CONCLUSION.md`

`docs/draft/ARTICLE_SKELETON.md` contains historical planning material and still has some old
coarse-memory framing. Treat chapter prose plus the current dossier/ledger as newer authority.

### 21.3 Dissertation

- `docs/dissertation/rewrite/00_FRONT_MATTER.md`
- `docs/dissertation/rewrite/01_VSTUP.md`
- `docs/dissertation/rewrite/02_ROZDIL1.md`
- `docs/dissertation/rewrite/03_ROZDIL2.md`
- `docs/dissertation/rewrite/04_ROZDIL3.md`
- `docs/dissertation/rewrite/05_ROZDIL4.md`
- `docs/dissertation/rewrite/06_VYSNOVKY.md`
- `docs/dissertation/rewrite/07_REFERENCES.md`
- `docs/dissertation/DISS_DIGEST.md` — restructure history and locked author decisions; contains
  superseded historical proposals, so follow explicit superseded labels.

### 21.4 Core code

- `hvp_rev/rhvd.py`
- `hvp_rev/hier_convae.py`
- `hvp_rev/hier3.py`
- `hvp_rev/faithful_nets.py`
- `hvp_rev/step1_mech.py`
- `hvp_rev/phasec_eval.py`
- `hvp_rev/phasec_summary.py`
- `scripts/gen_looping.py`
- `scripts/gen_moving.py`

### 21.5 Execution and verification

- `scripts/run_newvision_queue.sh`
- `scripts/run_newvision_queue.ps1`
- `scripts/run_cwstyle_seed1.sh`
- `scripts/run_cwstyle_regtwins.sh`
- `scripts/run_frontier_c2_60k.sh`
- `scripts/run_phaseC_mmnist.sh`
- `scripts/check_text_consistency.py`
- `scripts/make_cwstyle_table.py`
- `tests/smoke_cwstyle.py`

### 21.6 Evidence artifacts

- main queue log: `logs_rhcp/newvision_queue.log`
- per-arm evaluation: `logs_rhcp/rhvd_eval_{arm}_s{seed}[{tag}].npz`
- checkpoints: `logs_rhcp/rhvd_{arm}_s{seed}_t{target_steps}.pt`
- rollout strips: `logs_rhcp/rhvd_rollout_{arm}_s{seed}.png`
- figures: `docs/figures/`
- W&B project: `hgrid`

---

## 22. Reproducibility and operational notes

- The machine uses Windows/PowerShell.
- Plain `bash` may invoke a broken WSL installation. Use Git Bash explicitly:
  `C:\Program Files\Git\bin\bash.exe`.
- CPU smoke command:
  `$env:CUDA_VISIBLE_DEVICES='-1'; venv\Scripts\python.exe -m tests.smoke_cwstyle`
- Table/figure regeneration:
  `venv\Scripts\python.exe scripts\make_cwstyle_table.py`
- Text lint:
  `venv\Scripts\python.exe scripts\check_text_consistency.py`
- Final zero-placeholder gate:
  `venv\Scripts\python.exe scripts\check_text_consistency.py --gate`
- Never run two training controllers concurrently.
- Do not delete or overwrite unknown local NPZ/checkpoint artifacts.
- The repository may be dirty only because the live queue regenerated the CW table/figure; inspect
  before editing or committing.

---

## 23. Bottom-line assessment

The project is scientifically credible and considerably stronger than its initial architecture
story. Its best result is not that the proposed coarse hierarchy universally wins. Its best result
is that a disciplined causal programme discovered when compressed long history helps, when
full-resolution history catches up, when temporal anchors are preferable, when prediction heads
change the family ordering, when extra depth has no scale to own, and when nominal memory reach
fails because stored contents become untrustworthy.

The article can be a strong focused measurement paper if it resists the temptation to carry the
entire dissertation. The dissertation can be defensible as a systems-analysis synthesis if it
makes the joints between its strands explicit and does not imply that RoPE-ND or reversibility
caused Chapter 4's prediction gains.

The main scientific tasks still open are narrow and well-defined: the 512 regression edge, a
third seed for the critical trio, optimization robustness of the upper crossing, and the Phase-C
external boundary. The larger remaining workload is editorial and institutional: result swaps,
article restructuring, citations, figures, Ukrainian authoring, author-supplied fields, assembly,
and evidence backup.

For a new reviewer, the most useful posture is not “find whether hierarchy wins.” It is:

> **Audit whether the evidence truly supports the conditional context-packaging design rule, and
> whether the article and dissertation state that rule with enough precision, novelty discipline,
> and scope restraint to survive adversarial review.**

