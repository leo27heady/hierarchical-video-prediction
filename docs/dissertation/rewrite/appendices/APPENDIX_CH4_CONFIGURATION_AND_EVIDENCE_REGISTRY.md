# Supplementary Appendix — Chapter 4 Configuration and Evidence Registry

## Purpose and reading rules

This is the publication-facing appendix for Chapter 4. It replaces experiment chronology with a
curated registry of completed configurations, inferential families, resource fields, and evidence
bindings. It must be read with three rules:

1. an arm-level score is descriptive unless a paired comparison and interval are shown;
2. rows with different comparison licences are not cells of one causal factorial; and
3. **NOT_EXECUTED / OUT_OF_SCOPE** means that the frozen programme closed without that cell. It is
   not a pending result and cannot support a claim.

Effects follow the convention printed above each table. Clip-cluster intervals condition on a
trained model; training-seed summaries use training seeds as units. The primary evidence remains
the immutable result artifacts identified at the end of this appendix.

## A. Comparison licences

| Licence | Permitted reading | Prohibited reading |
|---|---|---|
| **MATCHED** | effect of the registered package pair at the named matched budget and interface | universal architectural superiority |
| **NEAR_MATCHED** | direction under the disclosed small budget difference | exact equal-budget causal effect |
| **MATCHED_CAPACITY** | mechanism comparison with closely matched backbone capacity and shared context | identical objectives or inference work |
| **WHOLE_CONFIGURATION** | performance of the complete executed configuration | effect of one co-changing component |
| **DEVELOPMENT_FACTORIAL** | seed-resolved effect and interaction on the adaptively used pool | untouched confirmation |
| **DEVELOPMENT_DIAGNOSTIC** | mechanism diagnosis with all stated confounds retained | claim-grade causal law |
| **ROBUSTNESS_ONLY** | sensitivity of already selected contrasts to the named intervention | new model selection or broader generalization |
| **REGISTERED_REPLICATION** | outcome-independent replication verdict for the frozen prediction | exploratory reinterpretation after failure |
| **PRIVILEGED_REFERENCE** | diagnostic value of explicit addressing or additional information | token-matched model ranking |
| **EXTERNAL_BOUNDARY** | scope behavior on a changed task and endpoint set | direct replication of LoopMom geometry |
| **DESCRIPTIVE_BOUNDARY** | competence or failure outside the central mechanism | equivalence, superiority, or mechanism confirmation |

## B. Complete observed LoopMom configuration grid

All scores below are foreground MSE over the 152-frame autoregressive rollout after the 48-frame
ground-truth warm-up; lower is better. Multi-seed entries show the mean and range over the executed
files named by the seed-coverage column; the four E4b cells were recomputed from the frozen input
artifacts so all five seeds are represented. They are descriptive receipts, not substitutes for the
paired, seed-resolved effects in Sections D–G. Resolution depth counts spatial levels;
`strided-flat` has one spatial level but two temporal rates. Except where stated, the regime is
B16×30k, bf16, EMA 0.999.

The **configuration row ID** is the stable one-to-one join key between Tables B.1 and B.2; repeated
arm names under different execution regimes receive distinct row IDs.

**Table B.1 — Package geometry and interface fields.**

| ID | Arm and package | Head | Reach | Context | Total | Spatial / rate depth |
|---|---|---|---:|---:|---:|---|
| LM01 | `hier-short`: 6 coarse + 6 fine | diffusion | 6 | 480 | 800 | 2 / 1 |
| LM02 | `flat-short`: 8 fine | diffusion | 8 | 512 | 832 | 1 / 1 |
| LM03 | `flat-short-reg`: 8 fine | regression | 8 | 512 | 832 | 1 / 1 |
| LM04 | `hier-s2-min`: 24 stride-2 coarse + 2 fine | diffusion | 48 | 512 | 800 | 2 / 2 |
| LM05 | `hier-s2-min-reg` | regression | 48 | 512 | 800 | 2 / 2 |
| LM06 | `spat18`: 18 coarse + 6 fine | diffusion | 18 | 672 | 992 | 2 / 1 |
| LM07 | `spat18-reg` | regression | 18 | 672 | 992 | 2 / 1 |
| LM08 | `flat-min`: 14 fine | diffusion | 14 | 896 | 1,216 | 1 / 1 |
| LM09 | `hier-min`: 48 coarse + 2 fine | diffusion | 48 | 896 | 1,216 | 2 / 1 |
| LM10 | `strided-flat`: 11 stride-4 fine anchors + 6 fine | diffusion | 48 | 1,088 | 1,408 | 1 / 2 |
| LM11 | `strided-flat-reg` | regression | 48 | 1,088 | 1,408 | 1 / 2 |
| LM12 | `flat-budget`: 18 fine | diffusion | 18 | 1,152 | 1,472 | 1 / 1 |
| LM13 | `flat-budget` | diffusion | 18 | 1,152 | 1,472 | 1 / 1 |
| LM14 | `reg-flat` | regression | 18 | 1,152 | 1,472 | 1 / 1 |
| LM15 | `hier-long`: 48 coarse + 6 fine | diffusion | 48 | 1,152 | 1,472 | 2 / 1 |
| LM16 | `hier-long` | diffusion | 48 | 1,152 | 1,472 | 2 / 1 |
| LM17 | `reg-long` | regression | 48 | 1,152 | 1,472 | 2 / 1 |
| LM18 | `hier3`: 48 top + 24 coarse + 9 fine | diffusion | 48 | 1,152 | 1,488 | 3 / 1 |
| LM19 | `flat-c1`: 28 fine | diffusion | 28 | 1,792 | 2,112 | 1 / 1 |
| LM20 | `hier-c1`: 48 coarse + 16 fine | diffusion | 48 | 1,792 | 2,112 | 2 / 1 |
| LM21 | `flat-c2`: 38 fine | diffusion | 38 | 2,432 | 2,752 | 1 / 1 |
| LM22 | `hier-c2`: 48 coarse + 26 fine | diffusion | 48 | 2,432 | 2,752 | 2 / 1 |
| LM23 | `flat-c2` | diffusion | 38 | 2,432 | 2,752 | 1 / 1 |
| LM24 | `hier-c2` | diffusion | 48 | 2,432 | 2,752 | 2 / 1 |
| LM25 | `flat-long`: 48 fine | diffusion | 48 | 3,072 | 3,392 | 1 / 1 |
| LM26 | `hier-rich`: 48 coarse + 36 fine | diffusion | 48 | 3,072 | 3,392 | 2 / 1 |

**Table B.2 — Execution, outcome, and evidence fields.**

| ID | Temporal / spatial factor | Executed seeds | Regime | Descriptive fg-MSE | Sharpness | Licence / role |
|---|---|---|---|---:|---:|---|
| LM01 | 1 / 2 | 0–1 | B16×30k | 0.2258 (0.2257–0.2260) | 0.0394 | near-matched lower reference |
| LM02 | 1 / 1 | 0–1 | B16×30k | 0.2215 (0.2206–0.2224) | 0.0426 | lower/B1 comparator |
| LM03 | 1 / 1 | 0 | B16×30k | 0.2205 | 0.0373 | B1 regression comparator |
| LM04 | 2 / 2 | 0–1 | B16×30k | 0.1881 (0.1869–0.1893) | 0.0712 | whole configuration B1 |
| LM05 | 2 / 2 | 0 | B16×30k | 0.1566 | 0.0831 | whole configuration B1 |
| LM06 | 1 / 2 | 0 | B16×30k | 0.2120 | 0.0498 | fixed-span diagnostic |
| LM07 | 1 / 2 | 0 | B16×30k | 0.2008 | 0.0480 | fixed-span diagnostic |
| LM08 | 1 / 1 | 0–1 | B16×30k | 0.2104 (0.2102–0.2106) | 0.0515 | matched 896 comparator |
| LM09 | 1 / 2 | 0–1 | B16×30k | 0.1725 (0.1706–0.1744) | 0.0848 | matched 896 candidate |
| LM10 | 4 / 1 | 0–4 | B16×30k | 0.1638 (0.1622–0.1652) | 0.0944 | near-matched package factorial |
| LM11 | 4 / 1 | 0–4 | B16×30k | 0.1657 (0.1574–0.1745) | 0.0823 | near-matched package factorial |
| LM12 | 1 / 1 | 0–2 | B16×30k | 0.2042 (0.2038–0.2046) | 0.0582 | matched 1,152 comparator |
| LM13 | 1 / 1 | 0 | B8×60k | 0.2032 | 0.0592 | optimizer-regime diagnostic |
| LM14 | 1 / 1 | 0–1 | B16×30k | 0.2009 (0.2004–0.2015) | 0.0514 | deterministic flat reference |
| LM15 | 1 / 2 | 0–4 | B16×30k | 0.1736 (0.1682–0.1848) | 0.0836 | matched 1,152 candidate |
| LM16 | 1 / 2 | 0 | B8×60k | 0.1666 | 0.0899 | optimizer-regime diagnostic |
| LM17 | 1 / 2 | 0–4 | B16×30k | 0.1632 (0.1503–0.1870) | 0.0777 | package factorial spatial arm |
| LM18 | 1 / 2×2 | 0 | B16×30k | 0.2044 | 0.0564 | depth-plus-objective/allocation diagnostic |
| LM19 | 1 / 1 | 0 | B16×30k | 0.1876 | 0.0737 | matched frontier comparator |
| LM20 | 1 / 2 | 0 | B16×30k | 0.1799 | 0.0779 | matched frontier candidate |
| LM21 | 1 / 1 | 0 | B16×30k | 0.1703 | 0.0893 | matched frontier comparator |
| LM22 | 1 / 2 | 0 | B16×30k | 0.1820 | 0.0771 | matched frontier candidate |
| LM23 | 1 / 1 | 0 | B8×60k | 0.1708 | 0.0889 | matched optimizer-regime comparator |
| LM24 | 1 / 2 | 0 | B8×60k | 0.1785 | 0.0795 | matched optimizer-regime candidate |
| LM25 | 1 / 1 | 0–1 | B8×60k | 0.1572 (0.1564–0.1580) | 0.0993 | matched equal-reach comparator |
| LM26 | 1 / 2 | 0–1 | B8×60k | 0.1691 (0.1687–0.1695) | 0.0893 | matched equal-budget hierarchy |

### Structurally absent cells

| Proposed cell | Status | Consequence |
|---|---|---|
| `flat-320` | **NOT_EXECUTED / OUT_OF_SCOPE** | no 320-token frontier point is claimed |
| `hier-s4-min` | **NOT_EXECUTED / OUT_OF_SCOPE** | no stride-4 coarse “edge law” is claimed |
| `hier3-reg` | **NOT_EXECUTED / OUT_OF_SCOPE** | no cross-head third-level conclusion is claimed |
| E5b/E5c localization | **NOT_AUTHORIZED after E5a** | the failed late-inversion replication is terminal |
| E9 | **SKIPPED_POST_FREEZE_NO_INCREMENTAL_UTILITY** | no further GPU evidence is required for the present claims |
| B1 fixed-target control | **NOT_EXECUTED / OUT_OF_SCOPE** | B1 remains permanently a whole-configuration result |

## C. Resource and protocol accounting

The standard rollout predictor has approximately 22.19 M trainable parameters. The third-level
probe adds approximately 0.30 M level-specific I/O parameters; its frozen top-level autoencoder
tap is a separately disclosed component. B16×30k and B8×60k both process approximately 480,000
training examples, but they are not the same optimizer regime.

Each spatial token contains 16 frozen-codec coefficients. The exact context-only array payload and
the width-384 embedding materialization are:

| Package | Tokens | Latent fp32 | Latent fp16/bf16 | Width-384 bf16 embeddings |
|---|---:|---:|---:|---:|
| 48 dense fine frames | 3,072 | 192 KiB | 96 KiB | 2.250 MiB |
| 48 coarse + 6 fine | 1,152 | 72 KiB | 36 KiB | 0.844 MiB |
| 48 coarse + 2 fine | 896 | 56 KiB | 28 KiB | 0.656 MiB |

These are context-state payloads, not peak VRAM. They exclude targets, attention/MLP
intermediates, allocator overhead, parameters, optimizer state and training activations; fp32
embeddings double the bf16 values.

| Regime / head | Batch | Updates | Approx. examples | EMA | NFE per rollout step | NFE per 38-step / 152-frame rollout | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| B16×30k diffusion | 16 | 30,000 | 480,000 | 0.999 | 16 default | 608 default | primary frontier regime |
| B8×60k diffusion | 8 | 60,000 | 480,000 | 0.999 | 16 default | 608 default | matched-example optimizer-regime control |
| B16×30k regression | 16 | 30,000 | 480,000 | 0.999 | 1 | 38 | deterministic rollout control |

Measured step time and peak VRAM were not captured for every arm under one profiling protocol.
Unmeasured fields are therefore reported as **NR**, not estimated.

| Arm | Context / total tokens | Measured seconds per training step | Measured peak VRAM | Scope |
|---|---:|---:|---:|---|
| `hier-s2-min` | 512 / 800 | 0.19–0.23 | 3.8 GB | B16 bf16 chain logs |
| `strided-flat` | 1,088 / 1,408 | 0.37–0.42 | 6.3 GB | B16 bf16 chain logs |
| `hier3` | 1,152 / 1,488 | 0.41–0.48 | 6.5 GB | B16 bf16 chain logs |
| `flat-c2` | 2,432 / 2,752 | 0.82–0.84 | 11.8 GB | B16 bf16 chain logs |
| all other arms | as in Section B | NR | NR | no cross-arm estimate is substituted |

Selected programme-level GPU charges that have independently recorded totals are E2 successful
evaluations (4.2169 h, plus approximately 0.01 h conservatively charged for an aborted pre-metric
attempt), E4a's two new regression cells (6.1228 h), E4b seeds 3–4 (24.9922 h), and E6's six
training/evaluation cells (20.5127 h). These are programme accounting receipts, not comparable
per-arm runtime benchmarks.

## D. Untouched frontier family

Effects are flat loss minus hierarchical loss, so positive values favor hierarchy. All intervals
belong to one studentized max-statistic clip-cluster family (critical value 2.741).

| Frozen contrast | Budget and regime | Seed | Effect | Simultaneous 95% CI | Decision |
|---|---|---:|---:|---|---|
| `hier-short` vs `flat-short` | 480 vs 512, B16×30k | 0 | −0.0038 | [−0.0081,+0.0005] | unresolved near-match |
| `hier-short` vs `flat-short` | 480 vs 512, B16×30k | 1 | −0.0060 | [−0.0105,−0.0015] | flat favored; near-match |
| `hier-min` vs `flat-min` | 896, B16×30k | 0 | +0.0474 | [+0.0305,+0.0642] | hierarchy confirmed |
| `hier-min` vs `flat-min` | 896, B16×30k | 1 | +0.0436 | [+0.0280,+0.0592] | hierarchy confirmed |
| `hier-long` vs `flat-budget` | 1,152, B16×30k | 0 | +0.0195 | [+0.0070,+0.0320] | hierarchy confirmed |
| `hier-long` vs `flat-budget` | 1,152, B16×30k | 1 | +0.0389 | [+0.0228,+0.0549] | hierarchy confirmed |
| `hier-long` vs `flat-budget` | 1,152, B16×30k | 2 | +0.0307 | [+0.0159,+0.0455] | hierarchy confirmed |
| `hier-c1` vs `flat-c1` | 1,792, B16×30k | 0 | +0.0084 | [−0.0013,+0.0181] | unresolved |
| `hier-c2` vs `flat-c2` | 2,432, B16×30k | 0 | −0.0150 | [−0.0260,−0.0039] | flat confirmed |
| `hier-c2` vs `flat-c2` | 2,432, B8×60k | 0 | −0.0122 | [−0.0213,−0.0032] | flat confirmed; separate optimizer regime |
| `hier-rich` vs `flat-long` | 3,072, B8×60k | 0 | −0.0129 | [−0.0217,−0.0041] | flat confirmed |
| `hier-rich` vs `flat-long` | 3,072, B8×60k | 1 | −0.0134 | [−0.0215,−0.0053] | flat confirmed |

The exact pair-derived reach effects for `hier-min` versus `flat-min` are +0.0980
[+0.0788,+0.1179] and +0.0914 [+0.0736,+0.1098] at seeds 0/1. For B1 versus
`flat-short`, the whole-configuration reach-band effects are +0.0692 [+0.0543,+0.0845] and
+0.0743 [+0.0588,+0.0899].

## E. Five-seed head × package factorial

Positive package effects favor `strided-flat` over `hier-long`. The two packages have the same
nominal reach but 1,088 versus 1,152 context tokens; the deficit favors the temporal package.

The absolute per-seed cells below are foreground MSE over the complete 152-frame rollout. They
make the factorial auditable without forcing reconstruction from effects.

| Seed | Diffusion temporal (`strided-flat`) | Diffusion spatial (`hier-long`) | Regression temporal (`strided-flat-reg`) | Regression spatial (`reg-long`) |
|---:|---:|---:|---:|---:|
| 0 | 0.1652 | 0.1848 | 0.1574 | 0.1682 |
| 1 | 0.1644 | 0.1690 | 0.1606 | 0.1503 |
| 2 | 0.1636 | 0.1747 | 0.1736 | 0.1528 |
| 3 | 0.1638 | 0.1713 | 0.1622 | 0.1870 |
| 4 | 0.1622 | 0.1682 | 0.1745 | 0.1579 |

| Seed | Diffusion temporal effect | Regression temporal effect | Head × package interaction |
|---:|---:|---:|---:|
| 0 | +0.0197 [+0.0143,+0.0252] | +0.0107 [+0.0062,+0.0158] | −0.0089 [−0.0151,−0.0025] |
| 1 | +0.0046 [+0.0007,+0.0084] | −0.0104 [−0.0146,−0.0062] | −0.0150 [−0.0199,−0.0104] |
| 2 | +0.0111 [+0.0070,+0.0151] | −0.0208 [−0.0270,−0.0150] | −0.0319 [−0.0401,−0.0245] |
| 3 | +0.0075 [+0.0037,+0.0114] | +0.0248 [+0.0181,+0.0323] | +0.0173 [+0.0106,+0.0245] |
| 4 | +0.0060 [+0.0027,+0.0092] | −0.0166 [−0.0229,−0.0108] | −0.0226 [−0.0297,−0.0160] |

| Estimand | Across-seed mean | SD | t95 interval | Terminal reading |
|---|---:|---:|---|---|
| diffusion package effect | +0.0098 | 0.0060 | [+0.0023,+0.0173] | temporal package stable across five tested seeds |
| regression package effect | −0.0024 | 0.0195 | [−0.0266,+0.0217] | heterogeneous; not equivalence |
| head × package interaction | −0.0122 | 0.0186 | [−0.0353,+0.0109] | sign-changing; no directional head law |

## F. Diagnostic context-form cells

| Diagnostic | Result | Licence and limitation |
|---|---|---|
| fixed 18-frame spatial abstraction, diffusion | `flat-budget` minus `spat18` = −0.0082 [−0.0115,−0.0050] | spatial coarsening saves 42% of context tokens but has a small diffusion cost |
| fixed 18-frame spatial abstraction, regression | `reg-flat` minus `spat18-reg` = +0.0008 [−0.0013,+0.0029] | unresolved at one training seed; not equivalence |
| depth-plus-objective/allocation | two-level `hier-long` error minus three-level `hier3` error = −0.0196 [−0.0248,−0.0146] | the three-level probe is worse, but target loss allocation, optimization load, stream allocation and top-level floor are inseparable; not “depth hurts” |
| blind coarse stream | ALL fg-MSE rises to 0.2236; hard re-entry increase +0.073 versus +0.022 at h1–8 | zeroing is out of distribution; direction supports retrieval use but magnitude is not information accounting |
| blind fine stream | h1–8 fg-MSE 0.2427; sharpness 0.0086 | renderer collapse plus OOD brittleness; no clean quantitative attribution |
| blind `hier3` top stream | ALL increase +0.0035 | tested top stream contributes little under this allocation; not a general top-level null |

## G. Sampler/NFE robustness family

Positive effects use the registered direction for each contrast: hierarchy for 896/1,152,
temporal packaging for 1,088-vs-1,152, and flat context for the 2,432 inversion.

| Contrast | NFE/step | Sampler seed | NFE/clip | Effect | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---|
| 896 reach | 8 | 0 | 304 | +0.0478 | [+0.0323,+0.0634] |
| 896 reach | 16 | 0 | 608 | +0.0465 | [+0.0310,+0.0621] |
| 896 reach | 32 | 0 | 1,216 | +0.0471 | [+0.0317,+0.0625] |
| 896 reach | 16 | 1 | 608 | +0.0482 | [+0.0325,+0.0638] |
| 896 reach | 16 | 2 | 608 | +0.0466 | [+0.0314,+0.0618] |
| 1,152 interior | 8 | 0 | 304 | +0.0152 | [+0.0045,+0.0259] |
| 1,152 interior | 16 | 0 | 608 | +0.0146 | [+0.0041,+0.0251] |
| 1,152 interior | 32 | 0 | 1,216 | +0.0140 | [+0.0036,+0.0245] |
| 1,152 interior | 16 | 1 | 608 | +0.0201 | [+0.0086,+0.0315] |
| 1,152 interior | 16 | 2 | 608 | +0.0200 | [+0.0087,+0.0313] |
| package shape | 8 | 0 | 304 | +0.0276 | [+0.0171,+0.0381] |
| package shape | 16 | 0 | 608 | +0.0283 | [+0.0172,+0.0394] |
| package shape | 32 | 0 | 1,216 | +0.0292 | [+0.0182,+0.0403] |
| package shape | 16 | 1 | 608 | +0.0251 | [+0.0160,+0.0342] |
| package shape | 16 | 2 | 608 | +0.0221 | [+0.0133,+0.0309] |
| 2,432 inversion | 8 | 0 | 304 | +0.0115 | [+0.0016,+0.0214] |
| 2,432 inversion | 16 | 0 | 608 | +0.0117 | [+0.0018,+0.0217] |
| 2,432 inversion | 32 | 0 | 1,216 | +0.0122 | [+0.0022,+0.0221] |
| 2,432 inversion | 16 | 1 | 608 | +0.0098 | [−0.0001,+0.0196] |
| 2,432 inversion | 16 | 2 | 608 | +0.0077 | [−0.0013,+0.0167] |

Terminal verdict: **DIRECTIONALLY_ROBUST**. This is checkpoint-only robustness conditional on
training seed 0, not training-seed or dataset robustness.

## H. Alternate-codec robustness family

| Codec | Contrast | Effect | Simultaneous 95% CI |
|---|---|---:|---|
| original | 896 reach | +0.0494 | [+0.0339,+0.0649] |
| original | 1,152 interior | +0.0283 | [+0.0169,+0.0397] |
| original | package shape | +0.0253 | [+0.0163,+0.0343] |
| original | 2,432 inversion | +0.0150 | [+0.0065,+0.0235] |
| alternate | 896 reach | +0.0521 | [+0.0374,+0.0669] |
| alternate | 1,152 interior | +0.0439 | [+0.0310,+0.0569] |
| alternate | package shape | +0.0112 | [+0.0039,+0.0184] |
| alternate | 2,432 inversion | +0.0079 | [−0.0006,+0.0165] |

Terminal verdict: **CODEC_DIRECTIONALLY_ROBUST**. The alternate row changes codec weights,
reconstruction, latent geometry and normalization together and remains conditional on predictor
seed 0.

## I. Registered negative, privileged reference, and external boundary

### I.1 Self-fed reliability replication

| Training seed | Early h1–24 | Late h61–256 |
|---:|---:|---:|
| 0 | +0.0204 [+0.0133,+0.0279] | −0.0064 [−0.0095,−0.0032] |
| 1 | +0.0082 [+0.0035,+0.0133] | +0.0074 [+0.0038,+0.0112] |
| 2 | +0.0230 [+0.0150,+0.0315] | +0.0087 [+0.0035,+0.0139] |

Terminal verdict: **NOT_REPLICATED**. No stable late trust boundary is established.

### I.2 Privileged pose-indexed raw-memory family

| Effect | Estimate | Simultaneous 95% CI |
|---|---:|---|
| raw K14 minus raw K48 | +0.0723 | [+0.0548,+0.0898] |
| raw K38 minus raw K48 | +0.0210 | [+0.0096,+0.0325] |
| `hier-min` minus raw K48 | +0.1219 | [+0.1031,+0.1407] |
| `hier-long` minus raw K48 | +0.1368 | [+0.1190,+0.1545] |
| `strided-flat` minus raw K48 | +0.1134 | [+0.0944,+0.1324] |
| `flat-c2` minus raw K48 | +0.1310 | [+0.1122,+0.1498] |

Terminal verdict: **RAW_ACCESS_CEILING_CI_CLEAN**. Exact future poses and raw warm-up pixels make
this a privileged reference rather than a neural baseline.

### I.3 Persistent-object long Moving-MNIST

For lower-is-better endpoints (foreground MSE, union MSE, trajectory error), effects are
flat minus hierarchy. For higher-is-better endpoints (identity accuracy, coverage), effects are
hierarchy minus flat. Positive therefore favors hierarchy for every column without changing any
endpoint denominator.

| Seed | Foreground MSE | Union MSE | Trajectory error | Identity accuracy | Coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.0036 [−0.0074,+0.0146] | +0.0102 [+0.0019,+0.0185] | +2.4786 [+0.2130,+4.7443] | −0.0762 [−0.1106,−0.0418] | +0.0304 [+0.0016,+0.0591] |
| 1 | −0.0031 [−0.0127,+0.0065] | +0.0122 [+0.0052,+0.0192] | +1.4223 [−1.0646,+3.9092] | −0.0561 [−0.0934,−0.0188] | +0.0159 [−0.0163,+0.0482] |
| 2 | −0.0024 [−0.0131,+0.0083] | +0.0169 [+0.0093,+0.0245] | +4.1325 [+1.5271,+6.7380] | −0.0647 [−0.0955,−0.0338] | +0.0498 [+0.0161,+0.0834] |

Terminal verdict: **MIXED_BOUNDARY**. Positive favors hierarchy. Hierarchy better preserves
support and continuity; flat context better preserves identity; foreground-only error is
unresolved and neither system solves the task.

## J. Evidence-to-artifact bindings

| Study | Frozen result artifact | SHA-256 |
|---|---|---|
| E2 untouched confirmation | `artifacts/confirmation_e2/results.json` | `85d01085e9ac074cd099e7b611101b30b7b7112b1f0230ea9163d9825cf2166e` |
| E4b head × package | `artifacts/e4b_head_package/results.json` | `eb917d948d852d022b872532b826bdb20fc1f17ecfddb55a76446f1821bc9e89` |
| E5a replication | `artifacts/e5a_trust_replication/results.json` | `c38244c207c24c17d6081a2dc75004b6ae96c63c7814d1cfdfbd87bb8ab5414b` |
| E7c sampler/NFE | `artifacts/e7c_sampler_nfe/results.json` | `93eef5069598b099f62013384e6d91978d7c9996f8fd4f67ab790486758054e6` |
| E7a alternate codec | `artifacts/e7a_codec_robustness/results.json` | `3f113628babb58cd9adf5223f04a28cc7bb54b59a451b6f5f9aa4ac8d2dd7173` |
| E8a pose mosaic | `artifacts/e8a_pose_mosaic/results.json` | `74b9357f950fbfa4e94160852c53205398f8efad8c2c453b1d8a726be22148be` |
| E6 persistent objects | `artifacts/phasec_v2/results.json` | `75a628ca6f2fa87e74e67bbe7b95d4acee6e0f35969d28fe97e72b0b40e981c7` |
| E6 completion validator | `artifacts/phasec_v2/completion_validation.json` | `091206aa6e2b1a9cf39f9cd4d921a2da8611216f8ecaf56e9bccb48d454915d8` |

The hash in a release manifest is the authority if this appendix and a later public archive ever
disagree. Dashboard records are mirrors; manuscript conclusions are regenerated from the frozen
local artifacts.

## K. Development mechanism and inference-allocation receipt

The corrected mechanism receipt executes regression and diffusion only. Although the harness also
prints the parameter count of an instantiated 7.28 M energy model, its mode list is
`('reg','diff')`; no energy result from that harness is admissible. Regression (7.27 M parameters)
was trained for 6,000 steps and diffusion (7.53 M) for 8,000 steps with the same six-frame,
two-level context interface. The principal held-out table uses six batches of 48 records and eight
candidates per record.

| Executed head | Best-of-8 fg-MSE, all | Easy | Hard novel strip | Mean over 8 candidates | Best-of-8 fg-sharpness |
|---|---:|---:|---:|---:|---:|
| regression with MC dropout | 0.0664 | 0.0232 | 0.1808 | 0.0714 | 0.1591 |
| diffusion, v-prediction + DDIM-16 | 0.0512 | 0.0120 | 0.1549 | 0.0640 | 0.1886 |
| persistence reference | 0.2367 | — | — | — | 0.0515 |

One regression candidate is one MC-dropout forward. One diffusion candidate is the declared number
of DDIM network evaluations. Consequently, best-of-8 does **not** equalize total inference work:

| Per-candidate NFE | Regression best-of-8 fg-MSE | Diffusion best-of-8 fg-MSE | Total regression / diffusion evaluations per record |
|---:|---:|---:|---:|
| 4 | 0.0680 | 0.0546 | 8 / 32 |
| 8 | 0.0680 | 0.0514 | 8 / 64 |
| 16 | 0.0680 | 0.0503 | 8 / 128 |

This NFE sweep uses two batches of 48 records, so its 16-NFE value is not substituted for the
six-batch main-table value. The licensed conclusion is matched-capacity/interface coverage plus
diffusion inference-allocation sensitivity, not equal-total-work superiority.

| Receipt | SHA-256 | Provenance class |
|---|---|---|
| `logs_rhcp/step1_mech_clean2.log` | `5333f8e8b74b38c6e9608b11f82e4f2494fa3425b31be75e1c6877cb622953e8` | hash-listed legacy log; not a content-addressed RunSpec result JSON/NPZ |

## L. Energy-based capability receipts

The energy-based evidence is a separate official-code adaptation, not a third matched-capacity
cell in Section K. Both S1 and S2 use the paper's `xxs` shape (width 384, 6 layers, 6 heads, FFN
multiplier 4), 16 context frames, 14.66 M parameters, effective fresh batch 128, 20,000 steps,
5,000-step warm-up, fp32, and one RTX 4060 Ti 16 GB. Their purpose is to separate energy-landscape
ordering from candidate-descent reachability.

| Recipe | Landscape-ordering receipt | Optimization / scale receipt | Descent-output receipt | Terminal reading |
|---|---|---|---|---|
| S2, fixed $\alpha=15.97$ | batch mean has lower energy than target on 21/21 probes; post-warm-up gap approximately 0.6–1.2 | loss approximately 0.597 | cold $d_{\mathrm{tgt}}=1.264$–1.292; fg-sharpness 0.0081–0.0144 | persistently misordered and unstructured |
| S1, learnable $\alpha:23.95\rightarrow33.27$ | target lowest on 15 consecutive probes from 6,000–20,000 steps; target-below-mean gap +0.22 to +3.09 | final loss 0.3614 | cold $d_{\mathrm{tgt}}=1.123$–1.166; cold fg-sharpness maximum 0.0345; paper-style best-of-4 $d_{\mathrm{tgt}}=1.186$–1.260 and sharpness 0.0102–0.0154 | ordering repaired; reachability still fails at tested scale |

The structured-output threshold was approximately 0.08 and the AE round-trip sharpness reference
was approximately 0.24. These values scope the observed collapse; they do not prove an impossibility
result for larger models, longer training, or different initializations.

| Receipt | SHA-256 | Provenance class |
|---|---|---|
| `logs_rhcp/ebt_scaled_s1.log` | `d1794167a4e9d6640cfd2c2ddd44965a7a7c71a572cdff4c823c45c2287e3365` | hash-listed legacy log; not a content-addressed RunSpec result JSON/NPZ |
| `logs_rhcp/ebt_scaled_s2.log` | `763221c3fe6bffea9d82270a768c1f0ea9db770edbdd9b88db89fd4748bd0568` | hash-listed legacy log; not a content-addressed RunSpec result JSON/NPZ |

## M. Chapter 2 reversible-memory and numerical-validation provenance

The retained end-to-end **deterministic** training-loop frontier was measured on an RTX 4060 Ti
16 GB with encoder, three-level core, decoder, and Adam active (batch 32, 64×64 resolution,
three levels). The asterisk denotes WDDM shared-memory spill beyond physical VRAM, not a valid
in-core GPU operating point. No stochastic innovations are supplied by this frontier path.

| Horizon $K$ | BPTT MB | Checkpoint MB | Reversible MB | BPTT ms | Checkpoint ms | Reversible ms |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3,809 | 1,354 | 1,065 | 549 | 322 | 257 |
| 20 | 6,592 | 1,391 | 1,078 | 399 | 544 | 479 |
| 40 | 12,162 | 1,471 | 1,091 | 740 | 1,028 | 882 |
| 80 | 23,305* | 1,637 | 1,125 | 43,797* | 3,131 | 2,556 |
| 120 | OOM | 1,801 | 1,152 | — | 3,960 | 2,760 |
| 160 | OOM | 1,967 | 1,176 | — | 4,066 | 3,362 |

The defensible description is an approximately horizon-independent reversible **total peak
allocation over the tested range**, with a modest measured edge over the implemented per-step
checkpoint comparator—not a universal constant-memory or universal speed theorem. The replay
routine uses spacing $\lfloor\sqrt K\rfloor$ and retains approximately $O(\sqrt K\,D_s)$ full-state
anchors; only its transient local activation graph is constant in $K$ under the stated model.

The separate energy-descent inner-loop panel shows why reversibility is not automatically useful
everywhere. At $K=128$, checkpointing uses 5.296 GB and 7.85 s/step, whereas reversible replay uses
5.282 GB and 15.37 s/step: only 14 MB (0.26%) is saved because other tensors dominate. The same
receipt reports CPU-fp64 gradient cosine greater than 0.999999 between reversible and checkpointed
variants. This is a numerical check for that implementation, not a general proof.

| Receipt / source | SHA-256 | Provenance limitation |
|---|---|---|
| `logs_rhcp/panelA_memfrontier.md` | `1349487ab314faac443dfb296269d88fa26a8a1a92d75d4c48256e4155bde81c` | measured legacy receipt; not a content-addressed result artifact |
| `logs_rhcp/mem_panel.md` | `8d863d8015770278d40767aac5ea994701e42e21accadce8b377723a93fe13c5` | measured legacy receipt; contains the inner-loop gradient-cosine statement |
| `hvp_rev/theory.py` | `0928c9b42c6f719048cfb872251650cf3fe5510232f887cfffe4f654a30fa0bd` | executable numerical-check source only; no standalone frozen T1–T4 output artifact was found |

The source contains four distinct checks. T1 tests deterministic reconstruction of one reversible
convolutional stage at thresholds $10^{-9}$ (fp64) and $10^{-4}$ (Lipschitz-constrained fp32). T2
tests 40-step store-the-noise reconstruction on one stage, requiring correct-noise drift below
$10^{-8}$ and wrong-noise drift above $10^{-2}$. T3 is a seeded 100-step perturbation-growth
diagnostic. T4 constructs a deterministic multi-level RHCP and requires reverse-replay/BPTT
gradient cosine above $0.9999$; despite its historical heading, the executed T4 path passes no
stochastic innovation. The memory table above is a separate deterministic training-loop
experiment. These scopes must not be merged into a single stochastic hierarchical result.

The absence of a standalone frozen T1–T4 output file is a review-edition provenance limitation,
not permission to reconstruct numerical results from prose. T1 and T2 also do not set an explicit
random seed. A future public release may add a seeded output receipt after rerunning the fixed
source, but the present dissertation cites only the encoded thresholds, measured legacy logs, and
source status above.
