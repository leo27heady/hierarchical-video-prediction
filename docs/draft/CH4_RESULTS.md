# Chapter 4 — Results: when does hierarchy pay?

This chapter reports the three experimental claims of the dissertation on the testbed family of
Chapter 2, followed by the temporal-capability boundary study at the second operating point.
Claim 2 (the generation head, §4.2) is settled first because every subsequent table depends on
knowing how much of an arm's score is owed to its head rather than its memory; Claim 1 (the
matched-token-budget memory result, §4.3) is the headline; Claim 3 (spatial abstraction, §4.4)
isolates the factor every hierarchy arm otherwise confounds. §4.5 reports the capability-regime
study at operating point B around its pre-committed decision rule, and §4.6 concludes.

## 4.1 Experimental setup recap

**Testbed family.** All experiments run on the memory-horizon testbed of Chapter 2 at two
operating points. **Operating point A (primary)** is the momentum looping world *loopmom*: a
192×64 toroidal world observed through a panning 64×64 viewport at speed 4 px/frame
($x_t = (x_{t-1} + 4a_t) \bmod 192$, $p_{hold} \approx 0.10$, $p_{cont} \approx 0.90$, measured),
so ~67% of the world is off-screen at any instant and off-screen content re-enters with a loop
period of ≈48 frames; train/test = 3,000×100 / 500×200 clips, clip-disjoint. **Operating point B
(capability regime)** is *hgrid3*, the same generator scaled to a 768-px world (12 digits, speed
4, near-ballistic circulation), whose effective loop is
$L_{\mathrm{eff}} = \mathrm{world}/(\mathrm{speed}\cdot(1-p_{hold})) \approx 768/(4\cdot0.9) \approx 213$
frames — deliberately beyond any affordable flat context window. Operating point B carries its own
frozen autoencoder (`hier_ae_hgrid3`, fine recon 0.0024 / coarse 0.0059) and its own re-measured
references: persistence fg-MSE 0.2381, AE round-trip ceiling fg-MSE 0.0108 (round-trip sharpness
0.2084), raw-GT sharpness 0.4218. The loopmom references (persistence 0.2367 single-step,
0.2552 under the rollout protocol; AE round-trip sharpness ceiling 0.2412; raw-GT sharpness
0.4785) do not transfer between operating points and are never mixed.

**Models.** The system arm (RHVD-mini) is a two-level latent diffusion model over the frozen AE
interface of §2.4: DiT denoiser d = 384, depth 8, heads 6 (22.19 M parameters), v-prediction, 16
DDIM steps, per-token roles/frame-ids/actions, 4-consecutive-frame prediction windows so the
training context layout equals the rollout context layout. Training: 30k steps, B = 16, bf16, EMA
0.999 (the flat-long arm is the disclosed exception, §4.3.4). Regression arms share the identical
backbone and token layout and replace the denoising head with direct latent prediction.

**Rollout protocol (operating point A).** Every rollout number in this chapter is measured over
**152 autoregressive frames after a 48-frame ground-truth warmup** (38 scheduler steps × 4
frames), best-of-1, on 100 held-out test clips with common random numbers across arms, paired
per-frame fg-MSE on decoded pixels under the GT foreground mask.

**Statistics.** All CIs quoted in this chapter are 95% paired **clip-level cluster bootstrap**
intervals. This is not optional hygiene: rollout records are 100 clips × 152 consecutive frames
with lag-1 autocorrelation +0.86, and frame-iid bootstrap CIs are 2–7× too narrow (the validity
audit found three secondary bin verdicts flipping to null under the correct intervals; all
headline verdicts survived). Records whose GT frame contains no foreground (3,263/15,200 = 21.5%,
where fg-MSE would silently fall back to full-frame MSE) are excluded from all stratified bins and
reported as their own bg-only row. Hardness is the exact novel-foreground label of §2.5, always
computed against the fixed 6-frame reference window (easy / mid / HARD re-entry), identically for
every arm. At operating point B the evaluation adds the **dt-since-last-visible stratifier**
(per-record time since the target region was last in view, computed from `rel_positions`;
pre-registered bins dt ∈ (48,96], (96,192], >192, crossed with hardness) and a warmup-SEEN
restriction so that gates are read only on re-entries of content that was answerable by
construction. One metric disclosure carried from the audit: per-horizon *sharpness* series are
chunk-batch means duplicated per record — valid for overall means and trend curves, never for
per-bin or per-clip readings; all sharpness-vs-horizon statements below are so scoped. Two
further scope notes. **Multiplicity:** the horizon × hardness matrices report per-cell CIs
without family-wise correction — the framing is estimation-first, and every headline claim rests
on a pre-registered primary cell plus independent-seed replication (and, in the band results, on
win-in-every-cell patterns), not on any single secondary bin; secondary-bin verdicts should be
read descriptively. **Training-window ranges:** arms with different context reach necessarily
sample training targets from different frame ranges (an arm with reach $r$ trains on targets
from frame $r$ onward), so shorter-reach flat arms see somewhat more distinct training targets —
a slack that favors the flat controls and is therefore conservative for every hierarchy win
reported; evaluation is reach-identical for all arms (fixed 48-frame warm-up). Full protocol,
floors and threats-to-validity: Chapter 2 and the validity audit
(`docs/draft/VALIDITY_AUDIT.md`).

## 4.2 Claim 2 — the generation head: a regime-dependent winner

The claim defended in this section is deliberately conditional: **which prediction head wins
depends on the inference regime; there is no one best head.** Three results establish this — the
single-step coverage study, its rollout reversal, and the two-sided energy-based study.

### 4.2.1 Single-step, oracle-coverage regime: diffusion wins every column

Table 4.1 reproduces the corrected mechanism comparison of Chapter 3 (single-step, k = 4,
best-of-8, NFE 16, matched ≈7.3–7.5 M arms, shared harness and standardised latents; hard split =
novel-strip > 0, 27% of samples).

**Table 4.1 — Matched-capacity head comparison, single-step (from Table 3.1; source
`logs_rhcp/step1_mech_clean2.log`).** References: persistence fg-MSE 0.2367 / sharp 0.0515;
raw-GT sharp 0.4785; AE round-trip sharpness ceiling 0.2412.

| Head | fg-MSE (all) | easy | HARD | mean-over-8 | fg-sharp |
|---|---:|---:|---:|---:|---:|
| Regression (MC-dropout) | 0.0664 | 0.0232 | 0.1808 | 0.0714 | 0.1591 |
| Diffusion (v-pred, DDIM) | **0.0512** | **0.0120** | **0.1549** | **0.0640** | **0.1886** |

Diffusion beats regression on every column, the separation concentrating on the genuinely
generative HARD bin (0.1808 → 0.1549), and the advantage is not an inference-budget artefact:
at matched NFE the regressor is flat (0.0680 at NFE 4/8/16) while diffusion improves
monotonically (0.0546 → 0.0514 → 0.0503). The regressor's mean-over-8 ≈ best-of-8 gap is near
zero — it averages over futures rather than distributing mass across them; diffusion's best-of-N
gap (0.0640 → 0.0512) is the coverage signature of a head that commits to distinct hypotheses.
This is the controlled replication of "regression averages, diffusion commits" (§3.4) — and its
protocol scope is the point: best-of-8 selection is an **oracle coverage regime**.

### 4.2.2 The rollout reversal: regression wins best-of-1

The pre-registered expectation for the rollout campaign (gate H2) was that the regressor, rolled
out autoregressively under the same scheduler, would fade toward the mean (the E1 precedent),
while diffusion would hold sharpness. The measured outcome **refutes** this — a reversal, reported
as such, not reframed: **reg-long beats hier-long (diffusion) at every horizon band and every hard
bin at equal sharpness** (fg-sharp 0.0741 vs 0.0729; ALL fg-MSE 0.1682 vs 0.1848; direct paired
cluster CI on the ALL delta [+0.0110, +0.0227]). The same holds at matched flat context:
reg-flat 0.2015 vs flat-budget (diffusion) 0.2038.

The two results reconcile without contradiction once the protocols are stated side by side: Table
4.1's diffusion win is single-step **best-of-8** — oracle coverage switched on; the rollout
headline is **best-of-1** — coverage off, and every sampled hypothesis is fed back and compounded.
Diffusion wins the coverage regime; regression wins the deterministic best-of-1 rollout regime.
Any claim of the form "head X is better" that omits the regime is, on this evidence, wrong.

### 4.2.3 The energy-based head: a two-sided negative at reachable scale

The third head, energy-based descent (Energy-Based Transformers), is studied exhaustively in
Chapter 3 §3.5 with the official implementation imported unchanged and both published training
recipes run at the paper's xxs shape (14.66 M, 20k converged steps, effective batch 128); we
condense the outcome here because it completes the head comparison. The study is two-sided in
that it decomposes the failure into two separable capabilities. **Landscape ordering is
recipe-dependent:** the S2 recipe converges to a stably *mis-ordered* landscape — E(mean) <
E(target) at all 21 probes over 20k steps — while the S1 recipe (the paper's generation recipe)
*repairs* the ordering, E(target) strictly lowest at all 15 consecutive probes from step 6k to
20k. **Descent reachability fails under both:** cold descent stays at latent distance
d_tgt ≈ 1.12–1.29 from the target (the batch mean itself sits at 0.964), decoded foreground
sharpness never exceeds 0.035 against the 0.08 structure threshold, and the paper's own inference
recipe (best-of-4 min-energy, 4 steps, Langevin 1) yields d_tgt 1.207 / fg-sharp 0.0127 (current-α
reading) and 1.233 / 0.0120 (init-α reading) at the final probe. A correctly ordered energy is
therefore not sufficient for generation at this scale — ordering and reachability dissociate.
The scope is stated in §3.5.5 and the configuration-gap table (Table 3.3): this is a capability
boundary at 16 GB-reachable scale, not a reproduction failure and not a contradiction of the
paper's claims; the diffusion control at half the parameter count generates on the same latents
(§3.5.6). For the head comparison the operational consequence is that energy descent contributes
no usable arm at either operating point, and the postponed testbed-B EBT cells (HGRID §9) remain
a characterization row for a later slot, not a gap in the present claims.

### 4.2.4 Verdict

Claim 2, as supported: **the head ranking is regime-dependent** — diffusion for single-step
best-of-N coverage, regression for the deterministic best-of-1 rollout, energy descent for
neither at this scale. The memory results that follow are therefore run with *both* viable heads
wherever the claim depends on it (§4.3.5, §4.4), so that no memory conclusion rides on a head
choice.

## 4.3 Claim 1 — hierarchical memory pays at matched token budget

### 4.3.1 Pre-test: retrieval emerges under a larger optimization budget

Before any rollout machinery, the single-step memory gate asked whether a long *coarse* context
helps at all: coarse-context 48 vs 6 at fixed fine context 6 (1,232 vs 560 tokens), best-of-8,
stratified by the exact hardness label. The result is an emergence-with-budget pattern: at 6k
steps the HARD-bin delta is null on both seeds (+0.0014 [−0.0004, +0.0033] and +0.0008 [−0.0009,
+0.0025]); at 20k steps it is a decisive WIN on both seeds (+0.0135 [+0.0080, +0.0196] and
+0.0143 [+0.0083, +0.0208]; hard-bin absolute 0.1053 → 0.0918 on seed 0), with sharpness tied
(≈0.188 both arms) — the win is retrieval, not sharpness. Four disclosures. First, "2 seeds"
means two *training* seeds sharing one CRN evaluation set (by design, for pairing). Second,
these evaluation windows are near-independent rather than fully clustered, so the audit's
worst-case design-effect correction widens the 20k hard-bin CIs to ≈[+0.005, +0.022] (s0) and
≈[+0.006, +0.023] (s1) — the WIN survives. Third, **each budget is its own complete,
budget-matched training run** (a fully annealed 6k schedule vs a fully annealed 20k schedule),
not a single run probed at two checkpoints — so the contrast is *budget*, with schedule shape a
bundled covariate; the 6k arms were, if anything, the better-converged endpoints of their own
schedules, which works against the objection that they were merely mid-anneal. Fourth, the 20k
budget was a **post-null follow-up**: the originally registered read was at 6k, its null was
recorded, and the 20k extension was then run and replicated on both seeds — the escalation is
part of the record, and the claim rests on the twice-replicated 20k result, not on a selected
read. The methodological moral carries through this whole chapter: retrieval-from-memory needs
a *large optimization budget*, and short-budget ablations (6k steps here) would have spuriously
killed it.

[FIG ch4-1: hard-bin Δ with CIs at 6k vs 20k × 2 seeds (data DONE, T3; plot QUEUED). Caption:
retrieval emerges under the larger optimization budget — null at 6k on both seeds, CI-clean win
at 20k on both (each budget its own annealed schedule).]

### 4.3.2 The rollout matrix

Table 4.2 gives the full arm ordering under the rollout protocol (ALL-row fg-MSE over 152 AR
frames after the 48-frame GT warmup, seed 0).

**Table 4.2 — Rollout arm ordering, operating point A (source
`logs_rhcp/compare_clustered.log`).**

| arm | context | ctx tokens | ALL fg-MSE | fg-sharp | note |
|---|---|---:|---:|---:|---|
| flat-long (diff) | 48 fine frames | 3072 (2.7×) | **0.1564** | **0.0996** | B=8×60k sample-matched (caveat, §4.3.4) |
| reg-long | 48c+6f, regression | 1152 | 0.1682 | 0.0741 | |
| hier-long (diff) | 48c+6f | 1152 | 0.1848 | 0.0729 | THE system arm |
| reg-flat | 18 fine frames, regression | 1152 | 0.2015 | 0.0512 | |
| flat-budget (diff) | 18 fine frames | 1152 | 0.2038 | 0.0582 | token-matched flat control |
| hier-short (diff) | 6c+6f | 480 | 0.2260 | 0.0392 | memory ablation |
| persistence | — | — | 0.2552 | — | floor |

Three laws are visible in the ordering, and each is defended separately below: (i) **within each
head, the hierarchical long-memory arm beats its token-matched flat control** (hier-long <
flat-budget; reg-long < reg-flat); (ii) **across heads, regression wins the best-of-1 rollout**
(§4.2.2); (iii) **the full-resolution long window wins outright at 2.7× the tokens** (§4.3.4).
Both long-memory arms also beat hier-short in every bin — long context helps — but the claim
defended here is the *hierarchical packaging* of it at matched budget, which is exactly what the
flat-budget control isolates.

[FIG ch4-2: per-horizon rollout curves — asset EXISTS: `logs_rhcp/fig_horizon_curves_loopmom.png`.
Three panels over rollout horizon h (frames after the 48-frame GT warmup): (a) fg-MSE vs h on all
frames and (b) on HARD re-entry frames only (novel-fg-at-6 > 0.1), seven series each — hier-long,
flat-budget, flat-long, reg-long, reg-flat, hier-short, persistence floor — with cluster-CI bands;
(c) foreground-sharpness retention vs h (batch-mean series; AE round-trip ceiling 0.2412 marked).
Caption: matched-budget hierarchy (hier-long vs flat-budget) separates exactly on the
memory-dependent frames and horizons; flat-long is the outright frontier at 2.7× tokens;
sharpness fades monotonically for all arms.]

### 4.3.3 The headline pairing at both seeds: the win is horizon-gated

Table 4.3 is the headline: hier-long vs flat-budget at equal 1,152 context tokens, the full
horizon × hardness matrix (Δ = flat − hier, positive = hierarchy better).

**Table 4.3 — hier-long vs flat-budget (seed 0), 152 AR frames after 48-frame GT warmup, paired
per-frame fg-MSE, clip-level cluster bootstrap; empty-GT-fg records (21.5%) excluded from bins
(source `logs_rhcp/compare_clustered.log`).** Sharpness 0.0729 vs 0.0582; persistence 0.2552.

| horizon | hardness | n | flat-budget | hier-long | Δ | 95% CI | verdict |
|---|---|---:|---:|---:|---:|---|---|
| h 1–8 | easy | 434 | 0.0440 | 0.0419 | +0.0022 | [−0.0040, +0.0089] | null |
| h 1–8 | HARD | 150 | 0.1435 | 0.1359 | +0.0075 | [−0.0164, +0.0327] | null |
| h 9–24 | easy | 919 | 0.1216 | 0.1189 | +0.0028 | [−0.0075, +0.0133] | null |
| h 9–24 | mid | 45 | 0.1661 | 0.1624 | +0.0038 | [−0.0121, +0.0182] | null |
| h 9–24 | HARD | 282 | 0.1800 | 0.1728 | +0.0071 | [−0.0110, +0.0260] | null |
| h 25–60 | easy | 2286 | 0.2210 | 0.1989 | +0.0221 | [+0.0080, +0.0368] | **WIN** |
| h 25–60 | mid | 79 | 0.2218 | 0.2036 | +0.0182 | [−0.0049, +0.0437] | null |
| h 25–60 | HARD | 550 | 0.2554 | 0.2209 | **+0.0344** | [+0.0162, +0.0531] | **WIN** |
| h 61+ | easy | 5610 | 0.3060 | 0.2788 | +0.0272 | [+0.0166, +0.0382] | **WIN** |
| h 61+ | mid | 209 | 0.3224 | 0.2842 | +0.0382 | [+0.0178, +0.0616] | **WIN** |
| h 61+ | HARD | 1353 | 0.3073 | 0.2759 | **+0.0314** | [+0.0202, +0.0431] | **WIN** |
| bg-only | | 3263 | 0.0067 | 0.0047 | +0.0020 | [+0.0013, +0.0026] | WIN |
| ALL | | 15200 | 0.2038 | 0.1848 | **+0.0189** | [+0.0121, +0.0264] | **WIN** |

On seed 0 the matrix reads as the claim in one table: **null in every bin below horizon 25; a
CI-excluding-zero win in every easy/HARD bin from h 25–60 onward** (the sparse mid cell, n = 79,
is null at h 25–60 and WIN at h 61+). The advantage appears exactly where the memory horizon
demands it — h ≈ 24 is half a loop period, past which the horizon exceeds what 18 flat frames
carry — and nowhere else.

**Seed-1 replication (pre-registered before running).** The same pairing retrained at seed 1 wins
**in every cell**: ALL +0.0354 [+0.0267, +0.0447]; HARD h 25–60 +0.0622; sharpness 0.088 vs
0.057. Two-seed status of the headline: **CONFIRMED** (seed 0: ALL +0.0189 [+0.0121, +0.0264]).
The honest nuance is the onset boundary: it is seed-variable — seed 0 is null below h 25 while
seed 1 wins already from h 1–8, and the hier-long arm itself moves 0.1848 → 0.1690 across seeds
over a stable flat baseline. The robust, two-seed claim is therefore **the win and its
mid-horizon concentration, not the exact onset frame**; the "no advantage before h 25" phrasing
is a seed-0 observation, retained only as such.

**Per-horizon behaviour and the deep-horizon disclosure.** hier-long is below the persistence
floor at *every* horizon out to 152 frames (>3 loop periods); its error saturates at 0.221–0.222
over the last 70 frames rather than diverging — bounded drift. Sharpness, however, fades
monotonically for **all** arms (hier-long 0.2023 at h 1–4 → 0.0323 at h 105–152, batch-mean
series): the pre-registered sharpness-retention clause of gate G2b (>0.5× single-step sharpness at
depth) is **not met** at h ≥ 81, and we state this plainly as the deep-horizon limitation rather
than tuning it away. The blur ceiling is owned by the frozen decoder (AE round-trip 0.2412), not
by the dynamics.

[FIG ch4-3: horizon × hardness win map — Δ(flat − hier) with CI-based verdict shading, seeds 0
and 1 side by side (data DONE, T4 + seed-1; plot QUEUED). Caption: the matched-budget win is
horizon-gated on seed 0 and everywhere-positive on seed 1; the shared feature is the mid-horizon
peak on hard re-entry bins.]

[FIG ch4-4: qualitative rollout strips at re-entry events — asset `docs/figures/
fig7_samples_claim1.png`. Caption (honest reading per the figures-audit — the earlier
"re-places vs hallucinates" caption OVER-CLAIMED and is retracted): both arms lose sharpness at
deep horizons; the hierarchical advantage is a modest average improvement concentrated on hard
re-entry bins, visible as better-localized (though blurred) content at re-entry positions, not
as a categorical re-place-vs-hallucinate difference.]

### 4.3.4 flat-long: the Pareto frontier at 2.7× tokens

The flat-long arm (48 full-resolution frames, 3,072 tokens) beats every other arm on both error
and sharpness: ALL 0.1564, fg-sharp 0.0996; against flat-budget its ALL delta is +0.0473
[+0.0362, +0.0592]. This is reported with the same prominence as the headline, because it is what
keeps the claim honest: **at this operating point, hierarchy is an efficiency point, not a
capability necessity.** The hierarchical arm recovers most of the long-memory benefit at ~37% of
the context tokens; a party willing to pay 2.7× buys the frontier outright. (Protocol caveat,
disclosed wherever flat-long is quoted: it trained at B = 8 × 60k steps — sample-matched to
B = 16 × 30k, but with a different EMA horizon and gradient-noise scale than the other arms.)
The regime where no affordable flat window can cover the loop at all is exactly what operating
point B exists to test (§4.5).

### 4.3.5 Mechanism-agnosticism: the memory win is not a diffusion artefact

The pre-registered mechanism-agnosticism gate (red-team P0) required reg-long to beat the
token-matched reg-flat with CI excluding zero in the hard/high-horizon bins. Outcome: **PASSED —
reg-long beats reg-flat in every bin at BOTH seeds** (ALL +0.0334 [+0.0248, +0.0423] seed 0; +0.0501 [+0.0396, +0.0612] seed 1 — two-seed; hard-bin wins at every
horizon including h 1–8 HARD +0.0447; sharper, 0.0741 vs 0.0512). Notably the regression pairing
shows no null prelude — the win starts at h 1–8 — so the horizon-gating observed on the diffusion
pairing at seed 0 is a property of that head/seed combination, not of the memory mechanism.
Combined with Table 4.3, the matched-budget hierarchy win now holds under **both** viable heads:
whatever the head, packaging the same token budget as long-coarse-plus-recent-fine beats spending
it on a shorter full-resolution window. This is the sense in which Claim 1 is
mechanism-agnostic.

## 4.4 Claim 3 — spatial abstraction: economic and head-conditional, not intrinsic

Every hierarchy arm so far confounds two factors: coarse spatial resolution of the history, and
the longer window its cheaper tokens buy (an accepted external-review finding). The Claim-3
factorial isolates the resolution factor: a window × history-resolution grid on operating point A
(shared frozen AE, seed 0, protocol identical to the rollout campaign so all existing records
pair exactly), holding recency equal — every coarse-history arm keeps the 6 most recent fine
frames, exactly the hier-long design. The claim under test, in the author's original form: coarse
history is a *more abstract, less distracted* memory representation.

**Table 4.4 — Window × history-resolution factorial (fg-MSE ALL over 152 AR frames, cluster CIs;
context tokens in parentheses; source `docs/SPATIAL_FACTORIAL.md`, MEASURED RESULTS, from
`rhvd_eval_spat18{,-reg}_s0.npz` + the existing campaign records).**

| window | full-res history | coarse history | paired Δ (full − coarse) |
|---|---|---|---|
| 18f, diff | flat-budget 0.2038 (1152) | spat18 0.2120 (672) | **−0.0082 [−0.0115, −0.0050] LOSS** (short-h concentrated, null by h>120; sharp 0.058 vs 0.050) |
| 18f, reg | reg-flat 0.2015 (1152) | spat18-reg 0.2008 (672) | **+0.0008 [−0.0013, +0.0029] null** (h61–120 HARD +0.0047 WIN; tiny h1–8 loss) |
| 48f, diff | flat-long 0.1564 (3072) | hier-long 0.1848 (1152) | −0.0284 (from Table 4.2/4.3; 2.7× token gap) |

**G-S1 (abstraction sufficiency, W18) — head-dependent.** For the diffusion head, coarse history
is a small, real, short-horizon-concentrated cost: −0.0082 ALL at 42% token savings. For the
regression head — the better rollout head (§4.2.2) — coarse history is **statistically free**:
+0.0008 [−0.0013, +0.0029], null ALL at the same 42% savings, with the long-horizon HARD bin
slightly *better* (h61–120 HARD +0.0047). The "intrinsically better abstraction" hypothesis is
therefore **not supported**; the supported form of Claim 3 is **economic and head-conditional**:
spatial abstraction is lossy compression whose token savings buy the memory window, and under the
regression head the loss term is ≈0.

**G-S2 (interaction).** The abstraction cost per saved token is similar at both windows for
diffusion (≈0.007–0.008 per 480 tokens); there is no evidence the cost shrinks as the window
grows.

**G-S3 (window at fixed representation).** hier-long 0.1848 (W48, 1152 tok) vs spat18 0.2120
(W18, 672 tok): extending the window via coarse tokens buys −0.027 — an order that dwarfs the
abstraction cost, the same window-ordering law as Table 4.2.

**Combined-optimum readout.** spat18-reg (672 tokens) at 0.2008 matches or beats the diffusion
flat arm flat-budget (1,152 tokens) at 0.2038 — regression + coarse history at 58% of the tokens;
and with the long window (reg-long 0.1682 at 1,152 tokens) the recipe **regression head + coarse
history + long window** is the cheapest strong configuration measured in this campaign.

**Honest link to the prior spatial-gate null.** This project's earlier per-pixel spatial-gate
study on Moving MNIST found that an upper spatial level engaged unstably and added no quality —
a clean null for spatial abstraction as an *intrinsic* representational win. The factorial is
consistent with that history rather than overturning it: isolated from the window confound,
coarse history is not intrinsically better (it costs under diffusion, and is at best free under
regression); its value is what its cheapness buys. One declared omission: the temporal-only
corner (strided full-resolution history) was not re-imported into operating point A, because the
strided mechanism is separately characterized at operating point B (§4.5) and the W×R grid
answers the resolution claim without it.

## 4.5 The temporal-capability boundary study (operating point B)

Operating point A proves *efficiency at matched budget*: the loop (~48) sits inside the
affordable flat window (48 frames), so flat-long can always buy the result with tokens. The
H-grid study (`docs/HGRID_DESIGN.md`, pre-registered and frozen before any launch) asks the
capability question: in a regime where **no affordable flat window covers the memory horizon**
(hgrid3: effective loop ≈ 196–213 frames), does temporal abstraction — a strided, slow-rate
coarse level — deliver what flat context cannot? At the time of writing, the single-step
characterization is measured and the pre-committed disambiguation runs are pending; this section
therefore presents the shared narrative spine, quotes the frozen decision rule, and provides both
resolution branches explicitly marked. Exactly one branch will survive to the final text.

### 4.5.1 The aliasing episode: a diagnosed failure, not a discarded run

The first dataset iteration (*hgrid*, world 768 with background wavenumbers scaled ×4 to preserve
the frozen loopmom autoencoder) produced a dead-flat null at the 20k single-step pre-check (HARD
−0.0008 [−0.0027, +0.0012]; all dt bands null) and a CI-clean **LOSS** at the budget-matched 40k
extension (HARD −0.0025 [−0.0042, −0.0008]; ALL −0.0012; dt>192 −0.0026): more optimization made
the unaddressable memory slots pure attention cost. The cause is established, not speculative:
the ×4 wavenumber scaling made the background **exactly 192-px periodic**, so the four aliased
viewport positions are pixel-identical — the crop test measures pairwise MSE **0.000000** between
them. Retrieval lost the absolute-position landmark it plausibly used on loopmom (where k = 1
components made positions unique). The AE-reuse constraint *caused* the failure and was dropped
(fresh AE per testbed variant; within-grid comparability needs only one shared AE per campaign).
The second iteration (hgrid2, a DC brightness cue) was aborted mid-run for the mechanistically
stronger variant; two position-probe attempts from that period failed their own positive controls
and are retracted on the record. The third iteration, **hgrid3** (unscaled background with k = 1
forced), restores position uniqueness (crop-uniqueness 0.004–0.010 vs hgrid2's 0.0006–0.003) and
passes the full two-sided regime verification: median re-entry gap 192 frames, 204 beyond-48
re-entry frames per clip, warmup world-coverage 88.5%.

### 4.5.2 Geometry: memory span vs the effective loop

The pre-check arm carries 48 coarse slots at stride 4 — a memory span of 192 frames. The relevant
horizon, however, is not $\mathrm{world}/\mathrm{speed} = 192$ but the **effective loop**

$$L_{\mathrm{eff}} \;=\; \frac{\mathrm{world}}{\mathrm{speed}\cdot(1-p_{hold})} \;\approx\; \frac{768}{4\cdot 0.9} \;\approx\; 213 \ \text{frames},$$

because hold frames advance time without advancing the pan. The 48-slot arm's span of 192 frames
therefore falls *short* of its own target loop (~196–213 measured) — a geometry error inherited
from the loopmom intuition, and the motivation for the span-224 arm below.

### 4.5.3 Reachability exists: the widened-eval result and its internal control

The 4×-widened paired re-evaluation of the same hgrid3 models (eval_nb 24 → 96; pre-registered
before running; both reads disclosed) flips the underpowered nb = 24 nulls to CI-clean wins for
the strided arm — **48-slot stride-4 (reach 195 frames): HARD +0.0016 [+0.0007, +0.0025] WIN;
REACHABLE-HARD +0.0021 [+0.0008, +0.0036] WIN; dt 49–96 +0.0032 WIN; ALL +0.0005 WIN** — while
the contiguous-48 positive-control arm (loopmom's exact winning design, but reach only 51 frames
here) is **null in every cell**. The effect appears exactly where the memory reach covers the
re-entry interval and vanishes where it cannot: a reach dose-response, serving as the internal
control. Two readings attach. First, beyond-window retrieval on the position-unique wide world
**exists** but is ~7× smaller than loopmom's (+0.002 vs +0.0135 absolute; ~2% vs ~13% relative) —
consistent with the span shortfall diluting it and with the harder addressing problem at this
scale. Second, the nb = 24 pre-checks were underpowered for this magnitude, not wrong about the
aliased hgrid (whose 40k read was a CI-clean LOSS; the aliasing verdict stands).

### 4.5.4 The span-224 result and the pre-committed decision rule

The pre-approved Path-B shot — 56 slots at stride 4, span 224 > loop, 40k steps, eval_nb 96 —
returned, on a single seed: **HARD +0.0273; REACHABLE-HARD +0.0363; beyond-227 null**, with the
effect confined to the hard-reachable cells (easy, mid and beyond-reach all null). Taken at face
value this is the threshold signature: an order-of-magnitude jump once the span exceeds the
effective loop, localized exactly where reach predicts. It is, however, **explicitly labeled
single-seed pending disambiguation**: it lacks a same-pool 48-vs-56 comparison and was measured
against the weakest (nctx6) baseline. Its recorded status is *"threshold HYPOTHESIS —
systematic-or-lucky, undetermined."* The resolution rule was written before the disambiguation
runs, converged with two external reviewers, and is binding; we quote it:

> **R1 (replication):** stride56x4 seed-1 REACHABLE-HARD (48 < dt ≤ 227, seen ≥ 0.5) vs seed-1
> nctx6: CI > 0.
> **R2 (reach causality):** stride56x4 beats contig56 on REACHABLE-HARD, CI > 0 (shared pool,
> seed 0; contig56 = 56 contiguous coarse + 6 fine — exactly the same tokens/slots as stride56x4,
> reach 59 vs 227).
> **R3 (informative, non-gating):** 48-slot seed-1 — flat again (threshold story) vs jumped (luck
> story); reported either way.
> **PASS = R1 AND R2** → run the decisive rollout pair only (d1-56slot vs d0-diff-long on hgrid3,
> WARM = 224, uniform B8×40k); the full Tier-A grid only if the pair's dt-stratified
> reachable-HARD rollout gate is CI-clean green.
> **FAIL of either** → the H-grid closes as the single-step characterization (reach dose-response
> + aliasing ablation), labeled with the seed outcome; no further capability spend.
> **Anything not covered above → default to FAIL** (no soft reads).

**[RESOLVED: PASS — R1 AND R2 passed 2026-07-05; measured numbers below are final. Mechanism
wording revised per external review: the original "attributable to where the slots point"
formulation over-read the seed-0 evidence and is retracted.]**
R1 (replication): the span-224 arm beats its baseline on an independent seed — reachable-HARD
+0.0134 [+0.0107, +0.0161], HARD +0.0089 [+0.0069, +0.0109] — at 2.7× smaller magnitude than
seed 0 (+0.0363 / +0.0273): the effect is replicable but seed-noisy (seed mean ≈ +0.025
reachable). R2 (same-token control): the strided arm beats the contiguous arm with identical
token and slot counts (reach 227 vs 59) at both seeds (differential band +0.0374
[+0.0334, +0.0416] seed 0; +0.0144 [+0.0118, +0.0171] seed 1). The MECHANISM ATTRIBUTION is
seed-dependent and we do not over-read it: at seed 1 the win concentrates in the
reach-differential bands exactly as a per-sample reach-causal account predicts (56-vs-48 shared
+0.0053 vs differential +0.0181; 56-vs-contig shared null −0.0005 vs differential +0.0144), but
at seed 0 it is uniform across shared and differential bands (+0.0343 vs +0.0366) — a
circuit-level rather than per-sample-reach signature. R3: the span-192 arm stays ≈flat at both
seeds (+0.0004; +0.0011) and the contiguous arms never help (both seeds). The reliable two-seed
statement is therefore: **a memory span exceeding the effective loop reliably enables retrieval
learning (2/2 span>loop runs CI-positive vs 0/4 span≤loop runs), while the magnitude and
per-sample usage pattern of the learned circuit vary by seed.** Disclosure: the single-step gate
CIs in this subsection are sample-iid, not clip-clustered; the load-bearing cells survive a
conservative 2–3× design-effect widening (a clip-clustered recomputation is reported alongside).
The pre-registered decisive rollout pair (d1-56slot vs d0-diff-long, WARM = 224, uniform B8×40k)
is thereby justified as the confirmatory experiment, with the teacher-forced caveat stated: the
single-step study conditions on GT context, and the rollout pair is the real test.

*[The pre-drafted FAIL branch (H-grid closes at the single-step characterization if R1 or R2
fails) was removed 2026-07-07: R1 AND R2 passed, so that condition never occurred — the branch
text survives in git history. The study's actual closure came one level later, at the decisive
rollout pair (§4.5.6 below): single-step PASS, rollout RED.]*

### 4.5.5 Methodology lessons (threats-style)

1. **The effective-loop formula.** A memory mechanism's span must be sized against
   $L_{\mathrm{eff}} = \mathrm{world}/(\mathrm{speed}(1-p_{hold}))$, not world/speed: hold frames
   consume slots without advancing the loop. The 48-slot arm's 192-frame span silently fell ~2–10%
   short of its own target (~196–213), plausibly diluting the effect it was built to detect.
2. **The AE-reuse trap.** Reusing a frozen autoencoder across testbed variants forced a background
   transformation (×4 wavenumber scaling) that made the world exactly viewport-periodic and
   destroyed the positional landmarks retrieval depends on — a dataset-side confound that
   manifested as a *mechanism* failure (memory slots scoring as pure attention cost). Rule
   adopted: fresh AE per testbed variant; share one AE only within a campaign.
3. **Evaluation power.** The nb = 24 pre-checks were adequately powered for loopmom-magnitude
   effects (+0.0135) but not for the ~+0.002 effects of the harder regime; the pre-registered 4×
   widening flipped nulls to CI-clean wins on the *same* checkpoints. Reading a null requires a
   power argument against the plausible effect size, not just a CI spanning zero — and both the
   underpowered and widened reads are disclosed.

## 4.6 Chapter conclusions

**Claim 1 — hierarchical memory at matched token budget: CONFIRMED, two seeds, both heads.** At
equal context tokens (1,152), the long-coarse hierarchy beats the flat control on 152-AR-frame
rollouts: seed 0 ALL +0.0189 [+0.0121, +0.0264] with the win concentrated from h ≥ 25 on hard
re-entry bins (h 25–60 HARD +0.0344 [+0.0162, +0.0531]); seed 1 ALL +0.0354 [+0.0267, +0.0447],
winning every cell. The onset boundary is seed-variable; the win and its mid-horizon peak are
not. The result is mechanism-agnostic at two seeds (reg-long > reg-flat, ALL +0.0334 s0 /
+0.0501 s1, every bin on both) —
and honestly bounded: flat-long at 2.7× tokens is the outright frontier (0.1564), so at operating
point A hierarchy is an efficiency point, not a necessity.

**Claim 2 — the generation head: a regime-dependent winner.** Diffusion wins the single-step
oracle-coverage regime on every column at matched capacity and NFE (0.0512 vs 0.0664 ALL);
regression wins the deterministic best-of-1 rollout at equal sharpness (reg-long vs hier-long,
ALL CI [+0.0110, +0.0227]) — the pre-registered fade expectation was refuted, and we report the
reversal as such. Energy-based descent contributes a two-sided negative at reachable scale:
ordering is recipe-dependent (S1 repairs it), reachability fails under both recipes and under the
paper's own sampler (Ch. 3 §3.5). No statement of the form "one best head" survives this chapter.

**Claim 3 — spatial abstraction: economic and head-conditional.** Isolated from the window
confound, coarse history is not an intrinsically better representation: it costs the diffusion
head −0.0082 [−0.0115, −0.0050] at 42% token savings, and is statistically free for the
regression head (+0.0008 [−0.0013, +0.0029]) at the same savings. Its real value is what the
saved tokens buy: window extension at fixed representation is worth −0.027 (G-S3), dwarfing the
abstraction cost.

**Combined optimum.** The cheapest strong configuration measured is **regression head + coarse
history + long window** (reg-long, 0.1682 at 1,152 tokens); at the short window the same recipe
(spat18-reg, 672 tokens, 0.2008) matches the diffusion flat control at 1,152 tokens (0.2038).

**The capability boundary (operating point B) — RESOLVED (2026-07-05/06).** The pre-committed
rule of §4.5.4 resolved in two stages. First, the span-threshold hypothesis was CONFIRMED at the
single-step level (R1 ∧ R2 passed: 2/2 span>loop runs CI-positive vs 0/4 span≤loop). Second, the
thereby-authorized decisive rollout pair returned **RED**: over 256 self-fed frames every
pre-registered gate cell is null, the early-horizon win (h1–24, CI-clean while the slot memory
still carries near-ground-truth content) fades and inverts to small systematic losses beyond
h61 as the buffer fills with self-generated, degraded content — so no capability headline is
claimed, per the binding rule. The closure statement, read together with §4.3: **hierarchical
slot memory pays where its contents remain trustworthy** — continuously refreshed within the
recent window (testbed A) or near a ground-truth anchor (testbed B, h ≤ 24) — and cannot yet be
sustained by pure self-feeding across a 213-frame loop at this scale. The matched-budget result
of §4.3 stands unchanged, with the H-grid as its boundary-condition section.

---

*Draft-only provenance notes (remove before submission).* §4.1 testbed/floors: `docs/draft/CH2_TESTBED.md`
(loopmom refs 0.2367/0.2412/0.4785), `docs/HGRID_DESIGN.md` §1 + addendum item 3 (hgrid3 floors
0.2381/0.0108/0.2084/0.4218, AE fine 0.0024/coarse 0.0059, regime gap 192 / 204 frames / coverage
88.5%); model/protocol: `docs/draft/ARTICLE_SKELETON.md` §4.1/§4.4; cluster-bootstrap rationale
(lag-1 +0.86, 2–7× narrow, 21.5% empty-fg): `docs/draft/VALIDITY_AUDIT.md` MAJOR-2; dt-stratifier:
`docs/HGRID_DESIGN.md` §2.5–6; batch-mean sharp caveat: VALIDITY_AUDIT disclosure 10. §4.2 T1/NFE:
skeleton T1/T1b from `logs_rhcp/step1_mech_clean2.log` (clean.log superseded — do not cite
0.0489/0.0363); reversal + reconciliation: skeleton §4.6/T4 (reg-long vs hier-long direct cluster
CI [+0.0110,+0.0227]); EBT condensation: skeleton T2 + `docs/draft/CH3_MECHANISM.md` §3.5 (S2
21/21 probes, S1 15/15 from 6k, gap +0.22…+3.09, d_tgt 1.12–1.29 vs mean 0.964, sharp max 0.0345,
paper-inference 1.207/0.0127 and 1.233/0.0120; postponed B-cells: HGRID §9). §4.3 T3 emergence +
design-effect CIs: skeleton T3 + VALIDITY_AUDIT MAJOR-2; matrix/headline/curves:
`logs_rhcp/compare_clustered.log` (hier +0.0189 [+0.0121,+0.0264]; flat-long +0.0473
[+0.0362,+0.0592]; reg-long-vs-base +0.0356 [+0.0267,+0.0448]) + skeleton T4/T5 (saturation
0.221–0.222; sharp 0.2023→0.0323; G2b clause not met h≥81); seed-1: `docs/SPATIAL_FACTORIAL.md`
Claim-1 replication block (+0.0354 [+0.0267,+0.0447]; HARD h25-60 +0.0622; sharp 0.088 vs 0.057;
spread 0.1848→0.1690); reg-flat gate: skeleton corrections ledger (+0.0334 [+0.0248,+0.0423];
h1-8 HARD +0.0447). Fig ch4-2 asset: `logs_rhcp/fig_horizon_curves_loopmom.png` (verified
2026-07-05: 3 panels, 7 series, CI bands, batch-mean sharpness label). §4.4: `docs/SPATIAL_FACTORIAL.md`
MEASURED RESULTS verbatim (spat18 0.2120/672 tok, Δ −0.0082 [−0.0115,−0.0050]; spat18-reg 0.2008,
Δ +0.0008 [−0.0013,+0.0029], h61-120 HARD +0.0047; G-S2 ~0.007–0.008/480 tok; G-S3 −0.027);
prior spatial-gate null: project memory `per-pixel-spatial-dlh-result`. §4.5: `docs/HGRID_DESIGN.md`
addenda (hgrid 20k HARD −0.0008 [−0.0027,+0.0012], 40k HARD −0.0025 [−0.0042,−0.0008] / ALL
−0.0012 / dt>192 −0.0026; crop MSE 0.000000; hgrid3 crop-uniqueness 0.004–0.010; widened eval
strided HARD +0.0016 [+0.0007,+0.0025], REACHABLE +0.0021 [+0.0008,+0.0036], dt49-96 +0.0032,
ALL +0.0005; contig48 null; ~7× / +0.002-vs-+0.0135 / 2%-vs-13%; span-224 HARD +0.0273,
REACHABLE +0.0363, beyond-227 null; R1/R2/R3 quoted from the PRE-COMMITTED block; rollout-CI
resolution ±0.005–0.02). Branch placeholders RESOLVED 2026-07-05/06: disambiguation = PASS
(FAIL branch removed), decisive pair = RED (§4.5.6); §4.6 carries the resolved conclusion.

### §4.5.6 The decisive rollout pair — closure (2026-07-06; the rollout-level conclusion)

The pre-registered decisive pair returned RED: over 256 self-fed rollout frames the gate cells
are null (dt 49–96 HARD +0.0020 [−0.0071,+0.0111]; dt 97–192 −0.0028 [−0.0088,+0.0035]; dt>192
−0.0018 [−0.0065,+0.0027]), so the single-step capability does NOT convert to a rollout
capability at this scale, and no further capability experiments were run (binding rule). The
horizon profile is the finding: d1-56 wins every h1–24 bin CI-clean (HARD +0.0211/+0.0224)
while its slot memory still carries near-ground-truth content, fades to null by h25–60, and
inverts to small systematic losses beyond h61 as the buffer fills with self-generated,
degraded content. Read together with §4.3: hierarchical slot memory pays where its contents
remain trustworthy — continuously refreshed within the recent window (testbed A, matched-budget
win over 152 frames) or near a ground-truth anchor (testbed B, h≤24) — and cannot yet be
sustained by pure self-feeding across a 213-frame loop. This bounds the mechanism honestly and
completes the boundary-condition section; the §4.3 matched-budget result is unaffected.

---

### §4.3.x Token-budget frontier [MEASURED 2026-07-07] (integrate as §4.3.6 on next pass)

**Design (completed frontier — every flat point gets a same-budget hierarchical partner; author
request 2026-07-06, pre-registered in `docs/SPATIAL_FACTORIAL.md` FRONTIER COMPLETION block,
commit 220b44f, arms registered in `hvp_rev/rhvd.py`).** Table 4.2 left the frontier's two ends
unpaired: flat-long (3,072 tokens) and hier-short (480 tokens) had no budget-matched partners.
Two new arms complete the grid. At the top, **hier-rich** (48 coarse + 36 fine = 3,072 context
tokens) vs flat-long — trained at B = 8 × 60k, sample-matched to flat-long's disclosed regime
*by design*, so the §4.3.4 EMA-horizon/gradient-noise caveat cancels within this pair (both arms
share it). At the bottom, **flat-short** (8 fine frames = 512 tokens) vs hier-short (480 tokens),
B = 16 × 30k; the 32-token (≈6.7%) budget mismatch favors flat-short and is disclosed. Diffusion
heads first (matching the existing diffusion frontier curve); regression twins (hier-rich-reg,
flat-short-reg) are registered but not launched. Protocol otherwise identical to the §4.3
campaign (seed 0, loopmom, WARM = 48, 152 AR frames, 100 clips, cluster CIs, empty-fg excluded),
so all records pair exactly with the existing npz set. Measured runtime reference: the
3,392-token hier-rich sequence runs ≈0.55 s/step ⇒ ≈9.2 h train + eval.

**Pre-registered reading (quoted verbatim from `docs/SPATIAL_FACTORIAL.md`, written before any
result):**

> "PRE-REGISTERED READING: at 3072 tokens the window is SATURATED at 48 frames by the T=100
> train clips — hierarchy cannot buy more window, so per the factorial's abstraction cost the
> honest expectation is hier-rich ~<= flat-long (CONVERGENCE at the frontier top). The
> efficiency claim's correct form is 'the hierarchy curve reaches a given quality at fewer
> tokens and converges with flat where the window saturates' — a frontier-from-below, not
> dominance at every budget. At the low end, hier-short vs flat-short tests the frontier's
> bottom."

**Table 4.x — Frontier completion (MEASURED 2026-07-07; fills from
`rhvd_eval_{hier-rich,flat-short}_s0.npz` via `logs_rhcp/frontier_top.log` /
`logs_rhcp/frontier_bottom.log`).** Measured partners, for reference: flat-long 0.1564 /
sharp 0.0996 (3,072 tok; B8×60k); hier-short 0.2260 / 0.0392 (480 tok) — both from Table 4.2.

| frontier read | arm(s) | ALL fg-MSE | fg-sharp | paired Δ (flat − hier) | 95% CI |
|---|---|---:|---:|---:|---|
| top, new arm | hier-rich (48c+36f, 3072 tok, B8×60k) | 0.1695 | 0.0898 | — | — |
| bottom, new arm | flat-short (8f, 512 tok, B16×30k) | 0.2224 | 0.0432 | — | — |
| lower edge, new arm | hier-min (48c+2f, 896 tok, B16×30k) | 0.1706 | 0.0861 | — | — |
| lower edge, new arm | flat-min (14f, 896 tok, B16×30k) | 0.2106 | 0.0516 | — | — |
| frontier top | flat-long vs hier-rich (equal 3,072 tok, equal regime) | — | — | −0.0131 | [−0.0201, −0.0069] **flat WINS** (REPLICATED seed 1: −0.0107 [−0.0163, −0.0058] — two-seed) |
| frontier bottom | flat-short vs hier-short (512 vs 480 tok) | — | — | −0.0036 | [−0.0063, −0.0010] **flat WINS** |
| frontier lower edge | flat-min vs hier-min (equal 896 tok, equal regime) | — | — | **+0.0399** | [+0.0299, +0.0501] **HIER WINS** (every bin; REPLICATED seed 1: +0.0358 [+0.0267, +0.0454], every bin — two-seed) |

[FIG `docs/figures/fig9_frontier.png`: quality-vs-token-budget (log-x), two measured 4-point
curves — hier (480 → 896 → 1152 → 3072: 0.2260, **0.1706**, 0.1848, 0.1695) and flat
(512 → 896 → 1152 → 3072: 0.2224, 0.2106, 0.2038, 0.1564) — measured shape: hierarchy wins the
band [896, ≈2059) tokens (upper crossing interpolated from the curves), with the effect PEAKING
at the lower edge, which sits exactly at the geometric span-threshold prediction (48×16 coarse
tokens + minimal fine tail ≈ 830–896); flat wins below the threshold (512) and at window
saturation (3072). Caption: the token-budget frontier — hierarchy wins exactly the band where
its coarse span crosses the memory horizon while flat's window does not.]

**Reading (measured resolution of the pre-registered dual branch — a third outcome; reported
under the no-soft-reads rule, the §4.5.4 default-to-FAIL discipline applied to wording).**
Neither pre-registered branch survived: hier-rich neither converges with nor beats flat-long —
it **LOSES CI-clean** at equal budget and equal regime (−0.0131 [−0.0201, −0.0069], a small
uniform loss across essentially every horizon × hardness bin in `frontier_top.log`), and at the
bottom **flat-short beats hier-short** (−0.0036 [−0.0063, −0.0010], concentrated at h ≤ 60 and
fading to null beyond h 61). This is *stricter* than the pre-registered convergence expectation
and is reported as measured. The mechanism reading is consistent with the factorial: at the
top, the 48-frame window is saturated by the T = 100 train clips, so the coarse slots can no
longer buy window — they merely displace full-resolution frames and pay the diffusion-head
abstraction cost (the −0.0131 loss is of the same kind as G-S1's −0.0082 at W18); at the
bottom, the 6–8-frame window is below the memory-relevance scale, so coarse slots are
redundancy and flat's extra fine frames win. The 1,152-token headline (§4.3.3, two seeds, both
heads) is untouched. The claim of §4.3 therefore takes its final, bounded form: **hierarchy
dominates the token-budget frontier exactly where window extension is both possible and
valuable** — a bounded, dose-response-supported efficiency claim, not dominance at every
budget. Sharpness, same discipline: hier-rich 0.0898 vs flat-long 0.0996 and hier-short 0.0392
vs flat-short 0.0432 — the window (token budget spent on frames) drives sharpness, and
hierarchy does NOT retain sharpness better at matched saturation.

**Lower edge (measured 2026-07-07; pre-registered the same morning, before the runs, in
`docs/SPATIAL_FACTORIAL.md` FRONTIER LOWER EDGE — the primary branch measured).** The band
theory makes a quantitative prediction: the win should switch on at the cheapest budget whose
coarse span covers the ~48-frame loop, i.e. 48×16 coarse tokens plus a minimal fine tail ≈ 896
tokens — a budget at which flat affords only a 14-frame window. The measured pair confirms it,
and more strongly than predicted: **hier-min (48c+2f) beats flat-min (14f) in every horizon ×
hardness cell**, ALL **+0.0399 [+0.0299, +0.0501]** — the largest hierarchy delta of the
campaign (vs +0.0189/+0.0354 at 1,152) — with the dt-stratified gates showing the reach
signature exactly where the mechanism lives (dt ≤ 48 HARD +0.0750 WIN; dt 49–96 HARD +0.0282
WIN; dt 97–192, beyond the 48-frame span, null). Two structural readings follow. First, the
win band's **lower edge is at the geometric prediction**: flat wins at 512, hierarchy wins at
896 — the switch-on is bracketed to (512, 896] and coincides with the span-threshold arithmetic
written before the run. Second, the **effect peaks at the lower edge**: moving down-budget
inside the band, flat's window falls away from the horizon faster than the hierarchy's
abstraction cost grows, so the efficiency argument is strongest exactly where budgets are
tightest — hier-min at 896 tokens (0.1706) reaches a quality that the flat curve only attains
between 1,152 and 3,072 tokens (interpolated ≈2.3–3× the budget; the measured upper crossing of
the curves sits at ≈2,059 tokens). The fine-context question resolved with the seeds: hier-min's
seed-mean (0.1725) sits within hier-long's seed band (0.1690–0.1848, mean 0.1769) — the
pre-registered "fine-context floor" alternative is refuted at w_f = 2 (two seeds), and the hier
curve from 896 tokens on reads as a plateau at ≈0.17 rather than a dip. Disclosures resolved: the pre-registered seed-1 replication (R-A, `frontier_low_s1.log`)
REPLICATED the result — every cell WIN again, ALL +0.0358 [+0.0267, +0.0454], the same dt-gate
reach signature (dt≤48 HARD +0.0666; beyond-span null) — so the 896-token point is TWO-SEED
CONFIRMED (seed-mean Δ ≈ +0.0379, the campaign's largest two-seed effect; hier-min seed spread
0.0038, flat-min 0.0004). Both the 896 and 1,152 budgets now carry two seeds. The final claim form is unchanged but now edge-measured: hierarchy
wins **the band of budgets where its span crosses the memory horizon and flat's window does
not** — [896, ≈2,059) tokens as measured on this testbed — and inverts outside it.

*Provenance (this section only): design/arms `docs/SPATIAL_FACTORIAL.md` FRONTIER COMPLETION +
FRONTIER MEASURED (2026-07-07) + FRONTIER LOWER EDGE (pre-reg + MEASURED same day, commit
441076e; chain marker FRONTIER_LOW_DONE) + `docs/HANDOFF_2026-07-06.md` §6 (launch 08:54:46
2026-07-06; chain: hier-rich 60k → flat-short 30k → compare flat-long,hier-rich → compare
flat-short,hier-short; FRONTIER_DONE sentinel); arm tuples `hvp_rev/rhvd.py` (hier-rich
(48,36,diff), flat-short (0,8,diff) + reg twins; hier-min (48,2,diff), flat-min (0,14,diff));
per-bin rows `logs_rhcp/frontier_top.log` / `frontier_bottom.log` / `frontier_low.log`; measured
partner rows from Table 4.2 (`logs_rhcp/compare_clustered.log`). On the integration pass: move
the new rows into Table 4.2 and renumber this section to §4.3.6 (branch already resolved
above).*
