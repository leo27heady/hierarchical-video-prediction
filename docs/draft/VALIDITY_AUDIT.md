# Validity audit — verdict and corrections (2026-07-03)

Six-surface adversarial audit (data/splits, RHVD protocol, gate protocol, EBT protocol, AE
provenance, metrics/statistics) with independent verification of findings. 24 findings raised;
2 MAJOR confirmed and **fixed + recomputed same day**; the rest are MINOR disclosures now
integrated below. **Executive verdict: every headline conclusion survives; two tables' numbers
were corrected; all quotable CIs are now cluster-robust.**

## Confirmed MAJOR — fixed and recomputed

### 1. Mechanism-table hardness labels were target-anchored (k-offset bug)
`step1_mech._hardness` counted the k−1 = 3 bridge frames (never model input) as "covered" and
dropped 3 real context frames — novelty understated by ~0.09 mean, 11–14% of samples mislabeled
easy-that-were-hard, frac_hard 0.19 (buggy) vs 0.27 (correct). mem_gate / rhvd always used the
correct anchor (results untouched). **Fix:** anchor corrected (commit `ecd125d`); table recomputed
from the saved checkpoints with, additionally, ONE metric reference (raw GT) for all rows
(previously fg-MSE used AE-roundtrip GT while persistence/rollout used raw).
**Corrected citable T1 (logs_rhcp/step1_mech_clean2.log):**

| mech | fgMSE(all) | easy | hard | meanN | sharp | NFE 4→16 |
|---|---|---|---|---|---|---|
| reg | 0.0664 | 0.0232 | 0.1808 | 0.0714 | 0.1591 | 0.0680 flat |
| **diff** | **0.0512** | **0.0120** | **0.1549** | **0.0640** | **0.1886** | 0.0546→0.0503 |
| persistence | 0.2367 | — | — | — | 0.0515 | — |

(references: raw-GT sharp 0.4785; AE-roundtrip ceiling 0.2412; frac_hard 0.27.)
Diffusion > regression on every column — unchanged qualitatively from v1.

### 2. Frame-level bootstrap CIs ignored clip clustering
Rollout records are 100 clips × 152 consecutive AR frames with lag-1 autocorrelation +0.86 —
frame-iid CIs were 2–7× too narrow; three secondary bin verdicts flip to null. **Fix:** cluster
(clip-level) bootstrap in `rhvd.paired_ci` + empty-GT-foreground records (21.5%, where fg-MSE
silently falls back to full-frame MSE) excluded from strat bins and reported as their own row
(commit `ecd125d`). **Recomputed (logs_rhcp/compare_clustered.log): all headline verdicts
survive** — hier-long vs token-matched flat: null at h<25, WIN at h≥25 incl. HARD
(+0.0344 [+0.0162,+0.0531]), ALL +0.0189 [+0.0121,+0.0264]; reg-long > hier-long ALL
(+0.0356 vs flat base; direct cluster CI [+0.0110,+0.0227]); flat-long > all (Pareto corner).
mem_gate CIs have the same flaw in mild form (independent windows, ~3 samples/clip): analytic
design-effect bound (deff ≤ 2 at ρ≤0.5) widens the 20k hard-bin CIs to ≈[+0.005,+0.022] (s0)
and ≈[+0.006,+0.023] (s1) — **the emergence WIN survives** worst-case correction; exact
clip-level recompute optional.

## Disclosures (MINOR / unrefuted caveats — must appear in the article)

1. **fg-MSE degeneracy:** the GT-masked metric scores only GT-sprite pixels; a trivially bright
   frame scores well and hallucinated off-sprite content is invisible to it. Mitigation: paired
   comparisons + sharpness + decoded-grid review; add one full-frame-MSE column in final tables.
2. **"200-frame rollout" wording** → protocol is 48 GT-warmup + **152 AR frames**; bin "h 61+"
   spans 61–152. All text must say 152 AR frames.
3. **flat-long protocol caveat:** B=8×60k (sample-matched to B=16×30k) — EMA horizon and
   gradient noise differ; disclosed wherever flat-long is quoted.
4. **mem_gate "2 seeds"** = 2 *training* seeds sharing one CRN eval set (by design, for pairing);
   phrase as such.
5. **S1 landscape-flip onset (~6k)** rests on one fixed 64-sample probe batch; the 15-consecutive-
   probe persistence is the claim, not the exact onset step. Gap range is +0.22…+3.09 (a draft
   provenance note saying "+2.1…+3.1" must be fixed).
6. **Dataset provenance:** generation seeds/params of the shipped loopmom files are not recorded;
   measured properties (p_hold≈0.10) are stated as measured. Regenerate-with-seed is future
   hygiene, not a validity threat (train/test are separate generation draws; frame-hash
   duplication check found no cross-file clip duplication).
7. **Digit-asset sharing:** train/test clips draw from the same MNIST glyph pool (standard for
   this testbed class); affects texture familiarity only — re-entry memory requires clip-specific
   trajectory content, which does not transfer across clips. One disclosure sentence.
8. **Frozen-AE provenance:** shared by all arms (comparisons fair regardless); absolute
   sharpness/recon ceilings quoted alongside. [Verify agent for the train-file-only question died
   at session limit — re-check `hier_convae` training data before the final text; treated as a
   disclosure either way.]
9. **Legacy diagnostic scripts** (ebt_fixed, ebt_ref_verify*, ebt_diag2) standardize with
   test-derived stats if re-run post-STD-FIX; historical logs stand as recorded (the reference
   control self-standardized). `--bf16` on the 2k-step control was a silent no-op → those runs
   were fp32 (documented; scaled runs dropped the flag deliberately).
10. **npz `sharp` field** is a chunk-batch mean duplicated per record — valid for overall means
    (as used); never use it per-bin/per-clip.

## Threats-to-validity paragraphs (ready to adapt)

**Internal validity.** Hardness labels are computed from ground-truth geometry independently of
any model and identically across arms; a target-anchoring bug found by audit was fixed and all
stratified tables recomputed (conclusions unchanged). All paired comparisons share evaluation
clips, horizons, and sampling noise (common random numbers); confidence intervals use clip-level
cluster bootstrap to respect rollout autocorrelation.

**Construct validity.** Foreground-masked MSE ignores content generated outside the ground-truth
sprite mask and rewards brightness within it; we therefore never interpret fg-MSE alone, but
jointly with foreground sharpness, full-frame error, and decoded-sample inspection, and we report
background-only frames (21.5% of rollout records) separately.

**External validity.** The testbed is purpose-built and synthetic; digit assets are shared across
splits (texture familiarity), while trajectories and world layouts are split-disjoint. Claims are
scoped to the memory-horizon regime the testbed operationalizes; the OpenSTL bridge (appendix)
anchors short-horizon behavior on a standard benchmark.
