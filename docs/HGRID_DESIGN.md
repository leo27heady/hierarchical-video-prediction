# H-grid pre-registration v2 (FROZEN 2026-07-03) — depth × mechanism, capability regime

**Author-approved path:** full Tier A (5 arms, regression + diffusion = gates G-H1/G-H2/G-H3) +
mandatory seed-1 on the decisive pair. **ALL EBT cells POSTPONED** (spec in §9 — the settled
two-sided EBT study T2 on testbed A carries the article/dissertation EBT chapter unchanged).
**d2 (third level) DEFERRED**, gated on a d1 capability signal; its GPU hours fund the
emergence-extension reserve. **Model size PINNED at 22M (d384/8L) for every arm.** Supersedes v1
(which remains in git history). Everything below is pre-registered BEFORE any code or launch.

Goal: convert the hierarchy claim from *efficiency at matched token budget* (proven on testbed A,
T4, both mechanisms) into *capability* — a regime where NO affordable flat window covers the
memory horizon, so temporal abstraction (a strided, slow-rate coarse level) is necessary, not just
cheaper. Fixes both verified gaps of the external diagnosis (HANDOFF_2026-07-03 §6): the missing
mechanism (temporal stride + coarse-slot-per-window generation) and the regime cap
(loop ≤ flat window).

## 1. Regime — testbed operating point B (do this FIRST)

Generator changes (`scripts/gen_looping.py`, ~4 LOC):
- scale background wavenumbers `kx` by `world/192` (keeps integer wavenumbers → seamless torus
  wrap; without this the per-frame background regime silently changes and the AE-reuse premise
  breaks);
- print measured per-frame digit density.

Generation parameters (RECORD SEEDS — train and test; loopmom's absence of seeds is a documented
validity wart):
- `--world 768 --frames 480 --digits 12 --speed 4 --p_cont ~0.97 --p_hold 0.1`
- `--digits 12` keeps per-viewport digit density identical to loopmom (3·92/192 ≈ 12·92/768);
- `p_cont ~0.97–0.98` (NOT 0.9): at world 768 a p_cont=0.9 walk is diffusive (mean run ≈ 36 px)
  and re-entry would blow past the target into "never revisited = unknowable" — the generator's own
  documented failure mode. Near-ballistic circulation targets loop ≈ 192–240 frames. Tune p_cont
  against the verification below (1–2 quick iterations expected).
- counts: train 1200 clips, test 300 (host-RAM bound: ~4.1 GB train latents; do NOT use 3000).

**Two-sided regime verification (MUST PASS before any training):** measure via `rel_positions`
(W=768, speed=4):
1. median re-entry interval ∈ [192, ~280] (lower bound = capability regime vs the 48-frame flat
   window; upper bound = re-entry actually happens inside the rollout span);
2. minimum count of beyond-48-frame re-entry events per clip within the rollout span (starvation
   guard for the gate bins);
3. measured warmup world-coverage (fraction of world seen in frames 0..191) — reported, and used
   for the warmup-SEEN stratification in §6;
4. record the full re-entry-interval distribution + off-screen fraction + p_hold check.

**Floors re-measured on the NEW test split** (minutes of GPU): persistence fg-MSE, AE round-trip
ceiling, raw-GT sharpness. The loopmom floors (0.2367 / 0.2412 / 0.4785) DO NOT transfer.

**AE-reuse gate:** encode/decode ~256 new-train frames with frozen `logs_rhcp/hier_ae_mom.pt`;
accept if fine recon MSE ≤ ~2× the 0.0020 loopmom baseline; else retrain (cheap: 7k steps, <1h).
Standardization stats are recomputed from the NEW train split by `ActionHierData` (automatic),
test uses train stats as always.

## 2. Correctness fixes required BEFORE any ≥20k launch (silent-wrong-numbers list)

1. `rhvd.py` main: thread `W=768, speed=4` into every `ActionHierData` (default W=192 would wrap
   all hardness labels mod 192 → corrupt-but-plausible stratification across the whole campaign).
2. `_fg_empty_mask` (rhvd.py:436): parametrize test path + warm (currently hardcoded to
   `data/loopmom_test_500x200.npy`, warm=48 — would silently score the new grid against OLD
   dataset frames).
3. `WARM = 192` universally (all arms; = one full loop seen in GT warmup; keeps (clip,horizon)
   pairing valid in `compare()`).
4. `hbins` extended to cover h ≤ 288 (currently caps at 200; re-entry mass sits at h≈150–250).
5. **dt-since-last-visible stratifier** (eval-side, ~2–3h + unit test): per-record
   Δt-since-region-last-visible from `rel_positions`; pre-registered bins dt ∈ (48,96], (96,192],
   >192, crossed with the existing novel-fg-6 hardness. THE GATE VARIABLE — G-H1/2/3 are read on
   dt>48 × HARD cells. (Computable at eval time from clip ids + horizons; does not block training
   launches.)
6. Warmup-SEEN stratification: gates read ONLY on re-entries of content visible during warmup
   (answerable-by-construction); unseen-content rows reported separately.
7. No-stale-literal check: grep the grid code path for `loopmom` literals before launch.

**GO gate (must pass before Tier A):** (a) layout unit test — token counts, frame ids, stride
phase asserts (tj % 4 == 0), due/not-due windows; (b) GT-teacher-forced sanity rollout on the new
dataset — fg-MSE must land at the re-measured AE ceiling AND the hardness histogram must match the
generator-side verification script's.

## 3. Single-step pre-check — the kill-switch (~4h GPU, FIRST GPU job)

Reuse the `mem_gate` harness + ~15 LOC stride in its batcher: long-coarse-STRIDED (48 slots @
stride 4 = 192-frame span) vs short baseline, 7.5M single-step model, 20k steps, on the new
dataset. GREEN = beyond-window retrieval emerges single-step → proceed. RED (no emergence at 20k
single-step) = STOP; re-scope budgets/regime before burning ~25h of rollout arms. Rationale: the
only emergence evidence (6k null → 20k win) is from an easier regime; this is the cheapest test of
whether the knee moved out of reach.

## 4. Arms (Tier A — ALL at 22M d384/8L, uniform B=8 × 40k steps, EMA 0.999, ckpt/resume, spill guards)

Uniform optimizer regime B8×40k (sample-matched to B16×20k; single LR/EMA/warmup schedule across
the grid — kills the optimizer-regime confound on the headline; VRAM-safe at every seq length;
disclosed once, globally).

| arm | context | ctx tokens | seq | mech | role |
|---|---|---|---|---|---|
| d0-diff-long | 48 consecutive fine frames | 3072 | ~3392 | diff | flat frontier baseline |
| d0-reg-long | same | 3072 | ~3392 | reg | flat baseline, reg |
| d1-diff | 48 coarse slots @ stride 4 (span 192) + 6 fine | 1152 | ~1472 | diff | THE hierarchy arm |
| d1-reg | same | 1152 | ~1472 | reg | mechanism-agnosticism |
| d0-strided-flat (HYBRID control) | 44 full-res slots @ stride 4 (span ~176) + 6 consecutive fine | ~3200 | ~3520 | diff | abstraction vs subsampling |

- d1 design: with stride 4 and WIN=4, each window step predicts exactly ONE coarse slot (frame
  anchor = last frame of the 4-block) + 4 fine frames; coarse ring buffer (48 slots) appends the
  generated slot every macro-step. Train tj sampled ON the 4-grid; phase-locked to rollout.
  Coarse slot = per-frame coarse encoding of every 4th frame (pure data-layer subsampling; NO AE
  change).
- HYBRID control (replaces v1's naive strided-flat): mirrors d1's structure at full resolution —
  strided history + consecutive recent fine — so recency density is held equal and the contrast
  isolates "coarse abstraction vs full-res temporal subsampling". Pre-registered disclosures:
  token asymmetry favors the CONTROL (~2.2×; a d1 win is conservative); diagnostic — if the
  control trails d1 already in easy/h1–8 bins (where memory is irrelevant), it is declared
  recency-confounded and G-H3 is read accordingly.
- Token accounting is reported per arm together with param counts (frame_embedding grows to ~196
  entries for strided arms; params not byte-identical across arms — disclosed).
- 3392+-token arms: B=8 measured 0.55 s/step @ 7.38 GB (safe). NEVER pair two 3392-token jobs
  (14.76 GB = spill roulette). Concurrent training buys no throughput (measured ~0.95× serial);
  concurrency is used ONLY to overlap short evals with a running train job.

## 5. Rollout evaluation

WARM=192 GT warmup → AR rollout to h ≤ 288 (T=480; 288 = 1.5 loops, covers a full beyond-window
re-entry cycle; pre-registered — "h≥400" from v1 was arithmetically impossible at T=480). 100 eval
clips; NFE=16 diffusion; cluster (clip-level) bootstrap CIs; empty-fg exclusion; metrics: fg-MSE
(+ full-frame-MSE column per the audit disclosure), sharpness-vs-h, per-horizon curves, all
stratified by dt-since-visible × hardness × warmup-seen.
Probes-while-training at 10k/20k/30k on a reduced protocol (25 clips, gate bins only), overlapped
with the running train job (~1h total, budgeted).

## 6. Gates (pre-registered, read on dt>48 × HARD × warmup-SEEN cells, cluster CI > 0)

- **G-H1 (capability):** d1-diff beats d0-diff-long beyond the flat window. Passing = temporal
  abstraction delivers what the best affordable flat window cannot.
- **G-H2 (mechanism-agnostic):** d1-reg beats d0-reg-long, same cells.
- **G-H3 (abstraction vs subsampling):** d1-diff beats the HYBRID control, same cells.
- **Emergence extension rule (FROZEN):** if at 40k a gate's delta is positive with CI spanning 0
  AND monotonically growing across the 10k/20k/30k/40k probes AND train loss not plateaued →
  verdict "UNRESOLVED", decisive pair extends to 60k (B8; sample-matched 30k). Flat trend +
  plateaued loss → honest null. Reserve funded by the d2 cut.
- **Seed policy:** grid-wins branch REQUIRES seed-1 replication of the decisive pair (d1-diff +
  its strongest flat competitor, whichever of d0-diff-long / hybrid is closer) before the
  capability headline is claimed; single-seed results are labeled preliminary.
- Any null → the audited testbed-A matched-budget headline stands ("When Does Hierarchy Pay?");
  the grid is reported as measured boundary conditions. No empty branch.

## 7. Schedule (measured-cost basis; ~43 GPU-h ≈ 1.8–2.0 GPU-days; ~3–3.5 calendar days)

- **Day 1:** generator tweaks + generation + two-sided regime verification + floors + AE gate
  (CPU-heavy, ~0.8h GPU) → eval-path correctness fixes (§2 items 1–4; d0 arms need NO stride
  code) → **pre-check** (§3, ~4h GPU) → if GREEN, launch **d0-diff-long** overnight (6.9h).
- **Day 2:** stride build (window_batch / WindowDiT / rollout scheduler + ARMS + phase asserts,
  ~150–200 LOC) + dt-stratifier + layout unit test + GO gate. GPU by day: **d0-reg-long** (6.2h);
  evening launch **d1-diff + d1-reg + hybrid** (2.6 + 2.2 + 7.2h) overnight.
- **Day 3:** evals + gate reads (G-H1/G-H2/G-H3, seed 0) → grid-wins branch: launch **seed-1
  decisive pair** (~9.5h) overnight. Writing track starts (P0-doc debt, Ch.4 skeleton).
- **Day 4 (morning):** seed-1 lands → final grid tables with cluster CIs.
- Reserve: 2–3h (probe-steps VRAM checks before each new shape, spill restarts, WDDM jitter)
  + the emergence-extension reserve (+2h d1 / +6h d0 per extended arm) from the d2 cut.

## 8. Deferred: d2 (third level)

Runs ONLY if d1 shows a capability signal. Spec: 3-level AE (`HierConvAE3`, new class — do not
change `HierConvAE.encode`'s 2-output signature; top 16ch@2×2; 3-way random-level dropout;
~50–70 LOC + ~1h GPU retrain on the NEW train file) + top level at stride 16 (12 slots span 192;
top-due windows at (tj+4) ≡ 0 mod 16; WARM=192 keeps 4- and 16-phase alignment consistent) +
d2-diff arm (~1520 tok, ~2.7h). AE-confound rule (pre-registered): either ALL arms rerun on the
shared 3-level AE (comparable fine-recon verified) or the d2 comparison is explicitly labeled as
carrying an AE confound. G-H4 (does depth 2 pay at the 192-frame horizon?) deferred with it;
dissertation text stays depth-parameterized (L ∈ {1,2,3}) with d2 marked specified-not-run.

## 9. POSTPONED: the EBT cells (author decision 2026-07-03 — too slow for this window)

What is postponed (NOT cancelled), for a later ~5h GPU slot:
- **Two-cell EBT characterization row: d0-ebt + d1-ebt**, run in the `step1_mech` single-step
  harness (the SAME EBT class whose collapse signature was cross-validated against the official
  implementation at fair scale in T2 — NOT a new port into WindowDiT, which would reopen the
  "your port, not EBT" objection T2's reference control was built to kill). Stride added only to
  its data layer. 8–10k steps each, S1 (ordering-correct) recipe, dedicated high-LR α group.
- **Three pre-registered readouts per cell:** (a) energy-landscape ordering probes
  (E(target) vs E(mean), the T2 methodology); (b) descent reachability — fg-sharpness of descent
  output vs the collapse threshold; (c) single-step best-of-8 fg-MSE against the reg/diff cells.
- **Pre-registered expectation:** the collapse signature transfers to operating point B. The
  scientifically interesting alternative: the strided coarse level's lower-entropy targets make
  the per-level energy landscape better-ordered/reachable (the original "hierarchy helps EBT"
  hypothesis) — if (a)/(b) improve at d1 vs d0, THAT is a new finding and would justify a deeper
  EBT revisit.
- Protocol footnote in the grid table: single-step, reduced budget, fidelity anchored to T2.
- Explicitly rejected forever: full EBT rollout-parity arms (~20h+ GPU to measure collapse
  propagation — adds nothing over T2) and reversibility-on-EBT-inner-loop (measured closed:
  0.26% saving vs checkpointing, `logs_rhcp/mem_panel.md`).
- Article/dissertation impact of postponement: ZERO on the existing EBT chapter — §4.2 / T2
  (two-sided S1/S2 study, official-code control, 20k converged steps) is complete on testbed A
  and carries the EBT content in full. The postponed row would only ADD a measured third head to
  the grid table if run later.

## 10. Integration pointers (article + dissertation)

- Article: testbed family (operating points A = efficiency regime, B = capability regime) in
  Ch.2; mechanism study + EBT two-sided study (complete) in Ch.3; Ch.4 = matched-budget efficiency
  block (testbed A, done) + capability grid (testbed B, this memo); reversibility closes.
  Title branches: capability → "Hierarchy Earns Its Keep Where the Memory Horizon Demands It";
  null → "When Does Hierarchy Pay? A Memory-Horizon Study at Matched Token Budgets".
- Dissertation: Розділ 3 model section depth-parameterized (z^(ℓ) at geometric stride s^ℓ, s=4);
  §3.1.3 joint-generation claim re-scoped to within-rate (across rates: coarse slot precedes its
  fine window); §3.4 testbed as a two-operating-point family; Розділ 4 §4.4 = the grid; novelty
  item 1 uses the two-branch wording (see DISS_DIGEST + rewrite files); digest Q2 record gets the
  "architecture generalizes to the depth-parameterized family" note (author signed off by
  approving this memo).
