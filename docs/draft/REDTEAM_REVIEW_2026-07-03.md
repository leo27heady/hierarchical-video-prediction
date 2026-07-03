# Red-team review — final article + dissertation (2026-07-03)

Method: 11 read-only, CPU-only reviewers (code correctness, data leakage, statistics, EBT
faithfulness, reversibility accounting, narrative framing, dissertation coherence, planned
actions) + adversarial verification of every blocker/major finding, + independent numpy
re-derivation of the load-bearing numbers by the lead. No GPU used. 47 agents, ~3.2M tokens.
Raw findings: workflow `wulckjr70`. This supersedes nothing in `VALIDITY_AUDIT.md`; it goes
deeper on the science and the write-up.

Tally after verification: 0 surviving blockers, **26 major**, 34 minor, 10 reassurances,
11 strengths. The 10 raw "blockers" were all **downgraded to major** by the skeptic pass —
meaning: nothing fabricated, no bug that silently flips a computed result, but several places
where **the write-up outruns the evidence** badly enough to get hit at defense.

---

## Verdict in one paragraph

**The machinery is clean; the story is over-sold.** Every correctness check that could have
sunk the work passed: no data leakage (AE trained on train split only — the audit's one open
item, now closed), standardization is train-stats-only and leak-free, rollout buffers are pure
(no GT re-injected after warmup), the cluster bootstrap is correctly specified, common-random-
numbers pairing is exact, the EBT port is faithful to the official S1/S2 recipes line-for-line,
and the hardness anchor is correct. **What fails is the framing**, and it fails in two specific,
verified, and currently-hidden ways: (1) at matched token budget **regression beats the
diffusion "system" at rollout** at every horizon and is *not* blurrier, and (2) the **single
best model is flat**, not hierarchical. Both facts live only in the bottom-of-file corrections
ledger; the abstract, title, and contributions still read as if hierarchical diffusion won.
Plus the memory-emergence headline has an undisclosed confound, and the drafts carry stale
pre-fix numbers. None of this is fatal — there is a genuinely defensible, narrower article
underneath — but it has to be reframed before "headline" appears in print.

---

## What's solid — do NOT spend effort re-checking these

These were suspected and **verified clean** (reassurances) or are genuine strengths:

- **No data leakage.** Frozen AE `hier_ae_mom.pt` trained on `loopmom_train_3000x100.npy`
  only; the 500-clip test file was never loaded. Train/test are disjoint draws, **0 clip
  duplications**. (Closes audit disclosure #8.)
- **Standardization is correct.** Stats computed on train only; test `ActionHierData` receives
  `stats=data.stats`; `decode01` un-standardizes with the matching train constants. No
  test-stat leakage on either latent level.
- **Rollout protocol is honest.** Buffer purity holds (universal WARM=48 pairs (clip,horizon)
  exactly across arms), CRN genuinely shares noise across diffusion arms, empty-foreground
  exclusion (21.5%) is correct, the degenerate per-record `sharp` field is only ever used as an
  unbiased overall mean.
- **Statistics core is sound.** The clip-level cluster bootstrap is a legitimate estimator with
  healthy cluster counts (95–100 clips/bin); the audit's MAJOR-2 fix genuinely holds; the
  headline hier>flat-budget horizon-gated win reproduces to the digit.
- **EBT port is faithful.** Official `EBTTimeConcat` + `CausalReplayBuffer` imported unchanged;
  S1/S2 flags, α, and lr-multipliers match `ebt_s{1,2}_vae.sh` and the paper's hparam table.
  The scope section (§3.5.5) correctly refuses the strawman (the paper's video §5.2 claims
  scaling-rate + energy-as-uncertainty, *not* sharp small-scale generation).
- **Pre-registration is real** for the core win-criteria G0a/G2c and the 2-seed rule (predate
  results); the G2b sharpness-retention *failure* is disclosed, not hidden.
- **reg/diffusion training is stable** with clean atomic resume — no resume-poisoning or
  standardization fragility left in the publishable arms.

---

## A. The scientific tensions that matter (the real story)

All numbers below independently re-derived from the npz by the lead (CPU numpy) **and** by a
verifier agent — they are not in dispute.

**Rollout, 152 AR frames, best-of-1, seed 0:**

| arm | ctx tokens | mechanism | fg-MSE | sharp |
|---|---:|---|---:|---:|
| flat-long | 3072 | diffusion | **0.1564** | **0.0996** |
| reg-long | 1152 | **regression** | 0.1682 | 0.0741 |
| hier-long ("the system") | 1152 | diffusion | 0.1848 | 0.0729 |
| flat-budget | 1152 | diffusion | 0.2038 | 0.0582 |
| hier-short | 480 | diffusion | 0.2260 | 0.0392 |
| persistence | — | — | 0.2552 | — |

**A1 — Regression beats the diffusion system at rollout (H2 has INVERTED). [major, CONFIRMED]**
At matched 1152 tokens, reg-long beats hier-long at **every** horizon band and **every** hard
re-entry bin (ALL Δ +0.0167 [+0.0110,+0.0227]; h1-8 HARD +0.0398, h25-60 HARD +0.0235, h61+
HARD +0.0128 — all CIs exclude 0), and reg-long is **marginally sharper** (0.0741 vs 0.0729).
So the "regression only wins because fg-MSE rewards blur" escape hatch is **closed by the data**.
The pre-registered §4.6 expectation ("regression fades to the mean, diffusion holds sharpness")
is flatly refuted. Yet §4.6 and the T4 reg-long row still read "PENDING (running), pre-registered
to fade" — an examiner diffing the body against the ledger sees the article pre-registering a
prediction it already knows is false.

**A2 — The single best model is FLAT. [major, CONFIRMED]** flat-long (flat, 2.7× tokens) is the
best arm on *both* error and sharpness, beating hier-long at every band including hard re-entry
(h25-60 HARD +0.050, h61+ HARD +0.026). Hierarchy is therefore an **efficiency point** (recovers
most of the long-memory benefit at ~43% of the tokens), **not a quality frontier**. This decisive
fact appears only in the corrections ledger — absent from abstract, contributions, and the Ch.7
"one claim."

**A3 — The spine collapses to "more full-resolution context wins" unless reg-flat rescues it.
[major, CONFIRMED — this is the single biggest defense risk]** The only *published* hierarchy
demonstration (hier-long vs flat-budget) is **on the losing mechanism (diffusion)**. Ordering an
examiner sees: flat-long-diff < reg-long < hier-long-diff < flat-budget-diff. Natural adversarial
thesis: *"you've shown that at a fixed small budget, hierarchical packing beats flat packing
FOR DIFFUSION — but the winning recipe overall is just more full-resolution context, and your
best mechanism is regression."* The arm that decides whether hierarchy is a real mechanism-
agnostic property or a diffusion-only artifact is **reg-flat** (reg-hier vs reg-flat), which is
**still training (~5.5k/30k) and its eval npz does not exist**. This is now the load-bearing run.

**A4 — Ch.3 → Ch.4 protocol mismatch. [minor→major, PARTIALLY-CONFIRMED]** Ch.3's "diffusion
commits" win is **single-step, best-of-8** (oracle pixel selection — diffusion's coverage edge
switched ON). The rollout headline is **best-of-1** (that edge switched OFF), and there regression
wins. So "diffusion commits" (Ch.3) and "regression wins at rollout" (Ch.4) are partly an artifact
of different sample protocols. The promised best-of-N rollout "coverage diagnostic"
(STAGE2_DESIGN §56) was never produced. Fix by explicit reconciliation, not necessarily a new run
(diffusion's single-step edge is also precision+sharpness: mean-of-N 0.0640<0.0714, sharp
0.1886>0.1591 — not *only* coverage).

**Net for A:** the honest core that *survives* all of this is narrow but real and independently
re-verified: **at a fixed token budget, packing long-range context into a coarse level beats an
equal-token flat context, exactly past the memory horizon (null below h≈25, CI-excluding-zero win
beyond).** That is a legitimate, non-trivial contribution. It is *not* "hierarchy is the best
model" and *not* "diffusion is the right mechanism." Reframe to that, surface flat-long and
reg-long in the abstract, and the article stops being attackable on its own data.

---

## B. The memory-emergence headline has an undisclosed confound

**B1 — LR-schedule confound. [major, CONFIRMED]** `mem_gate.py:98` builds
`OneCycleLR(total_steps=steps)`, so the 6k and 20k runs are **separate, budget-matched schedules**,
not one run checkpointed twice. The 6k model is fully annealed to ~0 LR by step 6000; the 20k
model at step 6000 is still mid-schedule. So "retrieval emerges late in training" conflates
*training length* with *a different anneal schedule + 3.3× optimization budget*. Disclosed
nowhere. (The effect is still real — reframe to "appears only under a larger optimization budget,"
not a clean step-onset.)

**B2 — Forking paths: 20k is post-hoc. [major, CONFIRMED]** The *only* pre-registered escalation
in the code (`mem_gate.py:191`) is a **seed** rule: "if null on seed 0, run seed 1 before
declaring dead." That rule was satisfied and **failed at 6k** (both seeds null → by the code's own
criterion the headline is dead). The 20k re-run is an **unregistered training-length escalation**;
timestamps confirm it post-dates the null (and the commit literally titled "design memo
(pre-registered) after G0a WIN at 20k" post-dates the result). The hard-bin win *does* survive the
design-effect-widened CI on both seeds (LB +0.0057/+0.0058), so keep the result — but stop
labeling G0a "PASSED, 2-seed" as if the registered rule passed, and drop the ALL-bin "WIN" (it
reverts to null under the same correction; only the HARD bin is defensible).

---

## C. The EBT negative — trustworthy, and should you drop it?

**It is trustworthy, correctly scoped, and you should NOT drop it — but reframe it from "settled
dead" to a two-sided characterization.**

- **Port faithful, scope honest** (strengths above). The ordering×reachability decomposition is
  real and well-supported: S2 stays mis-ordered through 20k converged steps; S1 flips to correct
  ordering at ~6k (15/15 probes) yet still never generates.
- **C1 — probe norm confound. [minor, PARTIALLY-CONFIRMED]** The S2 "mis-ordered landscape"
  headline rests on E(batch-mean-token) < E(target), but the batch-mean latent is intrinsically
  **~8× lower-norm** than an individual target — a smoothness/norm effect could carry part of it.
  The rebuttal *is already in the probe* (E(randn) sits +8…+14 above E(target); the S1 flip
  overcomes the norm effect) but the article never says so. One-sentence disclosure fixes it.
- **C2 — "target flees at every step size" is overstated. [minor, OVERSTATED]** At the *operating*
  α (~12.6) the target is a shallow basin (d_tgt settles 0.55, between target and mean); the
  "grows to 85" only happens at α=1000–3000 (numerically diverging, 100× operating). Reword to
  keep the true conclusion, drop the divergence-tail number.
- **C3 — provenance landmine. [minor, OVERSTATED but real]** The copy/shift positive-control
  collapse is now cited as *vindicating* evidence, but the source log `ebt_verify_full.log:157`
  reached the OPPOSITE verdict from the same numbers ("harness wiring/scale bug → the whole
  negative is SUSPECT"). A reviewer who opens the cited log finds an unretracted contradiction.
  Add one sentence noting the earlier hypothesis and how the scaled S1 run resolved it.
- **C4 — the S1 ordering-flip softens "settled negative."** S1 *does* learn a correct landscape;
  the failure is purely reachability. Frame as "recipe-dependent ordering × scale-persistent
  reachability," not "EBT is dead."

**Bottom line on EBT:** it's your strongest *rigorous negative* and it's honest science — keep it
as the full §4.2 two-sided study you decided on. Just don't overstate it (norm confound, the
step-size tail, the harness-bug provenance) and don't call it "settled dead" when S1 repairs the
ordering.

---

## D. Reversibility & your specific question (apply it to the optimization steps)

Short answer, refined after a focused quantitative audit: **your idea is "not dead, but not a win
vs checkpointing" — and the answer depends on the recipe.** The theory backbone is genuinely
code-backed and honest, but the specific inner-loop no-go numbers are optimistic and unmeasured,
and the "cannot pay" phrasing is over-generalized.

**The nuanced answer (this is the useful part for you):**
- **For the recipe EBT actually ships** (truncated MCMC — `create_graph` on the *final* descent
  step only; `step1_mech.py:456`, `mem_panel.py:41-46`), steps k<K-1 contribute **zero** parameter
  gradient. Memory = **one** second-order graph + K tiny (y,v) states. Checkpointing keeps states/
  frees graphs; reversibility drops states to O(1). They **tie** — the saving is the trajectory
  states only. **Here the no-go is real.**
- **But it is NOT universal.** A **first-order, many-step** refiner — detached-Hessian /
  finite-difference HVP / **store-the-noise Langevin** with **no create_graph anywhere** — has no
  second-order graph at all; its peak is `K × (one energy-forward's activations)` = genuinely
  O(K), which OOMs at large K. **There, reversibility DOES save real activation memory** — a large
  fraction, not 0.2%. The catch: **so does checkpointing, identically** (both O(1) in K). So the
  first-order regime *rescues the idea into usefulness* but returns it to the **same Pareto tie as
  the time axis**, not a win. The genuine, honestly-scoped niche is therefore: *constant-memory,
  recompute-free backprop through a store-the-noise Langevin sampler with K≫128* — publishable,
  but with the same "ties checkpointing" caveat. The current §5.5 "cannot pay on inner loops"
  **erases this niche and should be qualified** to "cannot pay against the second-order truncated
  recipe; the first-order many-step regime reduces to the temporal Pareto tie."

**Two number problems in the no-go accounting (fix before submission):**
1. **The "~5.1 KB state / ~0.2% saving" drops the momentum buffer `v`.** The reconstructable state
   is (y,v) = 2560 floats ≈ **10 KB/sample** (the code `mem_panel.py:341` correctly uses the 2×
   factor; the prose in `ARTICLE_SKELETON.md:163`, `03_ROZDIL2.md:862`, `FINAL_ARTICLE_REVIEW:64`
   does not). Against the doc's own cited 1.5–2.3 GB graph, the honest saving is **~0.6–1.0%**, not
   0.2% (0.2% only holds vs a 3–4 GB peak *using the understated y-only state*). The <1% conclusion
   survives; the headline number is off ~3–5×.
2. **The 1.5–2.3 GB graph size is UNMEASURED.** `step1_mech.py` has no `max_memory_allocated` — it
   only *comments* that the naive graph "pegged 16 GB" (line 448). The measured 4-way panel
   (`mem_panel.py --measure`, T6c) and Panel A (`mem_frontier.py`, T6b) have **no committed
   artifacts** — their tables are transcribed prose. Either run them (T6c is the ~0.2% bar chart;
   `--verify` is CPU-safe and already backs the grad-identity) or drop the false precision.

**What genuinely IS novel and defensible in Ch.5** (verified code-backed; the residual after
checkpointing/Self-Forcing/DOODL/MALI/CrevNet eat the rest):
1. **Stochastic store-the-noise (T2)** — a provably-invertible *stochastic* recurrence (store ε →
   exact inverse; wrong ε breaks, 10.9). This is the one clean delta over *deterministic*
   CrevNet/TempVerseFormer, and `theory.py::t2_stochastic` actually runs it. **This is your real
   novel core.**
2. **Invertibility ≠ stability (T3)** — additive coupling is ≥1-expansive so Lip(F)≤1 still
   explodes 2.7e4×; a measured Behrmann-style result (honestly framed as "instantiated, not
   discovered") that explains the trained-RHCP fade.
3. The two-sided applicability boundary — valuable as intellectual honesty (once qualified per
   above).

**Strengths here (verified):** T1–T4 are all backed by real executable tests (`theory.py`,
`rollout_train.py`) — grad-cos vs BPTT genuinely computed, not asserted. §5.4's "delta over
TempVerseFormer" is honest and appropriately modest (claims only deterministic→stochastic, does
NOT claim the memory result as novel). And the **"~Pareto with checkpointing — NOT a win" caption
is applied flawlessly in every location** — no doc slips into claiming reversibility beats
checkpointing. That discipline is exemplary.

---

## E. Documentation debt (mechanical, but must-fix before submission)

- **Stale T1 numbers.** CH3 Table 3.1/3.2 + prose + §3.5.6 + skeleton T1 still show the pre-fix
  `clean.log` values (reg 0.0489/diff 0.0363, frac_hard 0.19). The corrected `clean2.log` values
  (reg 0.0664/diff 0.0512, frac_hard 0.27) that the audit *declared citable* were never
  propagated. [major ×2, CONFIRMED]
- **T4 headline table shows stale frame-iid CIs**, ~2.3–2.8× too narrow; one cell (h25-60 mid)
  flips WIN→null under the cluster bootstrap the ledger claims was applied. Regenerate the table
  body from `compare_clustered.log`. [major, CONFIRMED]
- **Skeleton self-contradicts:** §4.6/T4/§S mark reg-long & flat-long PENDING/RUNNING while the
  corrections ledger in the *same file* reports them DONE with narrative-overturning numbers.
  Both npz are fully evaluated. Reconcile. [major, CONFIRMED]
- **Single-step T1 eval** doesn't exclude empty-fg targets (~20%) → silently mixes full-frame MSE
  into every T1 cell (the rollout path already fixes this; apply the same treatment). [major]
- Smaller: "200-frame"→"152 AR frames" not propagated to drafts; fg-MSE degeneracy caveat lives
  only in the audit file (not CH2/CH3); token-budget bookkeeping quotes 1152/480/3072 *context*
  tokens while code/logs report 1472/800/3392 *total* — pick one and be consistent; "NFE 16"
  label on Table 3.1 hides that regression uses ~16× fewer forward passes.

---

## My honest thoughts

**Should you drop EBT for Ch. D? — This is a false choice; don't frame it that way.** The
dissertation already carries *both*: EBT as a rigorous two-sided negative (§4.2) and Ch. D as the
reversibility theory backbone. Neither depends on the other. Dropping EBT would throw away your
best honest-negative result *and* the very study that motivates the reachability discussion.
Keep both. The real decision is **not EBT-vs-Ch.D — it's the framing of the A4 positive claim**,
and that's where all the risk is.

**Is the dissertation aligned? — Yes, with one reframe.** The 4-article arc holds cleanly (A1
proposes hierarchy, A2 builds the memory/reversibility machinery, A3 defines consistency, A4
measures *when* hierarchy pays). The EBT-as-spine and "diffusion-as-limitation" framings are
verified *removed*; VQ/MobileCLIP/HVEBT contamination is gone; the continuous-diffusion paradigm
is consistent. The one misalignment: the front matter's novelty and the repeated "diffusion core
selected by mechanism analysis" thesis over-promise relative to what A4 delivers (flat > hierarchy
overall; regression ≥ diffusion at rollout). Fix the front-matter claims to match the honest core
and the arc is coherent and defensible.

**The honest A4 that survives everything** is a *memory-horizon efficiency study with regime-
dependent mechanism choice*:
- **Diffusion** wins **single-step** generative coverage + sharpness (Ch.3, corrected T1).
- **Regression** wins **deterministic best-of-1 rollout** at matched budget (Ch.4 — a genuine,
  honest finding: E1/MMNIST "regression blurs" is *not* universal; on this action-conditioned
  testbed the regressor is both more accurate and not blurrier).
- **Hierarchy** is the **token-efficient way to carry a long memory** (recovers most of full-res
  long-context benefit at ~43% of the tokens), demonstrated to pay **exactly past the memory
  horizon** — pending the reg-flat confirmation that it's mechanism-agnostic.
- **Reversibility + EBT** are two honest **boundary results** that scope the above.

That is a real dissertation chapter. It's less triumphant than "Hierarchy Earns Its Keep," but it
will not collapse under a committee's questions — and *this* version is the one your own data
supports.

**Title:** drop candidate 1 ("Hierarchy Earns Its Keep"). Use candidate 2 — **"When Does Hierarchy
Pay? A Memory-Horizon Study … at Matched Token Budgets."** It's the claim you can actually defend.

---

## Recommended path (prioritized, GPU-budget-aware)

**P0 — the one run that decides the spine (≈3 GPU-h, already ~⅕ done):**
1. **Finish reg-flat** (→30k) + rollout eval → compare reg-long(hier) vs reg-flat(flat) on the
   *same* horizon×hardness bins. **Pre-register the gate now:** hierarchy is mechanism-agnostic
   iff reg-long beats reg-flat with CI excluding 0 in the hard/high-horizon bins. Green → the
   hierarchy-efficiency claim is robust across mechanisms (spine holds, as efficiency). Red →
   hierarchy is a diffusion-only artifact; reframe hard to "long full-resolution context wins."

**P1 — de-risk the "flat wins" reading + the mechanism story (≈½ GPU-day):**
2. **flat-long-reg arm** (reg @ 3072 tokens) — is the overall frontier held by a *flat regressor*?
   Cheap (same recipe as reg-flat) and it closes the "who actually holds the large-window
   frontier" question a committee will ask.
3. **Seed-1 for hier-long + flat-budget** (the pre-registered-but-missing headline replication) —
   the T4 CIs are single-seed; the gate promised 2. Until it ships, downgrade "WIN" to "seed-0
   result, replication pending" in the headline.

**P2 — completeness (queued, low urgency):**
4. T6c measured inner-loop panel (the ~0.2% bar chart — confirms the accounting; run `--verify`
   first, it's CPU); T7 OpenSTL bridge (connectivity, not a leaderboard claim).

**P0-doc — do these regardless (no GPU, ~1 day):**
5. Propagate clean2 T1 numbers everywhere; regenerate the T4 table body with cluster CIs; delete
   the stale PENDING/§4.6 reg-long "fades to mean" text and replace with the measured reversal;
   surface flat-long + reg-long in the abstract; reframe the emergence claim as budget-dependent
   (drop "pre-registered 2-seed PASS" language); add the EBT norm-confound + provenance sentences;
   fix "152 frames" and the fg-MSE-degeneracy disclosure in the chapters; retitle to candidate 2.

The floor if every remaining bet nulls is still a complete, honest article: the testbed + protocol,
the corrected mechanism study, the two-sided EBT negative, the horizon-gated matched-budget
hierarchy result, and Ch. D. No branch leaves you empty — but P0 (reg-flat) is what determines
whether the headline is "hierarchy pays (any mechanism)" or the narrower "hierarchy pays for
diffusion; regression is the better rollout mechanism."
