# FINALIZATION PLAN — everything remaining for the dissertation + article (2026-07-07 evening)

> Companion documents: `HANDOFF_2026-07-07.md` (state summary — what exists and why),
> `COMPLETION_PLAN_2026-07-07.md` (the E-phase tracker this plan details),
> `docs/dissertation/rewrite/07_REFERENCES.md` (bibliography + binding),
> `docs/SPATIAL_FACTORIAL.md` (all frontier pre-registrations incl. the robustness program),
> `docs/NEXT_EXPERIMENTS_GENERALIZATION.md` (generalization protocols; TaxiBJ measured).
> Everything below is a LIST with description + argumentation, per author request; sample code
> included for every fill-in that follows an experiment landing.

---

## PART A — EXPERIMENTS: running, queued, remaining, and explicitly-not-needed

### A1. Running / queued now (the frontier robustness program; pre-registered, binding)

| # | run | cost | why (argumentation) | fills |
|---|---|---:|---|---|
| R-A | hier-min + flat-min, **seed 1** (B16×30k ×2 + compare) — **RUNNING** | ~6–7 h | The 896-token point is the campaign's largest effect (+0.0399) but single-seed; the binding two-seed rule forbids any headline use without replication. Also decides whether "hier-min ≥ hier-long" was seed noise (expected) or a real fine-context effect. | `logs_rhcp/frontier_low_s1.log`, `rhvd_eval_{hier-min,flat-min}_s1.npz` → CH4 §4.3.6 lower-edge paragraph ([SEED-1 PENDING] → measured), CH8 §8.1, fig9 seed markers, Розділ 4 §4.3.6, VYSNOVKY result 7 |
| R-B | hier-rich + flat-long, **seed 1** (B8×60k ×2 + compare) | ~19 h | The top pair (3,072 tok) is single-seed on BOTH arms; a CI-clean −0.0131 is unlikely to flip, but the plateau reading (hier ≈0.17 from 896 onward) needs the hier-rich point seed-checked, and symmetric seeds are the anti-cherry-picking guardrail. | `frontier_top_s1.log`, two npz → same doc set; fig9 seed markers at 3,072 |
| R-C | hier-long + flat-budget at **B8×60k**, seed 0 (budget-matched "train longer") | ~18 h | The legitimate version of "wait longer": the T3 emergence result (memory null@6k → win@20k) predicts the hier−flat delta at 1,152 GROWS with training budget. If hier-long-60k lands ≈0.169 (the plateau floor), the mid-point's seed-0 dip is fully explained; if the delta shrinks, that is the budget-saturation reading — either way pre-registered, no soft reads. | `frontier_mid_60k.log`, `rhvd_eval_{hier-long,flat-budget}-60k…npz` → a one-paragraph addition to CH4 §4.3.6 + Table 4.6a footnote |

**Launch commands (R-B, R-C — run each after the previous chain's DONE marker; one job at a time):**
```bash
# R-B (after RA_DONE):
venv/Scripts/python.exe -u -m hvp_rev.rhvd --arm hier-rich --steps 60000 --batch 8 --seed 1 --bf16 \
  > logs_rhcp/rhvd_hier-rich_s1.log 2>&1 && \
venv/Scripts/python.exe -u -m hvp_rev.rhvd --arm flat-long --steps 60000 --batch 8 --seed 1 --bf16 \
  > logs_rhcp/rhvd_flat-long_s1.log 2>&1 && \
venv/Scripts/python.exe -u -m hvp_rev.rhvd --compare flat-long,hier-rich --seed 1 \
  > logs_rhcp/frontier_top_s1.log 2>&1 && echo RB_DONE
# R-C (after RB_DONE): NOTE new arm names must be registered as 60k variants OR reuse arms with
# --steps 60000 --batch 8 (ckpt names encode steps, so no collision with the 30k ckpts):
venv/Scripts/python.exe -u -m hvp_rev.rhvd --arm hier-long --steps 60000 --batch 8 --seed 0 --bf16 \
  > logs_rhcp/rhvd_hier-long_60k.log 2>&1 && \
venv/Scripts/python.exe -u -m hvp_rev.rhvd --arm flat-budget --steps 60000 --batch 8 --seed 0 --bf16 \
  > logs_rhcp/rhvd_flat-budget_60k.log 2>&1 && \
venv/Scripts/python.exe -u -m hvp_rev.rhvd --compare flat-budget,hier-long --seed 0 \
  > logs_rhcp/frontier_mid_60k.log 2>&1 && echo RC_DONE
```
*(R-C caveat: `--compare` reads `rhvd_eval_{arm}_s0.npz`, which the 60k training overwrites —
archive the 30k npz first: `cp logs_rhcp/rhvd_eval_hier-long_s0.npz logs_rhcp/archive_30k/` etc.,
or add an `--eval_tag _60k` pass-through. CHECK before launching R-C.)*

### A2. Generalization campaign — remaining datasets (pre-registered in the memo)

| # | dataset | cost | why (argumentation) | build needed |
|---|---|---:|---|---|
| G-i | **2D rotations** ([A2] testbed) | 7–9 GPU-h + ~0.5 build-day | The cross-article continuity read: the NEW stack on the SAME testbed as the published [A2] baselines — the dissertation's strongest "beats the [A2] baselines" candidate (pre-registered [EXPECTED]); reg-head dominance + hier≈flat null #4 expected. **Send the [AUTHOR: A2 result tables] request EARLY — it blocks the comparison rows, not the runs.** | B3 dump script over `TempVerseFormer/simple-shape-dataset-toolbox` (record seeds!); B1 RGB AE (`HierConvAE(img_ch=3)` — constructor already parametrized, same pattern as TaxiBJ's img_ch=2); B4 Δ-conditioning token for read (A) |
| G-ii | **3D polycubes** ([A2] 3D testbed) | 8–10 GPU-h + ~0.1 marginal | The ONE generalization dataset where diffusion best-of-8 could win (occlusion multimodality) — Claim 2's coverage read on the author's own published testbed. Rides on G-i's build. | shared with G-i; [VERIFY 3D render timing with a 100-clip probe] |
| G-iii | **MMNIST-CIFAR** | 9–10 GPU-h + 0.25–0.5 build-day | Robustness of the T7 null on textured video; RGB; canonical test file already local. The AE-ceiling row is the scientific point (does a 1.85M-class AE carry CIFAR texture?). Confirmatory — schedule in a gap. | CIFAR compositing port into `mmnist_bridge`-pattern generator (bit-check vs canonical test file); B1 RGB AE |
| G-vi | **KTH-64** | ~1–1.5 days incl. download | **RECOMMEND SKIP** (option b, pre-authorized): resolution incomparability makes every row an asterisk, real-data transfer is already carried by TaxiBJ *with a leaderboard-band result* — KTH adds risk, not evidence. If run anyway: only under the disclosure protocol (option a). | video→npy adapter + person-split port |

**Template chain per dataset (the TaxiBJ/T7 pattern — build the bridge module, then):**
```bash
venv/Scripts/python.exe -u -m hvp_rev.rot2d_bridge --stage smoke --bf16          # ~5 min gate
venv/Scripts/python.exe -u -m hvp_rev.rot2d_bridge --stage all --bf16 --target_gb 14 \
  > logs_rhcp/rot2d_bridge_chain.log 2>&1 && echo ROT2D_ALL_DONE                  # background
```

### A3. Optional science (only if GPU idles; each needs pre-registration first)

- **Loop-80 operating point A′** (~1 GPU-day + dataset build): world ≈288 px, L_eff ≈ 80,
  T=200 clips. Argumentation: the ONLY remaining path to a *top-budget* hierarchy win (flat at
  3,072 tok = 48 frames < 80-frame loop) AND it maps where between loop-48 (works) and loop-213
  (fails) the trustworthy-memory boundary sits — the highest-value optional item, but new scope;
  article/dissertation are complete without it. Defer unless the queue drains early.
- EBT cells at operating point B (~5 h; HGRID §9 spec) — postponed, low priority.
- hgrid2 cue-strength ablation backfill (~2 h) — record-keeping value only.

### A4. Explicitly NOT needed (with the reason on the record)

- **Tier-A H-grid** — closed by the binding rule (pair RED); running it would violate the rule.
- **MMNIST 60k leaderboard chase** — rejected in the memo: the chapter's claim is transfer, not
  ranking; T7's negative-control value does not improve with more steps.
- **Best-of-seed reruns of any single arm** — forbidden by the robustness pre-registration
  (symmetric pairs only, all seeds reported).

---

## PART B — TEXT: what is DONE vs what remains

### B1. DONE (for orientation; all committed)
Article: all 8 chapters as prose (CH1 intro+abstract, CH2 testbed, CH3 mechanism+EBT, CH4
results incl. frontier §4.3.6 + closure §4.5.6 with all BRANCH markers resolved, CH5
generalization with T7+TaxiBJ measured, CH6 reversibility, CH7 related+non-claims, CH8
conclusion). Dissertation: 00–06 rewrites content-complete; all 12 audit fixes applied (novelty
re-scope, Положення block, decision-theory methods, ROZDIL2 math corrections, 0.26% sync,
Ch3-summary consistency, §1.4.4 alignment, 200-frame, MSE range, trustworthy clause); 07 master
bibliography with binding table; [A1]-as-first-hierarchy framing in three places.

### B2. Remaining text work — by owner

**Author-blocking (E7 — THE critical path; nothing else can substitute):**
1. Publication list: exact entries for [A1]–[A4] + 6 theses (07_REFERENCES §A); [A4]'s venue and
   status by defense decides the 3+1+6 count; DSMSI-2023 verification; Q1-vs-Scopus wording.
2. Formalities: НДР numbers ×2 (+wording vs the НДР-2 report), UDC confirmation, page counts at
   assembly, council template for «Положення, що виносяться на захист».
3. Evidence receipts: VQ-instability receipts for §1.3.2 (or soften to "preliminary
   experiments"); [A2] dataset/result tables (feeds §4.1.2 AND the G-i/G-ii comparison rows —
   send early); exact [A2] TaxiBJ/KTH rows for Tables 4.11/4.12.
4. Supervisor items: UA systems-under-risk sources (07_REFERENCES `UA-риз`), reversibility-scope
   sign-off (Q1b), Q4-amended TaxiBJ disposition acknowledgement.

**Mechanical, mine (E6/E8):**
5. Bibliography verification pass: clear the ~12 [VERIFY] flags against PDFs; MilliVid entry
   from the sibling repo; then the count decision (~95 entries vs enriching §1.1.2/§1.3 toward
   the traditional ~145 — recommend NOT padding).
6. Number-binding pass (after 5 + author entries): substitute inline [REF: descriptor] → [N]
   by the binding table, order of first mention. Sample substitution script:
```python
# scripts/bind_refs.py — run AFTER 07_REFERENCES.md is verified & author entries bound
import re, json
ORDER = ["01_VSTUP.md","02_ROZDIL1.md","03_ROZDIL2.md","04_ROZDIL3.md","05_ROZDIL4.md","06_VYSNOVKY.md"]
binding = json.load(open("scripts/ref_binding.json"))   # descriptor -> [keys] (export from §C table)
numbers, counter = {}, 0
def sub(m):
    global counter
    keys = binding[' '.join(m.group(1).split())]
    nums = []
    for k in keys:
        if k not in numbers: counter += 1; numbers[k] = counter
        nums.append(numbers[k])
    return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"
for f in ORDER:
    p = f"docs/dissertation/rewrite/{f}"
    s = open(p, encoding="utf-8").read()
    s = re.sub(r"\[REF:\s*([^\]]+)\]", sub, s)
    open(p, "w", encoding="utf-8").write(s)
json.dump(numbers, open("scripts/ref_numbers.json","w"))  # key -> N, drives the final list order
```
7. Article venue-format pass: venue choice [AUTHOR], length trim (CH4 §4.5.2–4.5.3 H-grid
   narrative → one table + paragraph; duplicated floor recitations), reference style conversion,
   appendix C (reproducibility statement — write from HANDOFF §7 artifact inventory, ~1 page).
8. Abstract/CH8 final polish after R-A/R-B/R-C land (delete [SEED-1 PENDING], insert seed-mean
   band wording).
9. UA translation (E9, LAST, on frozen text; 49 [UA-PASS:] idioms pre-placed) + merge into the
   submission .docx (replace donor content wholesale; regenerate ЗМІСТ; final abbreviation list;
   fill volume counts).

---

## PART C — FIGURES/IMAGES: inventory, argumentation, and generation code

### C1. Exists (12 assets)
fig1–fig9 (`docs/figures/`, regenerable; fig9 = the frontier band with seed-1 markers),
`taxibj_bridge_txb-*_t15000.png` (G-5 qualitative, [FIG gen-5] candidates),
`fig_horizon_curves_loopmom.png` (CH4 ch4-2 asset), training grids + EBT probe PNGs (spot evidence).

### C2. Missing — RENDER (data already on disk; ~0.5 day CPU total)

| fig | content | data source | why needed |
|---|---|---|---|
| F3/F4 (ch3-2/3-3) | EBT energy-ordering vs step (S1 flips at ~6k, S2 never) + cold-descent sharpness vs step | T2 tables in `ARTICLE_SKELETON.md` (from `ebt_scaled_s{1,2}.log`) | The two-sided EBT negative is a headline contribution (C3) with ZERO figures — reviewers will want the picture |
| F5 | memory-gate emergence: HARD-bin Δ with CIs, 6k vs 20k × 2 seeds | T3 table (mem_gate logs) | The "emergence needs budget" lesson in one glance |
| F9 | Panel A memory frontier: MB vs K, 3 curves, OOM markers | T6b table | Ch. 6's central measured claim |
| F10 | Lipschitz stability panel: perturbation growth vs Lip bound | T6a P3 row | invertibility ≠ stability — the chapter's most interesting result |
| F12 | inner-loop 4-way memory panel | T6c table (`mem_panel.md`) | closes the 0.26% no-go visually |
| F2 | mechanism qualitative grid (reg averages / diff commits) | compose from `step1_mech_clean2_{reg,diff}.png` | Claim 2's qualitative face |

**Sample code (F3 — EBT ordering; same pattern serves F4/F5/F9/F10/F12 with their tables):**
```python
# scripts/make_fig_ebt.py -> docs/figures/fig10_ebt_twosided.png
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
steps  = [0, 5000, 10000, 15000, 20000]
s2_tgt = [-1.004, -35.579, -42.136, -41.891, -41.761]   # ebt_scaled_s2.log probes
s2_mean= [-1.414, -36.605, -43.232, -42.982, -42.951]
s1_tgt = [-0.337, -25.654, -35.557, -40.858, -44.097]   # ebt_scaled_s1.log probes
s1_mean= [-0.601, -25.765, -32.595, -38.421, -41.961]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
for a, tgt, mean, ttl, flip in ((ax[0], s2_tgt, s2_mean, "S2 recipe: stays MIS-ordered (mean below target, 21/21 probes)", None),
                                (ax[1], s1_tgt, s1_mean, "S1 recipe: ordering REPAIRED from ~6k (15/15 probes)", 6000)):
    a.plot(steps, tgt, "-o", label="E(target)"); a.plot(steps, mean, "-s", label="E(batch mean)")
    if flip: a.axvline(flip, ls="--", lw=0.9, color="0.5"); a.text(flip, a.get_ylim()[1], " ordering flips", fontsize=8, va="top")
    a.set_title(ttl, fontsize=9.5); a.set_xlabel("training step"); a.legend(fontsize=8); a.grid(alpha=.25)
ax[0].set_ylabel("energy (lower = preferred by descent)")
fig.suptitle("Two-sided EBT study: landscape ordering is recipe-dependent; generation unreachable under both", fontsize=10.5)
fig.tight_layout(); fig.savefig("docs/figures/fig10_ebt_twosided.png", dpi=180)
```

**Fill-in code for the robustness landings (run after each DONE marker):**
```bash
# after RA_DONE / RB_DONE / RC_DONE — read verdicts:
tail -40 logs_rhcp/frontier_low_s1.log     # (resp. frontier_top_s1.log / frontier_mid_60k.log)
# then: update the seed-1 constants in scripts/make_fig9_frontier.py (hier/flat seed-1 markers,
# or switch the curves to seed-means once both seeds exist at 896/1152/3072) and regenerate:
venv/Scripts/python.exe scripts/make_fig9_frontier.py
# then swap the [SEED-1 PENDING] bracket in CH4 §4.3.6 / CH8 §8.1 / Розділ 4 §4.3.6 / VYSNOVKY
# for the measured CI, per the pre-registered reading (SPATIAL_FACTORIAL robustness block).
```

### C3. Missing — DRAW (schematics; ~0.5–1 day, matplotlib/SVG or hand-drawn)

| fig | content | why |
|---|---|---|
| F1 / [FIG ch2-1] | testbed world strip: 192-px torus, 64-px viewport, exit→re-entry arc, loop≈48 annotation | THE instrument in one picture; reviewers grasp the whole design from it |
| [FIG 3-1] | architecture: frozen 2-level AE (freeze icons) → standardize → WindowDiT (coarse ring 48 / fine ring / actions / τ) → 4-frame window → frozen decoder | the dissertation's model chapter and article Ch. 2/4 share it |
| [FIG 3-2] | token-sequence layout bar: 768+384+320 slots, frame-id axis 0…51, embedding stack | makes the matched-budget arithmetic visible — the claim's bookkeeping |
| F11 / [FIG 2-15] | three-axis reversibility taxonomy with the measured annotations (pays / cannot-pay) | Ch. 6 closing figure |
| ROZDIL1 concept diagrams (~4) | control-loop view of a generator; 3-level decomposition; uncertainty taxonomy; consistency-criteria map | genre expectation for a 124 systems-analysis chapter |
| [FIG gen-0] | recipe card: head × regime × representation decision table | CH5's one-glance summary |

### C4. Figure-numbering unification (E8): the article uses F1–F12 + gen-*; the dissertation uses
[FIG k-n] placeholders — build the final manifest (placeholder → asset path) when rendering; the
placeholders already carry content + source lines, so this is mechanical.

---

## PART D — the assembly checklist (E8, in order)

1. R-A/R-B/R-C + generalization results swapped in (Part A fills).
2. Bibliography verified → `bind_refs.py` → numbered list generated (ДСТУ at UA pass).
3. Figures rendered/drawn → manifest → placeholders replaced.
4. Author cluster (Part B2 items 1–4) resolved.
5. Cross-read pass: front-matter ↔ VSTUP ↔ VYSNOVKY numbers one final time (the three files carry
   the same 7 results — any late edit must propagate to all three).
6. ЗМІСТ regeneration, abbreviation-list finalization, volume counts, ДОДАТКИ А–Г content.
7. Author review pass over everything → freeze → UA translation → .docx merge.
8. Article: venue format, length trim, submission package (code/data availability from the
   artifact inventory).

**Risk ranking (unchanged from the audit):** 1) publication list / [A4] status (existential for
the count), 2) [VERIFY]-flag citations (opponent bait if wrong), 3) UA-translation drift (locked
by the one-canonical-text convention + UA-PASS idioms), 4) generalization [EXPECTED] shells left
unstruck at defense (the run-or-strike rule covers it), 5) seed-1 outcomes moving a frontier
edge (pre-registered re-bracketing covers it).
