# Figures — generated from audited npz/logs (2026-07-06)

All produced by `scripts/make_figures.py` (CPU, from `logs_rhcp/rhvd_eval_*.npz` + bridge logs),
`scripts/make_samples.py` (light GPU decode), and `scripts/make_fig9_frontier.py` (fig9, CPU).
Regenerable from the committed data.

| file | what | chapter slot | source |
|---|---|---|---|
| fig1_claim1_horizon_curves.png | fg-MSE(all), fg-MSE(HARD), sharpness vs horizon; 6 arms + floors | CH4 §4.3.2 [FIG ch4-2], Розділ 4 §4.3 | 6 loopmom npz s0 |
| fig2_claim1_winmap.png | horizon×hardness Δ(flat−hier) heatmap, seed 0 & seed 1 | CH4 §4.3.3 [FIG ch4-3] | hier-long/flat-budget s0,s1 |
| fig3_rollout_matrix.png | 7-arm ALL fg-MSE ordering + sharpness bars | CH4 §4.3.2 Table 4.2 companion | all loopmom npz |
| fig4_claim3_factorial.png | spatial factorial: diffusion (coarse loses) vs regression (coarse ~free) | CH4 §4.4 Table 4.4 | spat18{,-reg}, flat-budget, reg-flat |
| fig5_hgrid_pair_crossover.png | testbed B: d1 vs d0 rollout, early win → late inversion, both hit persistence | CH4 §4.5.5 | d1-56-diff, d0-diff-long |
| fig6_mmnist_bridge.png | T7 MMNIST negative control: flat ≈ hier (MSE/SSIM/LPIPS) | CH4 §4.1.3 / appendix | mmnist_bridge_chain.log |
| fig7_samples_claim1.png | qualitative GT vs hier-long vs flat-budget at h=1..148 (HONEST: both blur late) | CH4 §4.3.3 [FIG ch4-4] | GPU decode, seed 1 |
| fig8_mmnist_samples_hier.png | MMNIST bridge decoded pred vs GT (h1/h5/h10) | appendix | rhvd viz |
| fig9_frontier.png | token-budget frontier: fg-MSE(ALL, 152-frame rollout) vs context tokens (log-x), 6-point flat vs hier curves (896/1152/1792 hier-win, 512/2432/3072 flat-win) + reg context points; hier wins the band [896, ~2000) with the upper edge bracketed to (1792,2432) by R-F/R-G (shaded; geometric span-threshold dashed), peak +0.0399/+0.0358 at the lower edge (two seeds), inverts outside | CH4 §4.3.x (→§4.3.6), Розділ 4 §4.3.6 [FIG 4-x] | Table 4.2 + `frontier_{top,bottom,low,c1,c2}.log` |

| fig10_flat_vs_hier.png | flat vs hier at 6 matched budgets (512/896/1152/1792/2432/3072): (A) grouped bars seed-mean+min/max whiskers, per-budget winner; (B) paired Δ(flat−hier) per seed with 95% cluster CIs, shaded win band (896/1152/1792 hier), sign flip at both edges; two-seed except the 1792/2432 bracket | CH4 §4.3.6 companion, Розділ 4 §4.3.6 | rhvd_eval_*.npz + frontier_{bottom,low,top,c1,c2}.log + headline compares |

## Honesty notes carried in the figures
- fig1(c) + fig7: sharpness fades for ALL arms; the hierarchy advantage is a modest average gain,
  not a dramatic per-frame difference. The CH4 [FIG ch4-4] caption "hier re-places the digit / flat
  hallucinates" OVER-CLAIMS — replace with fig7's honest reading.
- fig5: on testbed B both arms degrade below persistence at long horizon (the capability claim is
  RED / ALL-null); the win is short-horizon only.
- fig2: horizon-gated onset is seed-0-specific; seed-1 wins everywhere. Report "win + mid-horizon
  peak," not the exact onset.

## Still missing (need a run or author input)
- EBT two-sided landscape/reachability figure (data in ebt_scaled_s{1,2}.log; not yet plotted).
- Reversibility memory-frontier panel (data in mem_panel.md / theory logs; not yet plotted).
- Testbed schematic (world + viewport + re-entry) — a drawn diagram, not from data.
- Architecture diagram (2-level AE + WindowDiT) — drawn.
