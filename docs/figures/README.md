# Figures — generated from audited artifacts, logs and frozen interfaces (2026-07-17)

All produced by `scripts/make_figures.py` (CPU, from `logs_rhcp/rhvd_eval_*.npz` + bridge logs),
`scripts/make_samples.py` (light GPU decode), and `scripts/make_fig9_frontier.py` (fig9, CPU).
Regenerable from the committed data.

| file | what | chapter slot | source |
|---|---|---|---|
| fig1_claim1_horizon_curves.png | fg-MSE(all), fg-MSE(HARD), sharpness vs horizon; 6 arms + floors | CH4 §4.3.2 [FIG ch4-2], Розділ 4 §4.3 | 6 loopmom npz s0 |
| fig2_claim1_winmap.png | horizon×hardness Δ(flat−hier) heatmap, seed 0 & seed 1 | CH4 §4.3.3 [FIG ch4-3] | hier-long/flat-budget s0,s1 |
| fig3_rollout_matrix.png | 7-arm ALL fg-MSE ordering + sharpness bars | CH4 §4.3.2 Table 4.2 companion | all loopmom npz |
| fig4_claim3_factorial.png | spatial factorial: diffusion (coarse loses) vs regression (coarse ~free) | CH4 §4.4 Table 4.4 | spat18{,-reg}, flat-budget, reg-flat |
| fig5_hgrid_pair_crossover.png | historical seed-0 testbed-B profile: early win → late inversion; must be paired with E5a, where the late inversion does not recur at seeds 1–2 | CH4 §4.5.5 | d1-56-diff, d0-diff-long; `docs/confirmation/E5A_RESULTS.md` |
| fig6_mmnist_bridge.png | T7 MMNIST negative control: flat ≈ hier (MSE/SSIM/LPIPS) | CH4 §4.1.3 / appendix | mmnist_bridge_chain.log |
| fig7_samples_claim1.png | qualitative GT vs hier-long vs flat-budget at h=1..148 (HONEST: both blur late) | CH4 §4.3.3 [FIG ch4-4] | GPU decode, seed 1 |
| fig8_mmnist_samples_hier.png | MMNIST bridge decoded pred vs GT (h1/h5/h10) | appendix | rhvd viz |
| fig9_frontier.png | historical development frontier retained for provenance only; its interpolated band is retired and it must not be used as a confirmatory result figure | historical appendix only | development NPZs and logs |

| fig10_flat_vs_hier.png | historical development comparison retained for provenance only; the 1,792-token point is unresolved under the untouched simultaneous family | historical appendix only | development NPZs and logs |
| article_confirmed_frontier.png/.pdf | untouched E2 simultaneous frontier plus exact 48-frame token arithmetic; 896/1,152 hierarchy-confirmed, 1,792 unresolved, 2,432/3,072 flat-confirmed, with no interpolation | fourth article main result; dissertation Chapter 4 | `artifacts/confirmation_e2/results.json`; `scripts/make_rft_result_figures.py` |
| fig15_loopmom_observability_schematic.png/.pdf | LoopMom toroidal-world/viewport schematic plus the registered pair-derived `14 < dt <= 48` reach band and a clearly privileged pose-mosaic reference | fourth article Methods; Розділ 4 §4.1/§4.6 | frozen E2 geometry in `experiments/configs/e2_untouched_confirmation.json`; deterministic `scripts/make_rft_core_schematics.py` |
| fig16_context_packaging_architecture.png/.pdf | frozen two-level codec, joint WindowDiT and the flat/spatial/temporal context-package interfaces, including the exact 1,088-token temporal-package accounting | fourth article Methods; Розділ 3 §3.1/§3.3 | `hvp_rev/rhvd.py`; deterministic `scripts/make_rft_core_schematics.py` |
| fig17_head_package_seed_heterogeneity.png/.pdf | five-seed diffusion, regression and head×package effects; clip CIs and across-training-seed t intervals are shown as different uncertainty levels | fourth article Results; Розділ 4 §4.4 | `artifacts/e4b_head_package/results.json`; `scripts/make_rft_result_figures.py` |
| fig18_trust_negative_replication.png/.pdf | registered early/late long-loop effects at seeds 0–2, showing the replicated early benefit and failed late-inversion replication | fourth article Results/limitations; Розділ 4 §4.5 | `artifacts/e5a_trust_replication/results.json`; `scripts/make_rft_result_figures.py` |
| fig19_sampler_nfe_robustness.png/.pdf | all 20 E7c effects across NFE and sampler seeds, with the two simultaneous intervals including zero visually marked | fourth article robustness; Розділ 4 §4.6.9 | `artifacts/e7c_sampler_nfe/results.json`; `scripts/make_rft_result_figures.py` |
| fig20_alternate_codec_robustness.png/.pdf | original- and alternate-codec E7a effects in the registered joint family; the alternate upper inversion is the sole interval crossing zero | fourth article robustness; Розділ 4 §4.6.10 | `artifacts/e7a_codec_robustness/results.json`; `scripts/make_rft_result_figures.py` |
| fig21_pose_indexed_raw_memory.png/.pdf | privileged exact-pose raw-memory spans, neural references and storage/coverage accounting | fourth article closure reference; Розділ 4 §4.6.11 | terminal E8a; generated from `artifacts/e8a_pose_mosaic/results.json`; `scripts/make_rft_result_figures.py` |
| fig22_phasec_v2_boundary.png/.pdf | persistent-object long Moving-MNIST simultaneous effects: hierarchy support/continuity versus flat identity fidelity | fourth article external boundary; Chapter 4 §4.6.2a | terminal E6; generated from `artifacts/phasec_v2/results.json`; `scripts/make_rft_result_figures.py` |
| fig22_token_sequence_layout.png/.pdf | exact 1,472-token one-step sequence, including the 1,152-token context bracket and embedding/conditioning layers | Розділ 3 §3.2.1 | `hvp_rev/rhvd.py`; `scripts/make_dissertation_missing_figures.py` |
| fig23_rollout_ring_buffer.png/.pdf | fixed 48-coarse/6-fine rollout buffers, DDIM-16 four-frame recursion and 200-frame evaluation timeline | Розділ 3 §3.3.1 | `hvp_rev/rhvd.py`; `scripts/make_dissertation_missing_figures.py` |
| fig24_generation_objectives.png/.pdf | direct regression, diffusion and energy-based objectives/inference under the shared context question | Розділ 4 §4.2 | frozen mechanism definitions; `scripts/make_dissertation_missing_figures.py` |
| fig25_ebt_generation_loop.png/.pdf | tested EBT scalar-energy head and candidate-gradient generation loop, separating ordering from reachability | Розділ 4 §4.2.4 | reference implementation contract; `scripts/make_dissertation_missing_figures.py` |
| fig26_ebt_landscape_ordering.png/.pdf | target/mean/random energies across 20k updates for S1 and S2, showing recipe-dependent ordering | Розділ 4 §4.2.4 | `logs_rhcp/ebt_scaled_s{1,2}.log`; deterministic parser |
| fig27_ebt_descent_sharpness.png/.pdf | cold-descent sharpness against the registered structure threshold and decoder reference | Розділ 4 §4.2.4 | `logs_rhcp/ebt_scaled_s{1,2}.log`; deterministic parser |
| fig28_memory_gate_optimization.png/.pdf | two-seed HARD-bin effects at 6k and 20k, separating optimization emergence from horizon claims | Розділ 4 §4.3.1 | four `mem_gate_full` logs; deterministic parser |
| fig29_mechanism_single_step_samples.png/.pdf | identical single-step rows for regression and diffusion, composed without pixel editing | Розділ 4 §4.2.2 | `logs_rhcp/step1_mech_clean_{reg,diff}.png` |
| fig30_control_system.png/.pdf | open dynamic control-system view of video generation, including entropy, control, state transition and autoregressive feedback | Розділ 1 Fig. 1.1 | deterministic `scripts/make_dissertation_foundation_figures.py` |
| fig31_consistency_landscape.png/.pdf | conceptual trajectory over the consistency criterion $J(x,u)$; explicitly not an empirical energy surface | Розділ 1 Fig. 1.2 | deterministic foundation schematic |
| fig32_system_decomposition.png/.pdf | encoder–predictor–decoder structural decomposition | Розділ 1 Fig. 1.3 | deterministic foundation schematic |
| fig33_visual_hierarchy.png/.pdf | fixed macro/meso/micro abstraction ordering | Розділ 1 Fig. 1.4 | deterministic foundation schematic |
| fig34_temporal_grid.png/.pdf | physical frame time × refinement pseudo-time with a sliding context window | Розділ 1 Fig. 1.5 | deterministic foundation schematic |
| fig35_fixed_grid_tokenization.png/.pdf | equal patch cost under non-uniform information density | Розділ 1 Fig. 1.6 | deterministic foundation schematic |
| fig36_forward_diffusion.png/.pdf | conceptual Gaussian forward corruption sequence | Розділ 1 Fig. 1.7 | deterministic foundation schematic; synthetic icon only |
| fig37_reverse_diffusion.png/.pdf | conceptual reverse denoising sequence | Розділ 1 Fig. 1.8 | deterministic foundation schematic; synthetic icon only |
| fig38_latent_diffusion.png/.pdf | encoder, iterative latent core and decoder | Розділ 1 Fig. 1.9 | deterministic foundation schematic |
| fig39_flat_long_context_barrier.png/.pdf | schematic attention/KV resource ceiling and a re-entry event outside a flat window | Розділ 1 Fig. 1.10 | deterministic foundation schematic; no measured resource scale |
| fig40_context_package_comparison.png/.pdf | flat, spatial-hierarchy and temporal-stride package shapes feeding a common predictor | Розділ 1 Fig. 1.12 | deterministic foundation schematic |
| fig41_image_to_image.png/.pdf | partial-noising image-to-image conditioning | Розділ 1 Fig. 1.13 | deterministic foundation schematic |
| fig42_controlnet.png/.pdf | structural control branch and frozen diffusion backbone | Розділ 1 Fig. 1.14 | deterministic foundation schematic after ControlNet |
| fig43_shared_attention.png/.pdf | shared style attention across distinct target trajectories | Розділ 1 Fig. 1.15 | deterministic foundation schematic after StyleAligned |
| fig44_trajectory_consistency.png/.pdf | conceptual memory-window error separation and re-entry timeline | Розділ 1 Fig. 1.16 | deterministic schematic; illustrative curves only |
| fig45_blt_variable_patching.png/.pdf | entropy-driven variable-size byte patching | Розділ 2 Fig. 2.1 | deterministic foundation schematic after BLT |
| fig46_vblt_architecture.png/.pdf | local encoder, time-causal global transformer and byte-classification decoder | Розділ 2 Fig. 2.2 | deterministic schematic of the [A1] method |
| fig47_rope_nd.png/.pdf | independent rotary coordinates over horizontal, vertical and temporal axes | Розділ 2 Fig. 2.5 | deterministic schematic of the [A1] method |
| fig48_activation_free_forward.png/.pdf | tape-free reversible forward phase | Розділ 2 Fig. 2.9 | deterministic algorithmic schematic |
| fig49_reverse_local_replay.png/.pdf | reconstruct → local graph → gradient → free reverse phase | Розділ 2 Fig. 2.10 | deterministic algorithmic schematic |
| fig50_time_agnostic_backprop.png/.pdf | two-phase time-agnostic backpropagation flow | Розділ 2 Fig. 2.11 | deterministic algorithmic schematic |
| fig51_reversible_memory_frontier.png/.pdf | BPTT/checkpoint/reverse-replay peak-memory frontier through $K=160$ | Розділ 2 Fig. 2.14 | measured repository Panel-A values |
| fig52_reversibility_axes.png/.pdf | depth/time/inner-optimization applicability taxonomy with measured inner-loop boundary | Розділ 2 Fig. 2.15 | `logs_rhcp/mem_panel.md` |
| fig53_reversible_block_published.png | forward/reverse additive-coupling block | Розділ 2 Fig. 2.6 | author’s published `TempVerseFormer/resources/figures/rev-block-final.png` |
| fig54_temporal_chaining_published.png | temporal chaining mechanism | Розділ 2 Fig. 2.7 | author’s published `TempVerseFormer/resources/figures/chain-example-final.png` |
| fig55_tempverseformer_architecture_published.png | VAE plus reversible temporal core | Розділ 2 Fig. 2.8 | author’s published `TempVerseFormer/resources/figures/tempverse-arch-final.png` |
| fig56_representation_scope.png/.pdf | VQ and continuous representation interfaces with the non-comparison boundary made explicit | Розділ 1 Fig. 1.11 | deterministic foundation schematic |
| fig57_tempverse_memory_published.png/.pdf | GPU memory versus 1–1,024 prediction steps | Розділ 2 Fig. 2.12 | author preprint Fig. 13, DOI `10.21203/rs.3.rs-6293520/v1` (CC BY 4.0); version of record DOI `10.1007/s00521-025-11707-3` |
| fig58_tempverse_time_published.png/.pdf | training time versus 1–1,024 prediction steps | Розділ 2 Fig. 2.13 | author preprint Fig. 14, DOI `10.21203/rs.3.rs-6293520/v1` (CC BY 4.0); version of record DOI `10.1007/s00521-025-11707-3` |
| fig59_vblt_2d_entropy_published.png/.pdf | V-BLT ground truth, prediction and entropy for a rotating 2-D shape | Розділ 2 Fig. 2.3 | Kulyk & Mokin 2025, published Fig. 7, DOI `10.31649/1997-9266-2025-182-5-129-139` |
| fig60_vblt_3d_entropy_published.png/.pdf | V-BLT ground truth, prediction and entropy for a rotating 3-D polycube | Розділ 2 Fig. 2.4 | Kulyk & Mokin 2025, published Fig. 8, DOI `10.31649/1997-9266-2025-182-5-129-139` |

## Honesty notes carried in the figures
- fig1(c) + fig7: sharpness fades for ALL arms; the hierarchy advantage is a modest average gain,
  not a dramatic per-frame difference. The CH4 [FIG ch4-4] caption "hier re-places the digit / flat
  hallucinates" OVER-CLAIMS — replace with fig7's honest reading.
- fig5 is seed-0-only and cannot support a stable trust boundary; E5a reproduces the early benefit
  at seeds 1–2 but gives positive late hierarchy effects. On seed 0 both arms degrade below
  persistence at long horizon, the capability gate is RED/ALL-null, and the win is short-horizon
  only.
- fig2: horizon-gated onset is seed-0-specific; seed-1 wins everywhere. Report "win + mid-horizon
  peak," not the exact onset.

## Terminal figure state

All experiment-dependent figures required for the fourth article are generated. Administrative
front matter and venue-specific graphical formatting remain author/venue inputs rather than
missing experiments.
