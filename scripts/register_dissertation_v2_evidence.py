"""Build traceable result slots, experiment registry and task/claim map (no training)."""
from __future__ import annotations
import json
import re
import sys
import subprocess
import types
from pathlib import Path
from refactor_dissertation_v2 import ROOT, EDIT, FIG, MAN, FILES, digest, write, jwrite
from dissertation_evidence_contracts import logical_family, integrated_snapshot, snapshot_markdown

sys.path.insert(0, str(ROOT))
# Load the pure-Python configuration submodule without the package's torch/state imports.
package = types.ModuleType("hvp_hca")
package.__path__ = [str(ROOT / "hvp_hca")]
sys.modules["hvp_hca"] = package
from hvp_hca.article4_campaign import load_campaign_run_spec


def main():
    snapshot = integrated_snapshot()
    campaign_file = ROOT / "experiments/v2/campaigns/article4_primary_30k_2026-09-01.json"
    campaign = json.loads(campaign_file.read_text(encoding="utf-8"))
    rows = []
    for entry in campaign["run_order"]:
        path = ROOT / entry["config"]
        spec = load_campaign_run_spec(path)
        plan = spec.plan
        rows.append({"experiment_id": spec.experiment_id, "config": entry["config"], "config_hash": spec.content_hash,
                     "file_sha256": digest(path.read_bytes()), "seed": spec.seed, "family": spec.family,
                     "budget": plan.token_count, "reach": plan.reach, "coarse_ages": list(plan.coarse_ages),
                     "top_ages": list(plan.top_ages), "fine_age_cells": list(plan.fine_age_cells),
                     "fine_tokens": plan.fine_tokens, "policy": spec.policy, "overlap": spec.overlap,
                     "steps": spec.train_updates, "status": "registered; result not imported",
                     "spec": spec.canonical_dict()})
    assert len(rows) == 80
    for role in ["LVL_three_level_from_selected_hierarchy", "OVR_exclusive_from_selected_feasible_hierarchy", "OVR_residual_from_selected_feasible_hierarchy"]:
        for seed in [0, 1]:
            rows.append({"experiment_id": f"DEPENDENCY_PENDING_{role}_S{seed:02d}", "config": None,
                         "config_hash": None, "seed": seed, "family": role.split("_")[0], "budget": 1792,
                         "steps": 30000, "status": "dependency unresolved in frozen editorial input; do not invent config/hash", "selection_role": role})
    jwrite(EDIT / "registries/experiment_matrix.json", {"source_campaign_sha256": digest(campaign_file.read_bytes()), "unique_configurations": 43, "runs": rows,
        "reuse": {"ALC": "three FRT B1792 policy pairs", "LVL": "Flat and eligible selected two-level ALC pair", "OVR": "eligible nested ALC pair"}})
    matrix = ["# Exact hierarchy experiment matrix", "", "86 registered training runs, 43 configurations, two seeds. The six dependency rows are not fabricated final run identifiers. No live completion status is inferred from this frozen registry.", "", "| Run identifier | Family | Tokens | Reach | Fine tokens | Seed | Config hash / status |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        matrix.append(f"| {row['experiment_id']} | {row['family']} | {row['budget']} | {row.get('reach', 'pending')} | {row.get('fine_tokens', 'pending')} | {row['seed']} | {row.get('config_hash') or row['status']} |")
    matrix += ["", "CEX varies payload, compute and reach together. LVL adds a top stream, reallocates payload and changes parameters. OVR changes ownership/representation and possibly ages. STS uses only 1,792 tokens. Full tensors and configs are in the JSON registry; the human labels and scientific questions are in Chapter 3."]
    write(EDIT / "EXPERIMENT_MATRIX.md", "\n".join(matrix))
    # Build every placeholder before inserting available bounded evidence.
    main_text = {n: (MAN / n).read_text(encoding="utf-8") for n in FILES}
    main_text[FILES[8]] = main_text[FILES[8]].replace('<!-- INTEGRATED_SOURCE_SNAPSHOT -->', snapshot_markdown(snapshot))
    templates = [p.read_text(encoding="utf-8") for p in (EDIT / "english").glob("*.md")]
    keys = sorted({key for text in list(main_text.values()) + templates for key in re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)})
    registry = []
    for key in keys:
        integrated = key.startswith("INT_") or key == "CLAIM_INT_SUMMARY"
        calibration = key == "A4_CAL_CONVERGENCE_FIG"
        confirmation = "CONFIRMATION" in key
        optional = "ROLLOUT" in key
        mec = "MEC" in key
        family = key.split("_")[1] if key.startswith("A4_") else "INTEGRATED"
        family_rows = logical_family(rows,family)
        status = "CPU correctness gate" if key == "INT_GATE_SUMMARY" else "archived calibration" if calibration else "pending"
        evidence = "mechanism diagnostic" if mec else "sealed-holdout confirmation of selected roles only" if confirmation else "registered development estimate" if not integrated else "conditional integrated evidence cell"
        source = "artifacts/v2/analysis/article4_schedule_calibration_2026-09-01.json" if calibration else "artifacts/v2/article4_primary_30k/post_training/final_report.json" if confirmation or mec else "artifacts/v2/article4_primary_30k/post_training/all_final_development_metrics.json" if not integrated and not optional else "separately authorized sealed rollout bundle (not yet registered)" if optional else "integrated accepted evidence bundle (not yet measured/locked)"
        kind = "same-checkpoint execution/profile" if any(x in key for x in ["CHECKPOINTING", "COMPONENT_MEMORY", "ROUTING_SWITCH"]) else "separately trained development ablation" if any(x in key for x in ["TRAINING_REGIME", "ABLATION"]) else "matched local baseline" if "BASELINE" in key else "long full-model confirmation" if integrated and any(x in key for x in ["MM_", "TAXIBJ_"]) else evidence
        row = {"key": key, "status": status, "evidence_status": evidence, "evidence_cell_kind": kind,
               "locations": [f"{n}:{i}" for n,t in main_text.items() for i,l in enumerate(t.splitlines(),1) if "{{"+key+"}}" in l],
               "source_artifact": source, "config_hashes": [r['config_hash'] for r in family_rows if r.get('config_hash')],
               "exact_configuration_ids": [r['experiment_id'] for r in family_rows],
               "logical_run_bindings":[{k:r[k] for k in ['experiment_id','config_hash','seed','selection_role','status','policy'] if k in r} for r in family_rows],
               "data_split": "locked selected confirmation" if confirmation or mec else "development validation" if not integrated else "validation-only selection then locked test for final cells",
               "population": "TRB four-frame endpoint; re-entry population only for registered selected contrasts" if not integrated else "Moving MNIST 10-to-10 or TaxiBJ 4-to-4; dataset-specific rows required",
               "training_seeds": [0] if calibration else [0,1], "uncertainty_unit": "show both training seeds; clip intervals conditional on trained model, if registered",
               "metric": "horizon4_foreground_mse" if not integrated else "declared quality metrics, memory allocated/reserved GiB, throughput; never combine unlike units",
               "metric_implementation": "scripts/v2/train_article4_campaign.py; hvp_rev/evaluation" if not integrated else "OpenSTL metric/config revision and paired profile boundaries must be bound to accepted bundle",
               "selection_role": "locked roles only" if confirmation or mec else "development; no test-driven selection",
               "sealed_holdout_exposed": "not read by editorial scripts; inspect accepted access ledger before filling",
               "generator": "scripts/make_dissertation_v2_assets.py" if calibration else "scripts/plot_dissertation_v2_results.py (validated accepted-row interface)",
               "generator_sha256": digest((ROOT / ("scripts/make_dissertation_v2_assets.py" if calibration else "scripts/plot_dissertation_v2_results.py")).read_bytes()),
               "expected_file_path": f"figures/dissertation_v2/{key.lower()}.svg" if key.endswith("FIG") else f"docs/dissertation/refactor_v2/results/{key.lower()}.md",
               "completion_check": "required paired seeds; source/config/checkpoint/data hashes; fixed endpoint; exact role/population; accepted immutable bundle; no incomplete records",
               "failure_absence_behavior": "retain explicit pending slot in draft; remove result claim/slot or state negative boundary at release; never invent a value",
               "allowed_claim_language": "name comparator, metric, population, protocol and evidence status; token count is not VRAM; no ratio multiplication",
               "positive_text": "On {population}, {candidate} improved {metric} relative to {comparator} under {matched_contract}; {seed_values} and {resource_values} define the observed effect. State direction reproduction only if both signs agree.",
               "neutral_text": "The declared quality tolerance was met without establishing a quality advantage; the measured resource difference was {resource_values} under {matched_contract}.",
               "negative_text": "The expected advantage was not established on {population}; report {seed_values}, failed tolerance/gate, and the narrower supported conclusion."}
        registry.append(row)
        if confirmation:
            row.update(metric='mean_foreground_mse; reentry_foreground_mse; non_reentry_foreground_mse (separate tables)',
                       population='500 fresh TRB clips × four horizons; common 96-frame observed / 28-frame recent-reference sprite-visibility flag; not package-specific E2 pixels',
                       metric_implementation='scripts/v2/finalize_article4_campaign.py:visibility_labels and evaluate_selected_run',
                       source_artifact='post_training/final_report.json.evaluations and per_clip_horizon_metrics.npz; selection_ledger.json binds roles',
                       positive_text='The selected role reproduced the declared effect on the common confirmation population; report per-seed mean_foreground_mse and reentry_foreground_mse separately. Do not promote the complete development frontier.',
                       neutral_text='The selected role did not establish a directional advantage on the common population; report both seed contrasts without calling the full family confirmed.',
                       negative_text='The selected development role did not reproduce on fresh confirmation; report the failed contrast and retain the development/confirmation distinction.')
        if mec:
            row.update(metric='attention mass by context span/layer',
                       population='one fixed clip 0; direct denoiser forward at diffusion time 0.5 with target latents; no sampling trajectory',
                       metric_implementation='scripts/v2/finalize_article4_campaign.py:attention_diagnostics, token_spans, evaluate_selected_run',
                       positive_text='The fixed attention probe assigned nonzero mass to older context. This is descriptive context use, not a causal ablation or replicated quality result.',
                       neutral_text='The fixed probe did not distinguish context-use patterns; no causal inference is drawn.',
                       negative_text='The fixed probe does not support the proposed context-use account; the measured quality result, if any, remains separate.')
        if integrated:
            dataset = 'mmnist' if '_MM_' in key else 'taxibj' if '_TAXIBJ_' in key else None
            planned = [r for r in snapshot['runs'] if r['dataset']==dataset] if dataset else []
            row.update(exact_configuration_ids=[r['run_id'] for r in planned],
                       config_hashes=[],prelock_configuration_bindings=planned,
                       source_snapshot='registries/integrated_source_snapshot.json',
                       population='Moving MNIST, 10 observed → 10 predicted frames' if dataset=='mmnist' else 'TaxiBJ, two channels, 4 observed → 4 predicted frames' if dataset=='taxibj' else row['population'])
            if any(s in key for s in ['MEMORY','CHECKPOINTING']):
                row.update(positive_text='Under the same checkpoint/function, batch, dtype and host, measured {memory_endpoint} decreased from {baseline} to {candidate}; report throughput/recomputation separately.',neutral_text='Measured memory was similar under the matched execution contract; report overhead without claiming a storage gain.',negative_text='The proposed execution mode did not reduce the declared measured peak, or failed parity; no reversible or full-model memory advantage is established.')
            if 'ROUTING' in key:
                row.update(positive_text='On the declared checkpoint/training regime and fixed quota, {route_endpoint} improved; this is allocation/stability evidence, not fewer stored tokens.',neutral_text='The route policy/regime changed membership without establishing improved quality/stability. Equal quota remains equal token count.',negative_text='The route policy/regime degraded quality/stability or collapsed; do not attribute a smaller-quota memory gain to this result.')
            if 'SYNTHESIS' in key or key=='CLAIM_INT_SUMMARY':
                row.update(positive_text='Dataset-specific matched results support the explicitly declared quality tolerances and measured memory comparisons. Report Moving MNIST and TaxiBJ separately; do not average unlike metrics or multiply component ratios.',neutral_text='Composition is compatible, but competitive quality or the proposed resource advantage is not established on every dataset. State each dataset result separately.',negative_text='The integrated claim is unsupported for the missing or failed dataset/component cells; preserve the narrower architecture-correctness and historical conclusions.')
        names = {'A4_FRT_PRIMARY_FIG':'a4_frt_budget_response','A4_ALC_PRIMARY_FIG':'a4_alc_policy_effect','A4_CEX_PRIMARY_FIG':'a4_cex_coarse_reach','A4_LVL_PRIMARY_FIG':'a4_lvl_depth_effect','A4_OVR_PRIMARY_FIG':'a4_ovr_overlap_effect','A4_STS_PRIMARY_FIG':'a4_sts_space_time_surface','INT_COMPONENT_MEMORY_FIG':'integrated_component_memory','INT_ROUTING_SWITCH_FIG':'integrated_routing_switch','INT_ROUTING_STABILITY_FIG':'integrated_routing_stability','INT_QUOTA_FRONTIER_FIG':'integrated_quota_frontier','INT_MM_PARETO_FIG':'integrated_moving_mnist_pareto','INT_TAXIBJ_PARETO_FIG':'integrated_taxibj_pareto'}
        if key in names: row['expected_file_path']=f"figures/dissertation_v2/{names[key]}.svg"
        row['generator_mode'] = 'table (one metric/population per table)' if 'TABLE' in key or 'PROFILE' in key else 'claim' if 'CLAIM' in key or 'SYNTHESIS' in key else 'figure'
        if row['generator_mode']=='claim': row['generator']='scripts/insert_dissertation_v2_results.py (shared-key insertion with accepted bundle)'
        if row['generator_mode'].startswith('table'): row['generator']='scripts/plot_dissertation_v2_results.py --kind table'
        row['generator_sha256']=digest((ROOT/('scripts/insert_dissertation_v2_results.py' if row['generator_mode']=='claim' else 'scripts/make_dissertation_v2_assets.py' if calibration else 'scripts/plot_dissertation_v2_results.py')).read_bytes())
        if calibration:
            row["expected_file_path"] = "figures/dissertation_v2/a4_calibration_30k.svg"
            replacement = "![Why was 30k selected?](../../../figures/dissertation_v2/a4_calibration_30k.svg)\n\nFigure D4-calibration — Horizon-4 foreground MSE against optimizer updates for the two seed-0 calibration configurations. Lower is better; points are saved 2k validations, not independent seed replicates. Source: hash-verified archived calibration logs and analysis; observed-range criterion only."
        elif key == "INT_GATE_SUMMARY":
            replacement = "The recorded CPU checks establish the bounded quota, coordinate, core inverse/parity, shape, checkpoint, and OpenSTL interface contracts listed in Appendix Ж.v2.1. Dynamic full-model parameter parity and deployed CUDA mixed-precision position/switch checks remain explicit gates. No GPU-memory or competitive-quality result is reported by this summary."
        else:
            replacement = f"> [RESULT PENDING: {key}; evidence={evidence}; source={source}; registry=refactor_v2/registries/placeholders.json. No empirical value inserted.]"
        for n in FILES:
            main_text[n] = main_text[n].replace("{{"+key+"}}", replacement)
    jwrite(EDIT / "registries/placeholders.json", registry)
    for name, text in main_text.items():
        write(MAN / name, text)
    tasks = [(1,"System / consistency","1","A3","conceptual synthesis; no pending empirical superiority"), (2,"Reversible temporal computation","2.2","A2","published results retained"), (3,"V-BLT / RoPE-ND / entropy","2.3","A1","fixed-patch evidence retained"), (4,"Retained context","3.1–3.6, 4.3–4.4","A4 in preparation","CLAIM_A4_SUMMARY"), (5,"Observability / statistics","3.5–3.6, 4.1, appendices В–Д","A4 in preparation","selected confirmation only"), (6,"Operating region / composition","4.7–4.11","A4 in preparation + legacy","CLAIM_INT_SUMMARY")]
    write(EDIT / "TASK_CLAIM_MAP.md", "# Task–chapter–conclusion–publication map\n\n| Task / conclusion number | Preserved scientific task | Main location | Publication | Claim boundary |\n|---:|---|---|---|---|\n" + "\n".join("| " + " | ".join(map(str,r)) + " |" for r in tasks) + "\n\nBoth abstract scientific blocks share one source. Every result-dependent summary slot uses the same registry key in front matter, novelty, Chapter 4, and conclusions. Published numerical tables are preserved, not replaced by integrated hypotheses.")
    write(EDIT / "RESULT_INSERTION_CONTRACT.md", """# Result insertion and figure contract

No pending row is a measured result. The registry includes positive, neutral, and negative text branches for every slot, including MEC, optional rollout, checkpointing, baselines, routing training/switch/stability, dataset-specific Pareto and sentence-level conclusions.

The accepted-row plotting interface requires status `accepted`, two explicit seeds, data/config/checkpoint hashes, metric units/direction, a fixed selection role, evidence status, and an input-file hash supplied on the command line. Result adapters must be checked against the finalizer's actual output; a path in the registry is not permission to accept a partial file. `all_final_development_metrics.json` is the actual Article 4 aggregate name; `selection_ledger.json` and `final_report.json` govern selected holdout evidence. No generic `aggregate.json` is treated as a real artifact.

Visual grammar: Flat dark gray, hierarchy blue, routed hierarchy orange, Full dark purple. Use two explicit seed markers and a mean, not n=2 Gaussian/SEM/t intervals. FRT uses exact numeric budgets, ALC horizontal policy effects, CEX actual coarse positions with compound-intervention caption, LVL/OVR paired dots, STS mean plus both seed surfaces. Confirmation and MEC are separate figures. Memory allocated/reserved, retained state, and throughput are separate views. MM/TaxiBJ plots stay separate. Qualitative strips use locked indices only after aggregate evidence.

Before release: accept the final artifacts; resolve dependencies; generate plots/tables and their exact JSON rows; select interpretation from measured results; update shared summary claims; normalize language; freeze numbering once; regenerate cross-reference/count maps; complete institutional/permission fields; render and audit the release document. An absent optional comparison contracts the corresponding claim rather than acquiring invented success values.
""")
    from insert_dissertation_v2_results import reapply
    reapply()
    print(f"Registered {len(registry)} evidence slots and {len(rows)} training rows")


if __name__ == "__main__":
    main()
