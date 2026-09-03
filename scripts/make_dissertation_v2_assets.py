"""Small, code-native, single-question SVGs; no GPU and no invented result points."""
from __future__ import annotations
import html
import json
import math
import subprocess
import textwrap
from pathlib import Path
from refactor_dissertation_v2 import ROOT, FIG, EDIT, BASE, MAN, digest, write, jwrite

COLORS = {"flat": "#444b55", "hierarchy": "#2675ad", "routing": "#c66a13", "full": "#5b326f"}


def svg_start(title, height=470):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}"><title>{html.escape(title)}</title><rect width="1100" height="{height}" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#202b36;font-size:18px} .title{font-size:25px;font-weight:bold} .small{font-size:15px}</style>',
            f'<text x="35" y="40" class="title">{html.escape(title)}</text>']


def box(out, x, y, w, h, title, body, color):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#f6f8fb" stroke="{color}" stroke-width="3"/>')
    lines = [title] + textwrap.wrap(body, max(20, int(w / 9.5)))
    for n, line in enumerate(lines):
        out.append(f'<text x="{x+15}" y="{y+30+n*25}" style="font-weight:{"bold" if n==0 else "normal"}">{html.escape(line)}</text>')


def flow(name, title, boxes, footer):
    out = svg_start(title, 400)
    for i, (heading, body, color) in enumerate(boxes):
        box(out, 35 + i * 357, 88, 316, 230, heading, body, color)
        if i < len(boxes)-1 and name != 'context_design_axes':
            out.append(f'<path d="M {357+i*357} 195 h 27 l -8 -5 m 8 5 l -8 5" fill="none" stroke="#555" stroke-width="2"/>')
    for i, line in enumerate(textwrap.wrap(footer, 115)):
        out.append(f'<text class="small" x="35" y="{352+i*20}">{html.escape(line)}</text>')
    out.append('</svg>')
    write(FIG / f"{name}.svg", ''.join(out))
    jwrite(FIG / f"{name}.json", {"kind": "conceptual_not_empirical", "question": title, "boxes": boxes, "limitation": footer})


def lineplot(name, title, groups, xlabel, ylabel, footer, x_ticks=None, log_y=False):
    out = svg_start(title, 560)
    xs = [r[0] for rows, _, _ in groups.values() for r in rows]
    ys = [r[1] for rows, _, _ in groups.values() for r in rows]
    xmin, xmax = min(xs), max(xs)
    transform = math.log10 if log_y else float
    ymin, ymax = (math.log10(min(ys)) - .08, math.log10(max(ys)) + .08) if log_y else (min(0, min(ys)), max(ys) * 1.08)
    if ymax == ymin:
        ymax += 1
    def x(v): return 110 + (v-xmin)/max(xmax-xmin, 1) * 900
    def y(v): return 420 - (transform(v)-ymin)/(ymax-ymin) * 300
    out.append('<path d="M110 100 V420 H1010" fill="none" stroke="#444"/>')
    for val in x_ticks or sorted(set(xs)):
        out.append(f'<text class="small" text-anchor="middle" x="{x(val)}" y="445">{val:g}</text>')
    for k in range(5):
        val = 10**(ymin + (ymax-ymin)*k/4) if log_y else ymin + (ymax-ymin)*k/4
        out.append(f'<path d="M110 {y(val)} H1010" stroke="#e5e8ed"/><text class="small" text-anchor="end" x="98" y="{y(val)+5}">{val:.3f}</text>')
    for idx, (label, (rows, color, dashed)) in enumerate(groups.items()):
        coords = ' '.join(f'{x(a)},{y(b)}' for a, b in rows)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dashed}"/>')
        for a, b in rows:
            out.append(f'<circle cx="{x(a)}" cy="{y(b)}" r="4.5" fill="{color}"/>')
        out.append(f'<text x="{120+idx*360}" y="82" style="fill:{color}">{html.escape(label)}</text>')
    out.append(f'<text x="500" y="480">{html.escape(xlabel)}</text><text transform="translate(27 370) rotate(-90)">{html.escape(ylabel)}</text>')
    for i, line in enumerate(textwrap.wrap(footer, 125)):
        out.append(f'<text class="small" x="35" y="{515+i*19}">{html.escape(line)}</text>')
    out.append('</svg>')
    write(FIG / f"{name}.svg", ''.join(out))


def dotplot(name, title, groups, footer):
    out = svg_start(title, 500)
    vals = [v for _, rows, _ in groups for r in rows for v in r[1:]]
    low, high = min(vals) - .01, max(vals) + .01
    def x(v): return 210+(v-low)/(high-low)*810
    out.append(f'<path d="M{x(0)} 100 V370" stroke="#888" stroke-dasharray="4 4"/>')
    for j, (label, rows, color) in enumerate(groups):
        yy = 145+j*115
        out.append(f'<text x="35" y="{yy+5}">{html.escape(label)}</text>')
        for i, row in enumerate(rows):
            seed, value, *interval = row
            yi = yy + (i-(len(rows)-1)/2)*16
            if interval:
                out.append(f'<path d="M{x(interval[0])} {yi} H{x(interval[1])}" stroke="{color}" stroke-width="2"/>')
            out.append(f'<circle cx="{x(value)}" cy="{yi}" r="5" fill="{color}"/><text class="small" x="{x(value)+8}" y="{yi-5}">s{seed}</text>')
    for k in range(6):
        value = low + (high-low)*k/5
        out.append(f'<text class="small" x="{x(value)}" y="400" text-anchor="middle">{value:+.3f}</text>')
    out.append('<text x="220" y="430">Δ foreground MSE [squared normalized intensity]; positive favors candidate</text>')
    for i, line in enumerate(textwrap.wrap(footer, 130)):
        out.append(f'<text class="small" x="35" y="{463+i*19}">{html.escape(line)}</text>')
    out.append('</svg>')
    write(FIG / f"{name}.svg", ''.join(out))


def main():
    H, R, F, G = COLORS["hierarchy"], COLORS["routing"], COLORS["full"], COLORS["flat"]
    definitions = [
        ("context_design_axes", "What changes when a context package changes?", [("Payload", "Count retained tokens and bytes explicitly. Keep target cost separate.", G), ("Age / reach", "Identify the oldest available observation. Nominal access is not successful retrieval.", H), ("Spatial detail", "Record cells at each age and level. Coarse coverage can lose fine identity.", R)], "Design coordinates, not a measured frontier. Longer retained history is not a longer forecast."),
        ("windowdit_contract", "Where are observations, targets, and trainable computation separated?", [("Frozen observations", "TRB frames to cached coarse/fine latents. Training-only channel normalization.", G), ("Manifest gather", "Exact ages and physical fine cells. Transfer the selected payload, not padded candidates.", H), ("WindowDiT", "Joint context + noisy four-frame target attention. Decode four predictions; no clean target in context.", F)], "Controlled allocation experiment. Frozen codec differs from the end-to-end OpenSTL encoder."),
        ("context_rollout_timeline", "When does a four-frame endpoint become a self-fed rollout?", [("Observed history", "Warm from ground truth. Every retained age precedes the prediction origin.", G), ("Four-frame output", "The v2 endpoint stops here. Evaluate the fourth-frame error separately from context age.", H), ("Optional next call", "Only in a rollout: update causal memory with predictions, advance by four, predict again.", R)], "Legacy or separately authorized rollout path. No backpropagation through the full forecast in the v2 endpoint."),
        ("integrated_architecture", "How can packing and internal reversibility coexist?", [("Ordinary boundary", "CNN fine/coarse pyramid. Score and gather K fine tokens; retain dense coarse coverage.", R), ("Reversible core", "Packed features + physical metadata. Additive coupling; ordinary and reversible modes share weights/function.", F), ("Ordinary decoder", "Future coarse queries attend to memory. Upsample residual and add the last observed frame.", H)], "Encoder/router/decoder remain outside the inverse. Dense candidates and boundary state still contribute to total memory."),
        ("physical_position_contract", "Why does changing packed order not change physical identity?", [("Physical identity", "Cell-centre x/y, signed time, cell scale, and level. Normalize by grid and protocol.", H), ("Gather together", "Feature and metadata follow the same selected index. Packed slot is not a coordinate.", R), ("Position + attention", "Fourier features and level embedding. Future queries use the same physical convention.", F)], "CPU fp32 permutation check is bounded evidence; deployed mixed-precision verification remains a separate gate."),
        ("reversible_replay", "What is reconstructed during reverse differentiation?", [("Recover state", "Retain the terminal paired state. Invert one coupling block from its output.", H), ("Replay local graph", "Recompute the deterministic local transition and its adjoint/parameter derivatives.", F), ("Accumulate / release", "Add gradients, release local intermediates, continue backward. Keep ordinary boundaries.", G)], "A storage schedule, not a prediction-quality claim. Recomputing has a runtime cost."),
        ("vblt_router_boundary", "Which evidence transfers from V-BLT to the router?", [("Published V-BLT", "Fixed local/global patches, RoPE-ND, categorical byte prediction and entropy diagnosis.", H), ("Motivated question", "Spatial complexity is uneven. Where should a finite detail quota be allocated?", G), ("Separate integration", "Residual/change/learned scores, physical Fourier metadata and a real gather. New evidence required.", R)], "No inherited router-quality or memory claim. Entropy was not the executed integrated route score."),
        ("decision_flow", "How should a resource-constrained predictor be chosen?", [("Locate bottleneck", "Missing past evidence? Lost identity? Training tape? Uneven spatial information?", G), ("Match intervention", "Context allocation; preserved fine detail; checkpoint/reversible execution; route/quota study.", H), ("Measure and decide", "Matched quality, memory, throughput and stability. Accept, narrow, or reject the proposed advantage.", F)], "A conditional decision rule. Component ratios from different resources and baselines never multiply.")]
    for row in definitions:
        flow(*row)
    out = svg_start("Four distinct memory terms, with distinct intervention points", 580)
    rows = [("Retained context/state", "Spatial/temporal allocation; quota changes. Routing at equal K changes membership, not count.", H),
            ("Internal activations", "Ordinary autograd, checkpointing or same-forward reversible core execution.", F),
            ("Parameters / optimizer", "Weights, gradients and optimizer state. Not removed by context packing or replay.", G),
            ("Workspace / boundaries", "Dense encoder candidates, decoder state, temporary kernels and allocator reserve.", R)]
    for i, (title, body, color) in enumerate(rows):
        box(out, 35, 75+i*115, 1020, 100, title, body, color)
    out.append('</svg>')
    write(FIG / "resource_interventions.svg", ''.join(out))
    jwrite(FIG / "resource_interventions.json", {"kind": "conceptual_not_empirical", "rows": rows})
    # Calibration is available and hash-verified, unlike incomplete live campaign results.
    report_path = ROOT / "artifacts/v2/analysis/article4_schedule_calibration_2026-09-01.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = next(v for v in report.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "metrics_path" in v[0])
    rows = []
    groups = {}
    for i, run in enumerate(runs):
        local = FIG / f"calibration_{run['policy']}.json"
        if local.exists():
            saved = json.loads(local.read_text(encoding="utf-8"))
            if saved["original_metrics_sha256"] != run["metrics_sha256"]:
                raise ValueError("calibration snapshot provenance changed")
            records = saved["records"]
        else:
            source = Path(run["metrics_path"])
            raw = source.read_bytes()
            if digest(raw) != run["metrics_sha256"]:
                raise ValueError("calibration log hash mismatch")
            records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            jwrite(local, {"original_metrics_sha256": digest(raw), "source": str(source), "config_hash": run["config_hash"], "records": records})
        series = [(r["step"], r["horizon4_foreground_mse"]) for r in records]
        groups[run["policy"] + " (seed 0)"] = (series, G if i == 0 else H, "" if i == 0 else "6 3")
        rows += [{"policy": run["policy"], "seed": 0, "step": x, "horizon4_foreground_mse": y, "config_hash": run["config_hash"]} for x, y in series]
    lineplot("a4_calibration_30k", "Why was the common 30k endpoint selected?", groups, "Optimizer updates", "H4 foreground MSE [normalized², log] ↓", "Calibration: two seed-0 configurations, not two training-seed replicates. All saved 2k points; observed-range criterion only.", [2000, 8000, 12000, 16000, 20000, 24000, 30000], log_y=True)
    jwrite(FIG / "a4_calibration_30k.json", {"source": str(report_path.relative_to(ROOT)), "source_sha256": digest(report_path.read_bytes()), "rows": rows})
    # Historical point transcription, explicitly not a reanalysis from raw clip arrays.
    source = subprocess.check_output(["git", "show", f"{BASE}:{(MAN/'05_CHAPTER_4.md').relative_to(ROOT).as_posix()}"], cwd=ROOT).decode("utf-8")
    import re
    block = source[source.index("**Таблиця 4.4"):source.index("### 4.5.2")]
    pairs = []
    for m in re.finditer(r"^\| ([0-4]) \| ([+−\d,]+) \[.*?\] \| ([+−\d,]+) \[", block, re.M):
        pairs.append((int(m[1]), float(m[2].replace(",", ".").replace("−", "-")), float(m[3].replace(",", ".").replace("−", "-"))))
    assert len(pairs) == 5
    groups_e4 = [("Diffusion", [(s, a) for s, a, b in pairs], H), ("Regression", [(s, b) for s, a, b in pairs], G)]
    dotplot("legacy_e4_seed_effects", "Does the package effect keep its sign across heads and seeds?", groups_e4, "Historical E4: positive favors full-resolution temporal anchors. Five training seeds; no seed-level confidence interval shown.")
    jwrite(FIG / "legacy_e4_seed_effects.json", {"kind": "mechanical_source_table_transcription", "baseline": BASE, "source_table": "4.4", "source_sha256": digest(block), "rows": pairs})
    e5 = [(0, -.0064, -.0095, -.0032), (1, .0074, .0038, .0112), (2, .0087, .0035, .0139)]
    dotplot("legacy_e5_late_effects", "Was the late self-fed hierarchy reversal reproduced?", [("Frames 61–256", e5, H)], "Historical E5a: positive favors hierarchy. Whiskers are 95% clip-conditional intervals for each trained model, not seed uncertainty.")
    jwrite(FIG / "legacy_e5_late_effects.json", {"kind": "source_paragraph_transcription", "baseline": BASE, "source_section": "4.7.1", "source_sha256": digest(source), "rows": e5})
    assets = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p.read_bytes()), "generator": "scripts/make_dissertation_v2_assets.py"} for p in sorted(FIG.glob("*")) if p.suffix in {".svg", ".json"}]
    jwrite(EDIT / "registries/generated_assets.json", assets)
    print(f"Generated {len(list(FIG.glob('*.svg')))} SVG assets with source rows and hashes")


if __name__ == "__main__":
    main()
