"""Plot an explicitly accepted, hash-addressed result table; never synthesize data.

Input JSON: status=accepted, evidence_status, metric, units, direction, population,
selection_role, source_artifacts[{path,sha256}], rows[{method,seed,x,value,
config_hash,checkpoint_sha256,data_sha256}], optional y_category for heatmaps.
One input bundle must contain one endpoint/population/units. This is an editorial
adapter interface, not an automatic promotion of live finalizer output.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def validate(bundle, path):
    required = {"evidence_status", "metric", "units", "direction", "population", "selection_role", "source_artifacts", "rows"}
    if bundle.get("status") != "accepted" or not required <= bundle.keys():
        raise ValueError("Only explicitly accepted, fully described evidence bundles can be plotted")
    if bundle["direction"] not in {"lower", "higher", "descriptive"} or not bundle["rows"]:
        raise ValueError("Metric direction and nonempty rows are required")
    for artifact in bundle["source_artifacts"]:
        source = Path(artifact["path"])
        source = source if source.is_absolute() else path.parent / source
        if hashlib.sha256(source.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("source artifact hash mismatch")
    if not bundle["source_artifacts"]:
        raise ValueError("at least one hash-verified source artifact is required")
    cells = defaultdict(set)
    for row in bundle["rows"]:
        if not math.isfinite(float(row["value"])) or row["seed"] not in [0,1]:
            raise ValueError("finite measured value and seed 0/1 required")
        for name in ["config_hash", "checkpoint_sha256", "data_sha256"]:
            value = row.get(name, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"missing SHA-256 field {name}")
        # Measured memory (Pareto x) can differ by seed. Pair by scientific cell.
        key = (row["method"], row.get("cell_id", row["x"]), row.get("y_category"))
        if row["seed"] in cells[key]:
            raise ValueError("duplicate seed/cell")
        cells[key].add(row["seed"])
    if any(seeds != {0,1} for seeds in cells.values()):
        raise ValueError("every result cell requires both training seeds")
    if bundle.get("expected_run_ids") is not None:
        actual = [r.get("experiment_id") for r in bundle["rows"]]
        if (len(actual) != len(set(actual)) and bundle.get('record_unit')!='run-layer') or set(actual) != set(bundle["expected_run_ids"]):
            raise ValueError("result family is incomplete or contains extra runs")
    if bundle.get('expected_rows_count') is not None and len(bundle['rows'])!=bundle['expected_rows_count']:
        raise ValueError('incomplete repeated-measure family')


def result_table(bundle):
    """One endpoint/population per table, with both seeds and no inferred uncertainty."""
    grouped = defaultdict(list)
    for r in bundle["rows"]:
        grouped[(r["method"], r.get("cell_id", r["x"]))].append(r)
    lines = [f"{bundle['population']} — {bundle['evidence_status']}", "",
             f"Metric: {bundle['metric']} [{bundle['units']}; {bundle['direction']} is better].",
             "", "| Method / cell | Seed 0 | Seed 1 | Arithmetic mean |", "|---|---:|---:|---:|"]
    for (method, cell), rr in grouped.items():
        values = {r['seed']: float(r['value']) for r in rr}
        label=(method+' / '+str(rr[0].get('cell_label',cell))).replace('|','&#124;').replace('\n',' ')
        lines.append(f"| {label} | {values[0]:.6g} | {values[1]:.6g} | {(values[0]+values[1])/2:.6g} |")
    lines += ["", "Uncertainty unit: two independent training outcomes shown explicitly; no training-seed interval inferred.",
              f"Selection role: {bundle['selection_role']}. Exact source rows and hashes accompany this table."]
    return "\n".join(lines)+"\n"


def method_color(name):
    """Stable across input ordering; a method keeps its semantic color."""
    n = name.lower().replace('_', '-').replace(' ', '-')
    if n.startswith('flat'): return '#444b55'
    if n == 'full': return '#5b326f'
    if any(s in n for s in ['dynamic', 'hybrid', 'density']): return '#c66a13'
    colors = {'linear-age':'#2675ad', 'geometric-age':'#164568', 'fine-heavy':'#42999a',
              'balanced-recent':'#5366a4', 'coarse-heavy':'#8a6a44', 'reach-optimized':'#6a7890'}
    return colors.get(n, '#2675ad')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--kind", choices=["curve", "dots", "heatmap", "pareto", "table"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xlabel", default="Registered design coordinate")
    args = parser.parse_args()
    raw = args.bundle.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise SystemExit("input bundle SHA-256 mismatch")
    bundle = json.loads(raw)
    validate(bundle, args.bundle)
    if args.kind == "table":
        if args.output.suffix != '.md': raise ValueError('result tables require .md output')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result_table(bundle),encoding='utf-8')
        args.output.with_suffix('.json').write_text(json.dumps({'bundle_sha256':args.sha256,'table_rows':bundle['rows']},indent=2)+'\n',encoding='utf-8')
        print(args.output)
        return
    # Import plotting only after all data acceptance gates pass. CPU only.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = bundle["rows"]
    names = list(dict.fromkeys(r["method"] for r in rows))
    if args.kind == 'dots':
        names.sort(key=lambda n:sum(r['value'] for r in rows if r['method']==n)/sum(r['method']==n for r in rows))
    colors = {n: method_color(n) for n in names}
    direction='descriptive; no better/worse direction' if bundle['direction']=='descriptive' else bundle['direction']+' is better'
    label = f"{bundle['metric']} [{bundle['units']}; {direction}]"
    if args.kind == "heatmap":
        fig, axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        xs = sorted(set(r["x"] for r in rows)); ys = list(dict.fromkeys(r["y_category"] for r in rows))
        arrays = []
        for seed in [0,1]:
            a = np.full((len(ys),len(xs)),np.nan)
            for r in rows:
                if r["seed"] == seed: a[ys.index(r["y_category"]),xs.index(r["x"])] = r["value"]
            if np.isnan(a).any(): raise ValueError("incomplete response surface")
            arrays.append(a)
        vmin=min(a.min() for a in arrays); vmax=max(a.max() for a in arrays)
        for ax, a, title in zip(axes,[sum(arrays)/2,*arrays],["Arithmetic mean","Training seed 0","Training seed 1"]):
            im=ax.imshow(a,cmap="cividis",vmin=vmin,vmax=vmax)
            ax.set_xticks(range(len(xs)),xs); ax.set_yticks(range(len(ys)),ys); ax.set_title(title); ax.set_xlabel(args.xlabel)
        fig.colorbar(im,ax=axes,label=label,shrink=.7)
    else:
        fig, ax = plt.subplots(figsize=(9,5),constrained_layout=True)
        for name in names:
            rr=[r for r in rows if r["method"]==name]
            for seed, marker in [(0,"o"),(1,"^")]:
                rseed=sorted([r for r in rr if r["seed"]==seed],key=lambda r:r["x"])
                if args.kind=="dots":
                    ax.scatter([r["value"] for r in rseed],[names.index(name)+(seed-.5)*.12 for _ in rseed],color=colors[name],marker=marker,label=f"{name}, s{seed}")
                else:
                    ax.scatter([r["x"] for r in rseed],[r["value"] for r in rseed],color=colors[name],marker=marker,label=f"{name}, s{seed}")
            if args.kind=="curve":
                xx=sorted(set(r["x"] for r in rr))
                ax.plot(xx,[sum(r["value"] for r in rr if r["x"]==x)/2 for x in xx],color=colors[name],linewidth=1.4)
        if args.kind=='curve': ax.set_xticks(sorted(set(r['x'] for r in rows)))
        if args.kind=="dots":
            ax.set_yticks(range(len(names)),names); ax.set_xlabel(label)
        else:
            ax.set_xlabel(args.xlabel); ax.set_ylabel(label)
        ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
    fig.suptitle(bundle["population"] + " — " + bundle["evidence_status"],fontsize=11)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({"bundle_sha256":args.sha256,"plotted_rows":rows,"caption_contract":{k:bundle[k] for k in ["metric","units","direction","population","evidence_status","selection_role"]}},indent=2)+"\n",encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
