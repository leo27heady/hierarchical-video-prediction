"""Pure-Python logical-family and prelock provenance contracts; never imports torch."""
import json
import subprocess
from pathlib import Path
from refactor_dissertation_v2 import ROOT, EDIT, digest, jwrite


def logical_family(rows, family, resolution=None):
    selected = [r for r in rows if r['family'] == family]
    alc = [r for r in rows if r['family'] == 'ALC' or (r['family'] == 'FRT' and r['budget'] == 1792)]
    if family == 'ALC': return alc
    if family not in {'LVL', 'OVR'}: return selected
    if resolution:
        keys = ['LVL_L1', 'LVL_L2'] if family == 'LVL' else ['OVR_nested']
        ids = {x for key in keys for x in resolution['reused_runs'][key]}
        return selected + [r for r in rows if r['experiment_id'] in ids]
    if family == 'LVL':
        selected += [r for r in alc if r.get('policy') == 'flat_recent']
    role = 'LVL_L2_selected_nonflat' if family == 'LVL' else 'OVR_nested_selected_feasible'
    return selected + [{'experiment_id':f'REUSED_ROLE_PENDING_{role}_S{s:02d}', 'config_hash':None,
                       'seed':s, 'family':family, 'selection_role':role,
                       'status':'resolve from locked dependency_resolution.json.reused_runs; not new training'} for s in [0,1]]


def integrated_snapshot():
    output = EDIT/'registries/integrated_source_snapshot.json'
    if output.exists():
        # A source snapshot never silently tracks a moving research checkout.
        snapshot=json.loads(output.read_text(encoding='utf-8'))
        jwrite(output,snapshot)  # Stable LF serialization; identities remain unchanged.
        return snapshot
    manifest_path = ROOT/'experiments/v2/campaigns/integrated_openstl_confirmation_2026-09-02.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    op = Path(manifest['openstl_root'])
    files = [manifest_path, ROOT/'scripts/v2/run_integrated_openstl_campaign.py',
             op/'openstl/models/hvp_integrated_model.py',op/'openstl/methods/hvp_integrated.py',
             op/'tests/test_models/test_hvp_integrated.py',
             op/'configs/mmnist/HVPIntegrated.py',op/'configs/taxibj/HVPIntegrated.py']
    heads = {}
    for label, folder in [('Path B',ROOT.parent/'hierarchical-video-prediction-path-b'),('OpenSTL-integrated',op)]:
        heads[label] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip()
    runs = [{**r,'config_file_sha256':digest(Path(r['config']).read_bytes()),
             'binding_status':'prelock file hash, not a finalized run/config/checkpoint identity'} for r in manifest['runs']]
    snapshot = {'date':'2026-09-03','status':manifest['stage'],'launch_authorized':manifest['launch_authorized'],
                'hvp_editorial_baseline':'cf3f661aa6e809dd30ca92f91e2d17ec257ac4b8',
                'source_heads':heads,'manifest_sha256':digest(manifest_path.read_bytes()),'runs':runs,
                'files':[{'path':str(p),'sha256':digest(p.read_bytes())} for p in files],
                'limitation':'Read-only editorial source snapshot. No launch or final experimental evidence lock. No checkpoint or trained-quality claim.'}
    jwrite(output,snapshot)
    return snapshot


def snapshot_markdown(snapshot):
    lines = ['#### Ж.v2.0 Inspected prelock source snapshot (3 September 2026)', '',
             'Status: `prelock_confirmation`; `launch_authorized=false`. These are source/configuration identities, not completed-run results or authorization.', '']
    for name, head in snapshot['source_heads'].items(): lines.append(f'- {name}: `{head}`.')
    lines += [f"- HVP editorial baseline: `{snapshot['hvp_editorial_baseline']}`.",
              f"- Integrated campaign manifest SHA-256: `{snapshot['manifest_sha256']}`.", '',
              '| Planned run | Dataset / seed | Config file SHA-256 (prelock) |','|---|---|---|']
    lines += [f"| {r['run_id']} | {r['dataset']} / {r['seed']} | `{r['config_file_sha256']}` |" for r in snapshot['runs']]
    lines += ['', 'Model, method adapter, test, controller and config file hashes:', '']
    lines += [f"- `{r['path']}` — `{r['sha256']}`." for r in snapshot['files']]
    lines += ['', 'Machine-readable source: `refactor_v2/registries/integrated_source_snapshot.json`. A later accepted evidence bundle must separately bind final runtime, checkpoint, dataset and test-access hashes. A changed external checkout does not silently update this snapshot.']
    return '\n'.join(lines)
