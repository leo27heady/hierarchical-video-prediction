"""Read a finalized A4 bundle through an explicit hash-bound editorial acceptance.

Acceptance JSON: status=accepted_for_dissertation, artifacts mapping names
aggregate/resolution/selection/final_report/data_manifest to {path,sha256}.
No protected arrays, checkpoint tensors, CUDA, or campaign writes are accessed.
The acceptance binds the data manifest, not a fresh expensive dataset rehash.
"""
import argparse
import json
from pathlib import Path
from refactor_dissertation_v2 import ROOT, EDIT, digest, jwrite
from dissertation_evidence_contracts import logical_family
from plot_dissertation_v2_results import validate


EXPECTED={'FRT':36,'ALC':14,'CEX':12,'LVL':6,'OVR':6,'STS':24}


def validate_aggregate(aggregate,resolution,matrix):
    campaign=matrix['source_campaign_sha256']
    if aggregate.get('run_count')!=86 or len(aggregate.get('rows',[]))!=86: raise ValueError('complete 86-run final aggregate required')
    if aggregate.get('campaign_sha256')!=campaign or resolution.get('parent_campaign_sha256')!=campaign: raise ValueError('campaign binding mismatch')
    if resolution.get('status')!='resolved' or resolution.get('selection_checkpoint')!=30000: raise ValueError('locked 30k dependency resolution required')
    known={r['experiment_id']:r['config_hash'] for r in matrix['runs'] if r.get('config_hash')}
    deps=resolution['run_order']
    if len(deps)!=6 or len({r['experiment_id'] for r in deps})!=6: raise ValueError('six unique resolved runs required')
    known.update({r['experiment_id']:r['config_hash'] for r in deps})
    ids=[r['experiment_id'] for r in aggregate['rows']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(known): raise ValueError('aggregate run membership mismatch')
    for r in aggregate['rows']:
        if r['step']!=30000 or r['config_hash']!=known[r['experiment_id']]: raise ValueError('stale config or non-final endpoint')
        if r['spec'].get('experiment_id')!=r['experiment_id'] or digest(json.dumps(r['spec'],sort_keys=True,separators=(',',':')))!=r['config_hash']:
            raise ValueError('nested canonical specification is not bound to its declared hash/identity')
        if 'horizon4_foreground_mse' not in r['metrics']: raise ValueError('wrong development endpoint')
        if r['spec']['seed'] not in [0,1]: raise ValueError('unexpected seed')


def read_acceptance(path,sha):
    if digest(path.read_bytes())!=sha: raise ValueError('acceptance SHA-256 mismatch')
    accept=json.loads(path.read_text(encoding='utf-8'))
    if accept.get('status')!='accepted_for_dissertation': raise ValueError('explicit editorial acceptance required')
    data={};sources=[]
    for name in ['aggregate','resolution','selection','final_report','data_manifest']:
        a=accept['artifacts'][name];p=Path(a['path']);p=p if p.is_absolute() else path.parent/p
        if digest(p.read_bytes())!=a['sha256']: raise ValueError(f'stale {name} artifact')
        data[name]=json.loads(p.read_text(encoding='utf-8'));sources.append({'path':str(p.resolve()),'sha256':a['sha256']})
    sources.append({'path':str(path.resolve()),'sha256':sha})
    return accept,data,sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('acceptance',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--family',choices=[*EXPECTED,'CONFIRMATION','MEC'],required=True)
    p.add_argument('--metric',default='horizon4_foreground_mse');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key',help='Optional registered figure/table key for the same family');a=p.parse_args()
    accept,data,sources=read_acceptance(a.acceptance,a.sha256)
    matrix=json.loads((EDIT/'registries/experiment_matrix.json').read_text(encoding='utf-8'))
    validate_aggregate(data['aggregate'],data['resolution'],matrix)
    ledger=data['selection'];report=data['final_report']
    if ledger.get('status')!='locked_from_development_validation' or ledger.get('primary_metric')!='horizon4_foreground_mse' or ledger.get('checkpoint_step')!=30000: raise ValueError('selection ledger not locked at declared endpoint')
    if report.get('status')!='complete' or report['selection_ledger_sha256']!=accept['artifacts']['selection']['sha256']: raise ValueError('finalizer or selection binding incomplete')
    if ledger['dependency_resolution_sha256']!=accept['artifacts']['resolution']['sha256']: raise ValueError('dependency/selection binding mismatch')
    if any(v.get('campaign_sha256')!=matrix['source_campaign_sha256'] for v in [report,ledger]): raise ValueError('selected campaign mismatch')
    allrows=[{**r,'family':r['spec']['family'],'budget':r['spec']['budget_tokens'],'policy':r['spec']['policy'],'seed':r['spec']['seed']} for r in data['aggregate']['rows']]
    lookup={r['experiment_id']:r for r in allrows}
    output=[]
    if a.family=='MEC':
        if a.metric not in {'normalized_entropy','source_mass:context_top','source_mass:context_coarse','source_mass:context_fine','source_mass:target_coarse','source_mass:target_fine'}: raise ValueError('declare one actual attention probe endpoint')
        required={r['experiment_id'] for role,v in ledger['roles'].items() if role!='memory_pareto_within_5pct' for r in v['runs']}
        selected=[r for r in report['evaluations'] if r['experiment_id'] in required]
        if len(selected)!=len(required) or {r['experiment_id'] for r in selected}!=required:raise ValueError('mechanism role union incomplete')
        for r in selected:
            base=lookup[r['experiment_id']];probe=r['attention']
            if probe['probe_clip']!=0 or probe['probe_diffusion_time']!=.5:raise ValueError('wrong diagnostic probe geometry')
            if r['config_hash']!=base['config_hash'] or r['checkpoint_sha256']!=base['checkpoint_sha256']:raise ValueError('probe checkpoint mismatch')
            if {l['layer'] for l in probe['layers']}!=set(range(base['spec']['depth'])):raise ValueError('probe layers incomplete')
            for layer in probe['layers']:
                value=layer['normalized_entropy'] if a.metric=='normalized_entropy' else layer['source_mass'][a.metric.split(':')[1]]
                output.append({'method':base['policy'].replace('_','-'),'cell_id':base['policy']+':layer'+str(layer['layer']),
                               'x':layer['layer'],'seed':base['seed'],'value':value,'experiment_id':r['experiment_id'],
                               'config_hash':r['config_hash'],'checkpoint_sha256':r['checkpoint_sha256'],'data_sha256':r['confirmation_manifest_sha256']})
        status='mechanism diagnostic';population='One fixed clip 0 / direct denoiser time 0.5; target-query attention averaged over heads; not a quality or causal effect'
        key='A4_MEC_DIAGNOSTIC_FIG'
    elif a.family=='CONFIRMATION':
        if a.metric not in {'mean_foreground_mse','reentry_foreground_mse','non_reentry_foreground_mse'}: raise ValueError('confirmation summary is not development horizon4')
        required={r['experiment_id'] for role,v in ledger['roles'].items() if role!='failure_hierarchy_1792' for r in v['runs']}
        selected=[r for r in report['evaluations'] if r['experiment_id'] in required]
        if len(selected)!=len(required) or {r['experiment_id'] for r in selected}!=required: raise ValueError('selected-role confirmation incomplete')
        for r in selected:
            base=lookup[r['experiment_id']]
            if r.get('status')!='complete' or r['config_hash']!=base['config_hash'] or r['checkpoint_sha256']!=base['checkpoint_sha256']: raise ValueError('selected checkpoint mismatch')
            if r['confirmation_manifest_sha256']!=report['confirmation_manifest_sha256']: raise ValueError('confirmation population hash mismatch')
            if r[a.metric] is None: raise ValueError('empty endpoint population cannot be represented as zero')
            output.append({'method':base['policy']+f" / B{base['budget']} / "+base['spec']['allocation_kind'],
                           'cell_id':base['spec']['allocation_kind']+'|'+json.dumps(base['spec']['allocation_args'],sort_keys=True),
                           'seed':base['seed'],'x':base['budget'],'value':r[a.metric],
                           'experiment_id':r['experiment_id'],'config_hash':r['config_hash'],'checkpoint_sha256':r['checkpoint_sha256'],
                           'data_sha256':r['confirmation_manifest_sha256']})
        status='sealed-holdout confirmation of selected roles only'
        population='500 fresh TRB clips × four frames; common 96/28 reference; '+('all frames' if a.metric=='mean_foreground_mse' else 'flagged re-entry frames' if a.metric=='reentry_foreground_mse' else 'non-re-entry frames')
        key='A4_CONFIRMATION_TABLE'
    else:
        if a.metric!='horizon4_foreground_mse': raise ValueError('primary development family requires horizon4_foreground_mse')
        selected=logical_family(allrows,a.family,data['resolution'])
        if len(selected)!=EXPECTED[a.family]: raise ValueError('logical family completeness failed')
        for r in selected:
            spec=r['spec'];args=spec['allocation_args'];x=r['budget'];method=r['policy'].replace('_','-')
            if spec['dataset_manifest_sha256']!=accept['artifacts']['data_manifest']['sha256']:raise ValueError('wrong development data-manifest binding')
            if a.family=='CEX': x=args['coarse_times'];method='Coarse expansion'
            if a.family=='LVL': x=r['levels'];method=f"{r['levels']} levels"
            if a.family=='OVR': method=spec['overlap']
            if a.family=='STS': x=args['quota'];method=args['temporal_policy']
            if x is None: raise ValueError('unknown design-coordinate schema; inspect exact config rather than guessing')
            item={'method':method,'cell_id':spec['allocation_kind']+'|'+json.dumps(args,sort_keys=True),'cell_label':f'{a.family}: {x}',
                  'seed':r['seed'],'x':x,'value':r['metrics'][a.metric],
                  'experiment_id':r['experiment_id'],'config_hash':r['config_hash'],'checkpoint_sha256':r['checkpoint_sha256'],
                  'data_sha256':accept['artifacts']['data_manifest']['sha256']}
            if a.family=='STS':item['y_category']=args['temporal_policy']
            output.append(item)
        status='registered development estimate';population='TRB development validation; frame 4; fixed four batches of 16, target >0.5 foreground and batch-weighted reduction'
        key=f'A4_{a.family}_PRIMARY_FIG'
    bundle={'status':'accepted','placeholder_key':key,'evidence_status':status,'metric':a.metric,'units':'normalized mass/entropy' if a.family=='MEC' else 'squared normalized intensity',
            'direction':'descriptive' if a.family=='MEC' else 'lower','population':population,'selection_role':'locked selected roles only' if a.family in {'CONFIRMATION','MEC'} else 'development; final 30k; no test selection',
            'source_artifacts':sources,'rows':output,'expected_run_ids':[r['experiment_id'] for r in selected]}
    if a.family=='MEC':bundle.update(record_unit='run-layer',expected_rows_count=sum(lookup[r['experiment_id']]['spec']['depth'] for r in selected))
    if a.key:
        registered=json.loads((EDIT/'registries/placeholders.json').read_text(encoding='utf-8'))
        if a.key not in {r['key'] for r in registered} or not a.key.startswith(f'A4_{a.family}_'):raise ValueError('key must be registered for the same evidence family')
        bundle['placeholder_key']=a.key
    validate(bundle,a.output);jwrite(a.output,bundle);print(a.output)


if __name__=='__main__':main()
