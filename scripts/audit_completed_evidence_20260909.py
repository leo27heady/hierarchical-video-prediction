"""Hash-verify completed A4 and bounded integrated evidence; never train or select."""
from pathlib import Path
import hashlib,json,shutil,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT.parent/'hierarchical-video-prediction-path-b'
A4=LIVE/'artifacts/v2/article4_primary_30k'
INT=Path('D:/hierarchical-video-prediction-storage/dissertation_remaining_2026-09-07')
OUT=ROOT/'docs/reviews/completed_evidence_2026-09-09'
sys.path.insert(0,str(ROOT))
from scripts.v2.analyze_article4_paired_v2 import paired,primary

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={}
    def copy(p,relative):
        target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(p);shutil.copy2(p,target);assert sha(target)==digest
        sources[str(p)]={'sha256':digest,'copy':str(target.relative_to(ROOT)).replace('\\','/')}
        return target
    campaign=LIVE/'experiments/v2/campaigns/article4_primary_30k_2026-09-01.json'
    state=read(A4/'campaign/state.json');final=read(A4/'post_training/final_report.json')
    assert state['status']==final['status']=='complete' and not state['failures']
    assert state['campaign_sha256']==final['campaign_sha256']==sha(campaign)
    for field in ['selection_ledger','confirmation_manifest','confirmation_audit']:
        p=LIVE/final[field];assert sha(p)==final[field+'_sha256'];copy(p,p.name)
    for name in ['all_final_development_metrics.json','final_report.json']:
        copy(A4/'post_training'/name,name)
    copy(A4/'dependency_resolution.json','dependency_resolution.json')
    ledger=read(OUT/'selection_ledger.json')
    rows=read(OUT/'all_final_development_metrics.json')['rows'];assert len(rows)==86
    assert len({r['config_hash'] for r in rows})==86
    for i,row in enumerate(rows):
        p=Path(row['checkpoint']);assert sha(p)==row['checkpoint_sha256'],p
        config=LIVE/row['config'];copy(config,'configs/'+config.name)
        run=p.parent.parent
        status=read(run/'status.json');assert status['status'] in ('complete','completed'),status
        metrics=[json.loads(x) for x in (run/'validation_metrics.jsonl').read_text().splitlines()]
        terminal=[m for m in metrics if int(m['step'])==30000];assert len(terminal)==1
        assert terminal[0]['horizon4_foreground_mse']==row['metrics']['horizon4_foreground_mse']
        assert np.isfinite(row['metrics']['horizon4_foreground_mse'])
        if i%20==0:print(f'checked checkpoint {i+1}/86',flush=True)
    evaluation={r['experiment_id']:r for r in final['evaluations']}
    arrays={}
    for run_id,record in evaluation.items():
        folder=A4/'post_training/evaluations'/f"{run_id}__{record['config_hash'][:12]}"
        p=copy(folder/'per_clip_horizon_metrics.npz','confirmation/'+run_id+'.npz')
        data=dict(np.load(p));ids,values=primary(data);assert len(ids)==500
        assert record['confirmation_manifest_sha256']==final['confirmation_manifest_sha256']
        arrays[run_id]=data
    roles={}
    for name,role in ledger['roles'].items():
        values=[]
        for r in role['runs']:
            data=arrays[r['experiment_id']];_,h4=primary(data)
            summary=evaluation[r['experiment_id']]
            values.append({'run_id':r['experiment_id'],'h4':float(h4.mean()),
                           'all_h_foreground_mse':float(data['foreground_mse'].mean()),
                           'reentry_foreground_mse':summary['reentry_foreground_mse'],
                           'reentry_records':summary['reentry_records'],
                           'attention_entropy':float(np.mean([v['normalized_entropy'] for v in summary['attention']['layers']]))})
        roles[name]={'policy':role['policy'],'budget':role['budget_tokens'],'seeds':values}
    comparisons={}
    for name in ['best_hierarchy_1792','memory_pareto_within_5pct']:
        comparisons[name]=[paired(arrays[ledger['roles']['best_flat_1792']['runs'][seed]['experiment_id']],
                                 arrays[ledger['roles'][name]['runs'][seed]['experiment_id']],4) for seed in (0,1)]
    val=read(INT/'validation300/result.json');assert val['status']=='completed' and len(val['runs'])<=6
    copy(INT/'validation300/result.json','validation300.json');copy(INT/'validation300/contract.json','validation300_contract.json')
    for r in val['runs']:
        data=dict(np.load(copy(INT/'validation300'/r['run_id']/'per_clip_horizon_metrics.npz','validation300/'+r['run_id']+'.npz')))
        assert len(np.unique(data['clip_id']))==300
        assert np.allclose(data['foreground_mse'][:,3].mean(),r['horizon4_foreground_mse'])
    profiles={};resource=[]
    for dataset in ('mmnist','taxibj'):
        for precision in ('fp32','bf16'):
            path=copy(INT/f'profiles/{dataset}_{precision}.json',f'profiles/{dataset}_{precision}.json')
            p=read(path);assert p['parity']['passed'] and len(p['cases'])==18 and len(p['core_only'])==6
            profiles[f'{dataset}_{precision}']=p
            for c in p['cases']:
                assert len(c['repeats'])==5 and all(np.isfinite(v['loss']) for v in c['repeats'])
                if precision=='bf16' and c['training'] and c['hierarchy']=='hierarchical' and c['routing']=='hybrid':
                    resource.append({'dataset':dataset,'mode':c['execution_mode'],'batch':p['batch'],
                        'allocated_gib':statistics.median(v['absolute_peak_allocated_bytes'] for v in c['repeats'])/2**30,
                        'reserved_gib':statistics.median(v['absolute_peak_reserved_bytes'] for v in c['repeats'])/2**30,
                        'incremental_gib':statistics.median(v['incremental_peak_allocated_bytes'] for v in c['repeats'])/2**30,
                        'seconds':c['median_seconds'],'packed_dtype':c['packed_state']['tokens_dtype']})
    screens={}
    for name in ['SCREEN_MM_FULL','SCREEN_TB_FULL','SCREEN_MM_FLAT','SCREEN_MM_STATIC']:
        folder=INT/'screens'/name;result=read(copy(folder/'result.json',f'screens/{name}/result.json'))
        assert result['status']=='completed' and result['test_evaluated'] is False and result['steps']==200
        copy(folder/'run_provenance.json',f'screens/{name}/run_provenance.json')
        events=[json.loads(x) for x in copy(folder/'metrics.jsonl',f'screens/{name}/metrics.jsonl').read_text().splitlines()]
        assert not any(r.get('event')=='test_start' for r in events)
        progress=[r for r in events if r.get('event')=='train_progress']
        result.update(updates_per_second=statistics.median(r['updates_per_second'] for r in progress[1:]),
                      last_grad_norm=progress[-1]['train_grad_norm'])
        screens[name]=result
    audit={'a4_runs':86,'roles':roles,'comparisons':comparisons,'resource':resource,'screens':screens,
           'resource_accepted':False,
           'resource_exclusion':'schema-2 saved-tensor pack hook retains original Tensor grad_fn; growing allocated baselines contaminate whole-model peaks; CPU weak-reference reproduction confirms cycle; require schema-3 remeasurement',
           'validation300':val['runs'],'sources':sources,'analysis_class':'descriptive; paired intervals adjusted across four contrasts; not a noninferiority test',
           'auditor_sha256':sha(Path(__file__))}
    write(OUT/'audit.json',audit)
    print(json.dumps({k:audit[k] for k in ['roles','comparisons','resource','screens']},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
