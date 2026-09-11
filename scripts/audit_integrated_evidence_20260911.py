"""Audit completed integrated evidence without selecting or training a model."""
from pathlib import Path
import hashlib,json,math,statistics,shutil
ROOT=Path(__file__).resolve().parents[1]
RUNS=Path('D:/hierarchical-video-prediction-storage/dissertation_remaining_2026-09-07')
OUT=ROOT/'docs/reviews/integrated_evidence_2026-09-11'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    sources={};quality=[];profiles=[]
    def copy(p,rel):
        q=OUT/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        assert sha(q)==sha(p);sources[str(p)]={'sha256':sha(p),'copy':q.relative_to(ROOT).as_posix()}
    assert read(RUNS/'result.json')['status']=='completed'
    for f in ['result.json','integrated_campaign.json','launch_authorization.json','integrated_preflight.json']:
        copy(RUNS/f,f)
    for rid in ['INT_MM_S00','INT_MM_S01','INT_TB_S00','INT_TB_S01']:
        p=RUNS/'confirmations'/rid;r=read(p/'result.json');c=read(p/'resolved_config.json');v=read(p/'run_provenance.json')
        logs=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()]
        assert r['status']=='completed' and logs[-1]['event']=='worker_complete'
        assert logs[-1]['test_metrics']==r['test_metrics']
        vals=[x for x in logs if x['event']=='validation'];test=[x for x in logs if x['event']=='test_start']
        assert len(vals)==c['epoch'] and len(test)==1
        assert max(x.get('step',0) for x in logs)==v['planned_optimizer_steps']
        assert min(x['val_loss'] for x in vals)==r['best_validation_loss']
        assert test[0]['checkpoint']==r['best_checkpoint']
        assert sha(Path(v['config_file']))==v['config_sha256']
        for x in logs:
            assert all(math.isfinite(z) for z in x.values() if isinstance(z,float))
        m=r['test_metrics'];pixels=math.prod(c['in_shape'][1:]);h=c['aft_seq_length']
        for key in ['mse','mae']:
            assert math.isclose(m[key],m['pixel_'+key]*pixels,rel_tol=1e-10)
            assert math.isclose(m[key],statistics.mean(m['horizon_'+key]),rel_tol=1e-10)
        assert all(math.isfinite(z) for z in m.values() if isinstance(z,(float,int)))
        assert m['samples_times_frames']==(100000 if c['dataname']=='mmnist' else 2000)
        selected=min(vals,key=lambda x:x['val_loss'])
        for f in ['result.json','resolved_config.json','run_provenance.json','metrics.jsonl','status.json']:
            copy(p/f,f'quality/{rid}/{f}')
        copy(p/'panels/locked_test.png',f'quality/{rid}/original_locked_test.png')
        quality.append(dict(run=rid,result=r,checkpoint_sha256=sha(Path(r['best_checkpoint'])),
            selected_epoch_zero_based=selected['epoch'],steps=v['planned_optimizer_steps'],epochs=len(vals),
            test_events=len(test),test_clips=m['samples_times_frames']//h))
    base=RUNS/'profiles_corrected_20260909';pr=read(base/'result.json');assert pr['status']=='completed'
    copy(base/'result.json','profiles/result.json');copy(base/'source.json','profiles/source.json')
    copy(base/'profile_integrated_openstl_memory.py','profiles/profile_integrated_openstl_memory.py')
    for item in pr['profiles']:
        p=Path(item['path']);assert sha(p)==item['sha256'];r=read(p)
        assert r['schema_version']==3 and r['diagnostic_retention_guard'] and r['parity']['passed']
        assert r['profiler_sha256']==sha(base/'profile_integrated_openstl_memory.py')
        assert len(r['cases'])==18
        for c in r['cases']:
            assert len(c['repeats'])==5 and c['scope']=='whole_model'
            for z in c['repeats']:
                assert all(math.isfinite(v) for v in z.values() if isinstance(v,(float,int)))
                assert z['absolute_peak_allocated_bytes']>=z['baseline_allocated_bytes']
            c['summary']={k:statistics.median(z[k] for z in c['repeats']) for k in c['repeats'][0]}
        copy(p,f'profiles/{p.name}');profiles.append(r)
    result=dict(status='verified',date='2026-09-11',quality=quality,profiles=profiles,sources=sources,
        resource_scope='72 whole-model cases verified; 72 core-operation slots remain unmeasured',
        metric_contract='Unclipped pixel/frame MSE and MAE in normalized loader units; PSNR/SSIM clipped to [0,1]; frame MSE sums C*H*W.',
        limitations=['No matched fully trained comparator','No per-clip test errors retained; no bootstrap intervals inferred','Synthetic initial-probe resource profiles, not trained checkpoints; cold AdamW, unlike full training Adam','Five timing repeats; substantial TaxiBJ timing variability'],
        auditor_sha256=sha(Path(__file__)))
    write(OUT/'audit.json',result)
    print(json.dumps({'quality_runs':len(quality),'profile_cases':sum(len(p['cases']) for p in profiles),'training_hours':sum(q['result']['elapsed_hours'] for q in quality)}))
if __name__=='__main__':main()
