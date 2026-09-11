"""Export fixed first test clips from the audited validation-selected checkpoints."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
RUNS=Path('D:/hierarchical-video-prediction-storage/dissertation_remaining_2026-09-07')
sys.path.insert(0,str(RUNS/'source/openstl'))
from openstl.models import HVPIntegrated_Model
OUT=ROOT/'docs/reviews/integrated_evidence_2026-09-11/examples'
def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    for dataset,rid in [('mmnist','INT_MM_S00'),('taxibj','INT_TB_S00')]:
        p=RUNS/'confirmations'/rid;c=json.loads((p/'resolved_config.json').read_text());r=json.loads((p/'result.json').read_text())
        model=HVPIntegrated_Model(**c).cuda().eval()
        ckpt=torch.load(r['best_checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in ckpt['state_dict'].items()},strict=True)
        if dataset=='mmnist':
            a=np.load(RUNS/'data/moving_mnist/mnist_test_seq.npy',mmap_mode='r')
            x=np.asarray(a[:10,:8],dtype=np.float32).transpose(1,0,2,3)[:,:,None]/255
            y=np.asarray(a[10:20,:8],dtype=np.float32).transpose(1,0,2,3)[:,:,None]/255
        else:
            x=(np.array(np.load(RUNS/'data/taxibj/memmap_v1/X_test.npy',mmap_mode='r')[:16])+1)*.5
            y=(np.array(np.load(RUNS/'data/taxibj/memmap_v1/Y_test.npy',mmap_mode='r')[:16])+1)*.5
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(torch.from_numpy(x).cuda()).float().cpu().numpy()
        assert np.isfinite(pred).all()
        np.savez_compressed(OUT/f'{dataset}.npz',inputs=x[:1],targets=y[:1],predictions=pred[:1])
        (OUT/f'{dataset}.json').write_text(json.dumps(dict(run=rid,test_index=0,batch=len(x),precision='bf16-mixed',checkpoint=r['best_checkpoint'],epoch=ckpt['epoch'],global_step=ckpt['global_step'],note='First fixed test clip, no ranking or cherry-picking; qualitative re-evaluation only'),indent=2)+'\n')
        print(dataset,'exported',pred.shape,flush=True)
if __name__=='__main__':main()
