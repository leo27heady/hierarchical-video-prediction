"""Read-only ordered physical-plan aliases; does not rewrite campaign entries."""
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from hvp_hca.article4_campaign import load_campaign_run_spec


def fingerprint(plan, seed):
    physical=dict(top_ages=plan.top_ages,coarse_ages=plan.coarse_ages,
                  fine_age_cells=plan.fine_age_cells,overlap=plan.overlap,
                  fine_representation=plan.fine_representation,seed=seed)
    return hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    campaign=json.loads(args.campaign.read_text()); groups=defaultdict(list)
    for entry in campaign['run_order']:
        spec=load_campaign_run_spec(args.root/entry['config'])
        groups[fingerprint(spec.plan,spec.seed)].append(spec.experiment_id)
    result=dict(scope='registered wave-one ordered physical plans, including training seed',
                executions=len(campaign['run_order']),unique_physical_plans_with_seed=len(groups),
                aliases={key:ids for key,ids in groups.items() if len(ids)>1},
                note='Plan identity is not evidence that unrelated training recipes are interchangeable.',
                campaign_sha256=hashlib.sha256(args.campaign.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
