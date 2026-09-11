"""Record release checks against the exact rendered PDF and current sources."""
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,sys
import pymupdf
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/dissertation/condensed_translated_humanized_ukrainian'
OUT=ROOT/'docs/reviews/completed_evidence_2026-09-09/verification'
OUT.mkdir(parents=True,exist_ok=True)

def main():
    checks=[]
    for args in [['scripts/check_manuscript_finalization.py'],['scripts/check_text_consistency.py','--gate'],['scripts/check_rft_consistency.py'],['scripts/audit_ukrainian_dissertation.py'],['scripts/check_hierarchy_narrative.py']]:
        r=subprocess.run([sys.executable,*args],cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8'},capture_output=True,encoding='utf-8')
        checks.append(dict(command=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    r=subprocess.run([str(ROOT.parent/'hierarchical-video-prediction/venv/Scripts/python.exe'),'-m','pytest','tests/test_hierarchy_narrative_gate.py','-q'],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
    checks.append(dict(command=['pytest','tests/test_hierarchy_narrative_gate.py'],exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    r=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    checks.append(dict(command=['git','diff','--check'],exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    (OUT/'verification_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert all(r['exit_code']==0 for r in checks),'A release check failed; see verification_checks.json'
    build={k.replace('\\','/'):v for k,v in json.loads((ROOT/'tmp/ukrainian_pdf/source_hashes.json').read_text(encoding='utf-8')).items()}
    receipt={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'display_math_preserved':True,'source_hashes':{}}
    for p in sorted(SRC.glob('0*.md')):
        old=(ROOT/'tmp/hierarchy_narrative_20260906/before'/p.name).read_text(encoding='utf-8')
        new=p.read_text(encoding='utf-8')
        assert re.findall(r'\$\$(.*?)\$\$',old,re.S)==re.findall(r'\$\$(.*?)\$\$',new,re.S),p.name
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert build[p.relative_to(ROOT).as_posix()]==digest,p.name
        receipt['source_hashes'][p.name]=digest
    receipt['build_sources_match']=True
    live=ROOT.parent/'hierarchical-video-prediction-path-b'
    receipt['live_controller']=json.loads((live/'artifacts/v2/article4_primary_30k/campaign/state.json').read_text(encoding='utf-8'))
    receipt['live_git_status']=subprocess.run(['git','status','--short'],cwd=live,check=True,capture_output=True,text=True).stdout
    receipt['live_campaign_sha256']=hashlib.sha256((live/'experiments/v2/campaigns/article4_primary_30k_2026-09-01.json').read_bytes()).hexdigest()
    (OUT/'source_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    pdf_path=ROOT/'output/pdf/DISSERTATION_UKRAINIAN_RESULTS_2026-09-09.pdf'
    pdf=pymupdf.open(pdf_path);sha=hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    audit=json.loads((ROOT/'tmp/ukrainian_pdf/inspection/audit.json').read_text(encoding='utf-8'))
    assert audit['pdf_sha256']==sha,'Render inspection belongs to another PDF'
    assert audit['pages']==len(pdf)
    assert not audit['outside_page_blocks'] and not audit['near_blank_pages'] and not audit['replacement_characters']
    assert not any('evidence:' in p.get_text() or 'Таблиця 4.X' in p.get_text() for p in pdf)
    audit.update(pdf=pdf_path.relative_to(ROOT).as_posix(),bookmarks=len(pdf.get_toc()),links=sum(len(p.get_links()) for p in pdf))
    (OUT/'render_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'checks_passed':len(checks),'pages':len(pdf),'bookmarks':audit['bookmarks'],'links':audit['links'],'source_hashes_match':True,'live_worktree_clean':not receipt['live_git_status']}))
if __name__=='__main__':main()
