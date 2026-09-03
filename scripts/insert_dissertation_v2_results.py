"""Insert accepted artifacts/shared claims at every occurrence of a registered key.

No experiment is run or automatically accepted. Bundle acceptance is a separate
editorial decision; hashes/units/seed completeness are mechanically enforced here.
Saved insertions reapply after assembly only if all original hashes still match.
"""
import argparse
import json
import re
from pathlib import Path
from refactor_dissertation_v2 import ROOT, EDIT, MAN, FILES, digest, jwrite, write
from plot_dissertation_v2_results import validate


def validate_insertion_bundle(bundle,path):
    if bundle.get('kind')!='claim':
        validate(bundle,path);return
    if bundle.get('status')!='accepted' or not bundle.get('evidence_bundles'):raise ValueError('shared claim requires accepted endpoint-specific evidence bundles')
    for ref in bundle['evidence_bundles']:
        source=Path(ref['path']);source=source if source.is_absolute() else path.parent/source
        if digest(source.read_bytes())!=ref['sha256']:raise ValueError('shared claim evidence changed')
        validate(json.loads(source.read_text(encoding='utf-8')),source)


def verify_record(record):
    bundle_path=Path(record['bundle'])
    if digest(bundle_path.read_bytes())!=record['bundle_sha256']: raise ValueError('accepted bundle changed')
    bundle=json.loads(bundle_path.read_text(encoding='utf-8'))
    validate_insertion_bundle(bundle,bundle_path)
    if bundle.get('placeholder_key')!=record['key']: raise ValueError('accepted bundle/key mismatch')
    if record.get('artifact'):
        artifact=Path(record['artifact'])
        if digest(artifact.read_bytes())!=record['artifact_sha256']: raise ValueError('generated artifact changed')
    return bundle


def reapply(records=None):
    path=EDIT/'registries/accepted_insertions.json'
    if records is None:
        if not path.exists(): return
        records=json.loads(path.read_text(encoding='utf-8'))
    slots=json.loads((EDIT/'registries/placeholders.json').read_text(encoding='utf-8'))
    for record in records:
        verify_record(record)
        found=[r for r in slots if r['key']==record['key']]
        if len(found)!=1: raise ValueError('unknown shared insertion key')
        row=found[0]
        for name in FILES:
            target=MAN/name
            text=target.read_text(encoding='utf-8')
            pattern=r'^> \[RESULT PENDING: '+re.escape(record['key'])+r';[^\n]*\]\s*$'
            text=re.sub(pattern,lambda _:record['replacement'],text,flags=re.M)
            write(target,text)
        row.update(status='accepted insertion',accepted_bundle_sha256=record['bundle_sha256'],
                   inserted_artifact_sha256=record.get('artifact_sha256'),interpretation=record.get('interpretation'))
    jwrite(EDIT/'registries/placeholders.json',slots)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reapply',action='store_true');p.add_argument('--key');p.add_argument('--bundle',type=Path)
    p.add_argument('--sha256');p.add_argument('--artifact',type=Path);a=p.parse_args()
    if a.reapply: reapply();return
    if not a.key or not a.bundle or not a.sha256: p.error('--key, --bundle and --sha256 are required')
    if digest(a.bundle.read_bytes())!=a.sha256: raise ValueError('input bundle hash mismatch')
    bundle=json.loads(a.bundle.read_text(encoding='utf-8'));validate_insertion_bundle(bundle,a.bundle)
    if bundle.get('placeholder_key')!=a.key: raise ValueError('bundle is not accepted for this key')
    record={'key':a.key,'bundle':str(a.bundle.resolve()),'bundle_sha256':a.sha256}
    if a.artifact:
        record.update(artifact=str(a.artifact.resolve()),artifact_sha256=digest(a.artifact.read_bytes()))
        sidecar=json.loads(a.artifact.with_suffix('.json').read_text(encoding='utf-8'))
        if sidecar.get('bundle_sha256')!=a.sha256: raise ValueError('artifact not generated from accepted bundle')
        if a.artifact.suffix=='.md': replacement=a.artifact.read_text(encoding='utf-8')
        elif a.artifact.suffix=='.svg':
            import os
            rel=os.path.relpath(a.artifact,MAN).replace('\\','/')
            replacement=f"![{bundle['metric']}]({rel})\n\n{bundle['population']}; {bundle['evidence_status']}; {bundle['metric']} [{bundle['units']}; {bundle['direction']} is better]. Both training seeds are shown; no seed interval inferred."
        else: raise ValueError('only verified Markdown tables or SVG figures are accepted')
    else:
        if bundle.get('interpretation') not in {'positive','neutral','negative'}: raise ValueError('claim requires an explicitly reviewed outcome branch')
        replacement=bundle.get('interpretation_text','')
        if not replacement.strip() or re.search(r'\{[^}]+\}|RESULT PENDING',replacement): raise ValueError('claim must be complete reviewed text, not a template')
        record['interpretation']=bundle['interpretation']
    record['replacement']=replacement
    saved=EDIT/'registries/accepted_insertions.json'
    records=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else []
    if any(r['key']==a.key for r in records): raise ValueError('immutable accepted key already exists; explicitly review replacement in version control')
    records.append(record);reapply(records);jwrite(saved,records)
    print(f'Inserted {a.key} at shared draft locations; acceptance recorded')


if __name__=='__main__':main()
