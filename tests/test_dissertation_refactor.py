"""CPU-only tests for editorial migration and fail-closed result input."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import insert_dissertation_v2_results as insertion
from plot_dissertation_v2_results import validate, result_table, method_color
from refactor_dissertation_v2 import section, replace_section
from dissertation_evidence_contracts import logical_family
from adapt_dissertation_a4_results import validate_aggregate, read_acceptance


class EditorialContracts(unittest.TestCase):
    def test_section_boundaries(self):
        text='# C\n\n## 3.1 A\n\n### 3.1.1 B\nx\n\n## 3.2 C\ny\n'
        self.assertIn('### 3.1.1',section(text,'3.1'))
        self.assertNotIn('## 3.2',section(text,'3.1'))
        self.assertIn('## 3.2',replace_section(text,'3.1','## 3.1 New\nz'))

    def test_pending_results_rejected(self):
        with self.assertRaises(ValueError): validate({'status':'pending'},Path('unused'))

    def test_seed_and_provenance_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'measured_fixture.txt'; path.write_text('test fixture only')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            row={'method':'Fixture','seed':0,'x':1,'value':.2,'config_hash':'a'*64,'checkpoint_sha256':'b'*64,'data_sha256':'c'*64}
            bundle={'status':'accepted','evidence_status':'test fixture','metric':'fixture','units':'unit','direction':'lower','population':'not research data','selection_role':'test','source_artifacts':[{'path':str(path),'sha256':sha}],'rows':[row]}
            with self.assertRaises(ValueError): validate(bundle,path)
            bundle['rows'].append({**row,'seed':1,'value':.3})
            validate(bundle,path)
            self.assertIn('0.25',result_table(bundle))
            for r in bundle['rows']:r['cell_id']='allocation|quota'
            self.assertIn('&#124;',result_table(bundle))
            for r in bundle['rows']:r['cell_id']='same-config';r['x']=1+r['seed']*.01
            validate(bundle,path)  # Paired quality-memory points may have different memory x.
            bundle['expected_run_ids']=['missing-experiment']
            with self.assertRaises(ValueError):validate(bundle,path)
            del bundle['expected_run_ids']
            bundle['rows'].append({**row})
            with self.assertRaises(ValueError): validate(bundle,path)

    def test_logical_families_include_reuse(self):
        matrix=json.loads((ROOT/'docs/dissertation/refactor_v2/registries/experiment_matrix.json').read_text(encoding='utf-8'))
        for family,number in [('FRT',36),('ALC',14),('CEX',12),('LVL',6),('OVR',6),('STS',24)]:
            self.assertEqual(len(logical_family(matrix['runs'],family)),number)

    def test_full_campaign_endpoint_and_hashes(self):
        matrix=json.loads((ROOT/'docs/dissertation/refactor_v2/registries/experiment_matrix.json').read_text(encoding='utf-8'))
        concrete=[r for r in matrix['runs'] if r.get('config_hash')]
        dependencies=[]
        for s in range(6):
            spec={'seed':s%2,'experiment_id':f'fixture-dependency-{s}'}
            sha=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            dependencies.append({'experiment_id':spec['experiment_id'],'config_hash':sha,'spec':spec})
        rr=[{'experiment_id':r['experiment_id'],'config_hash':r['config_hash'],'step':30000,'spec':r['spec'],'metrics':{'horizon4_foreground_mse':.2}} for r in concrete+dependencies]
        resolution={'status':'resolved','selection_checkpoint':30000,'parent_campaign_sha256':matrix['source_campaign_sha256'],'run_order':dependencies}
        aggregate={'run_count':86,'rows':rr,'campaign_sha256':matrix['source_campaign_sha256']}
        validate_aggregate(aggregate,resolution,matrix)
        rr[0]['spec']={**rr[0]['spec'],'policy':'tampered'}
        with self.assertRaises(ValueError):validate_aggregate(aggregate,resolution,matrix)
        rr[0]['spec']=concrete[0]['spec']
        rr[0]['step']=16000
        with self.assertRaises(ValueError):validate_aggregate(aggregate,resolution,matrix)
        rr[0]['step']=30000;rr[0]['metrics']={'mean_foreground_mse':.2}
        with self.assertRaises(ValueError):validate_aggregate(aggregate,resolution,matrix)
        rr[0]['metrics']={'horizon4_foreground_mse':.2};rr[0]['config_hash']='f'*64
        with self.assertRaises(ValueError):validate_aggregate(aggregate,resolution,matrix)

    def test_stale_acceptance_and_stable_styles(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'acceptance.json';p.write_text('{}')
            with self.assertRaises(ValueError):read_acceptance(p,'0'*64)
        self.assertEqual(method_color('Linear-age'),'#2675ad')
        self.assertNotEqual(method_color('Linear-age'),method_color('Geometric-age'))

    def test_shared_claim_insertion_and_replay_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'registries').mkdir();source=folder/'source.txt';source.write_text('test only')
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            row={'method':'Fixture','seed':0,'x':1,'value':.2,'config_hash':'a'*64,'checkpoint_sha256':'b'*64,'data_sha256':'c'*64}
            evidence={'status':'accepted','evidence_status':'test only','metric':'test','units':'unit','direction':'lower','population':'test fixture','selection_role':'test','source_artifacts':[{'path':str(source),'sha256':sha}],'rows':[row,{**row,'seed':1}]}
            ep=folder/'evidence.json';ep.write_text(json.dumps(evidence))
            claim={'kind':'claim','status':'accepted','placeholder_key':'CLAIM_TEST','interpretation':'neutral','interpretation_text':'Test-only claim.', 'evidence_bundles':[{'path':str(ep),'sha256':hashlib.sha256(ep.read_bytes()).hexdigest()}]}
            cp=folder/'claim.json';cp.write_text(json.dumps(claim))
            (folder/'registries/placeholders.json').write_text(json.dumps([{'key':'CLAIM_TEST'}]))
            files=['front.md','chapter.md','conclusion.md']
            for name in files:(folder/name).write_text('> [RESULT PENDING: CLAIM_TEST; source=test.]\n')
            record={'key':'CLAIM_TEST','bundle':str(cp),'bundle_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'replacement':'Test-only claim.','interpretation':'neutral'}
            with patch.object(insertion,'MAN',folder),patch.object(insertion,'EDIT',folder),patch.object(insertion,'FILES',files):
                insertion.reapply([record])
                self.assertTrue(all('Test-only claim.' in (folder/n).read_text() for n in files))
                source.write_text('changed source')
                with self.assertRaises(ValueError):insertion.reapply([record])

    def test_ledger_completeness(self):
        folder=ROOT/'docs/dissertation/refactor_v2/registries'
        self.assertEqual(len(json.loads((folder/'formula_dependencies.json').read_text(encoding='utf-8'))),114)
        self.assertEqual(len(json.loads((folder/'experiment_matrix.json').read_text(encoding='utf-8'))['runs']),86)
        self.assertTrue(all(r['verification_status']=='verbatim' for r in json.loads((folder/'appendix_migration.json').read_text(encoding='utf-8'))))


if __name__=='__main__': unittest.main()
