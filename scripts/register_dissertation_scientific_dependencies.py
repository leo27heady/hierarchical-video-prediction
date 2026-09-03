"""Bind every current numbered equation to its actual source or conceptual status."""
import json
import re
from pathlib import Path
from refactor_dissertation_v2 import ROOT,EDIT,MAN,FILES,digest,jwrite


def main():
    originals={r['current_number']:r for r in json.loads((EDIT/'registries/formula_dependencies.json').read_text(encoding='utf-8'))}
    op=ROOT.parent/'OpenSTL-integrated/openstl/models/hvp_integrated_model.py'
    temp=ROOT.parent/'TempVerseFormer/tempverse/temp_models/rev_transformer/model.py'
    rope=ROOT.parent/'visionBLT/modules/model/basic_components/rope.py'
    definitions={
      '1.S1':(None,'conceptual state/observation/conditional-future notation; not an implemented POMDP solver','Chapter 1.1; no empirical endpoint'),
      '2.R1':(temp,'RevFormer; published additive-coupling forward/inverse summary, exact algebra but finite-precision replay','2.2 quality/memory/runtime remain published configuration-level evidence'),
      '2.R2':(temp,'RevFormer and reversible autograd components; reverse-mode chain rule summary','2.2 activation-storage versus recomputation, not new measured quality'),
      '2.R3':(rope,'RotaryEmbeddingNd.forward, precompute_freqs_cis_nd; grouped physical-axis RoPE identity','2.3 fixed-patch V-BLT; not integrated Fourier position'),
      '3.P1':(ROOT/'hvp_hca/allocation.py','AllocationPlan; exact context age/cell identity','3.6 FRT/ALC/CEX/LVL/OVR/STS method contract'),
      '3.P2':(ROOT/'hvp_hca/allocation.py','AllocationPlan.token_count and reach; geometric state arithmetic','4.3/4.4 payload and oldest age, not measured VRAM'),
      '3.P3':(ROOT/'hvp_hca/data.py','PackedWindowBatcher and latent normalization; frozen two-level codec interface','3.2 frozen observation boundary'),
      '3.P4':(ROOT/'hvp_rev/rhvd.py','WindowDiT.forward: joint packed context/target sequence with physical metadata','3.2/3.3 four jointly denoised target frames'),
      '3.P5':(ROOT/'hvp_rev/rhvd.py','WindowDiffusion.loss: sum of coarse/fine per-stream MSE means','3.3 executed optimization objective; top stream context-only'),
      '3.P6':(None,'causal autoregressive buffer-update contract for legacy/optional rollout; not current v2 endpoint training','3.4 and optional 4.3 rollout slot; does not certify unimplemented v2 evaluator'),
      '3.P7':(ROOT/'hvp_hca/state.py','packed tensor payload accounting; byte arithmetic excludes encoder/decoder/workspace','3.4 retained-state boundary'),
      '3.P8':(ROOT/'hvp_rev/evaluation/reach.py','schematic world/viewport potential visibility; historical observability definition','3.5 historical E2 access, not v2 confirmation mask'),
      '3.P9':(ROOT/'hvp_rev/evaluation/reach.py','historical pair-specific reach-difference support; conceptual set form','4.3 historical E2 population only'),
      '3.P10':(ROOT/'hvp_hca/postcampaign.py','pair_results: arithmetic mean of independent training-seed endpoints','4.3/4.4 descriptive two-seed effects; no n=2 t interval'),
      '3.P11':(ROOT/'scripts/v2/train_article4_campaign.py','validation: horizon4_foreground_mse; target >0.5, batch foreground-pixel reduction then batch-size mean','4.2 calibration and development family selection; not confirmation four-horizon mean'),
      '3.I1':(op,'encode_packed / SpaceTimeRouter: gather K fine tokens with coarse coverage','4.8 quota/core payload; no total-VRAM ratio from this equation'),
      '3.I2':(op,'SpaceTimeRouter: normalized residual, temporal change, learned scalar; discrete membership','4.8 route switch/regime/quota/stability, separate estimands'),
      '3.I3':(op,'PhysicalTokenMetadata, _grid_metadata, Fourier position and level embedding','4.8 permutation/position correctness; not RoPE'),
      '3.I4':(op,'ReversibleTransformerCore additive coupling; ordinary boundaries excluded','4.8 identical-forward profile; parameter/gradient parity gates')}
    rows=[]
    for name in FILES[:7]:
        text=(MAN/name).read_text(encoding='utf-8')
        for m in re.finditer(r'\$\$(.*?)\\tag\{([^}]+)\}(.*?)\$\$',text,re.S):
            tag=m[2];head=re.findall(r'^#{1,3} .+',text[:m.start()],re.M)[-1]
            before=text[max(0,text.rfind('\n\n',0,m.start()-2)):m.start()].strip()
            end=text.find('\n\n',m.end()+2);after=text[m.end():end if end>=0 else len(text)].strip()
            if tag in definitions:
                path,implementation,estimand=definitions[tag]
                if path and not path.is_file(): raise ValueError(f'missing source {path}')
                binding={'path':str(path),'sha256':digest(path.read_bytes())} if path else None
                origin='English refactor definition or grouped implementation relation'
            else:
                old=originals[tag];binding={'source_number':tag,'ledger':'formula_dependencies.json','source_sha256':old['source_sha256']}
                implementation=old['implementation_location'];estimand=old['result_estimand'];origin='preserved source equation; detailed source-era attribution in dependency ledger'
            rows.append({'tag':tag,'location':f'{name}:{text[:m.start()].count(chr(10))+1}','section':head,
                         'left_hand_side_and_semantics':before,'immediate_explanation':after,'origin':origin,
                         'implementation_contract':implementation,'source_binding':binding,'result_or_downstream_use':estimand,
                         'equation_sha256':digest(m[0]),'outgoing_callouts':[n for n in re.findall(r'(?:Figure|Table|equation) ([\w.-]+)',text[m.end():m.end()+1400])]})
    jwrite(EDIT/'registries/current_formula_dependencies.json',rows)
    print(f'Bound {len(rows)} current equations ({len(definitions)} new/grouped scientific objects)')


if __name__=='__main__':main()
