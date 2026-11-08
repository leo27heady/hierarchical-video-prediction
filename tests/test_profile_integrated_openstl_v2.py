import importlib.util
from pathlib import Path
import torch
import pytest
import sys
import os
if not os.environ.get('HVP_OPENSTL_ROOT'):
    pytest.skip('set HVP_OPENSTL_ROOT to opt into external framework tests', allow_module_level=True)
ROOT=Path(__file__).resolve().parents[1]
OPENSTL=Path(os.environ.get('HVP_OPENSTL_ROOT',ROOT.parent/'OpenSTL-integrated'))
if not (OPENSTL/'openstl/models/hvp_integrated_model.py').exists():
    pytest.skip('integrated OpenSTL checkout required for profile contract tests',allow_module_level=True)
sys.path.insert(0,str(OPENSTL))
from openstl.models.hvp_integrated_model import HVPIntegrated_Model

path=ROOT/'scripts/v2/profile_integrated_openstl_memory.py'
spec=importlib.util.spec_from_file_location('hvp_profile_v2',path)
profile=importlib.util.module_from_spec(spec); spec.loader.exec_module(profile)


def tiny():
    return HVPIntegrated_Model(in_shape=(2,1,16,16),aft_seq_length=2,d_model=16,stem_width=8,
                               core_depth=2,decoder_depth=1,num_heads=2,detail_quota=4,
                               min_detail_per_frame=1,routing_mode='hybrid',position_frequencies=2).eval()


def test_unique_storage_counts_views_once():
    value=torch.ones(8)
    assert profile.tensor_bytes([value,value[:2]])==40
    assert profile.tensor_bytes([value,value[:2]],True)==32


def test_saved_tensor_diagnostic_releases_output_graph():
    import weakref, gc
    class Probe(torch.nn.Module):
        def forward(self, value):
            output = value.exp()
            self.output_ref = weakref.ref(output)
            return output
    model = Probe()
    profile.saved_tensor_accounting(model, torch.ones(5, requires_grad=True), 'fp32')
    gc.collect()
    assert model.output_ref() is None


def test_profile_parity_rejects_persistence_and_accepts_nonzero_head():
    model=tiny(); inputs=torch.rand(2,2,1,16,16); targets=torch.rand_like(inputs)
    with pytest.raises(RuntimeError,match='degenerate'): profile.parity(model,inputs,targets,'fp32')
    torch.nn.init.normal_(model.decoder.output.weight,std=.03)
    assert profile.parity(model,inputs,targets,'fp32')['batch']==2
    for mode in profile.MODES:
        profile.set_mode(model,mode)
        saved=profile.saved_tensor_accounting(model,inputs,'fp32')
        assert saved['forward_saved_unique_storage_bytes']>0
    packed=model.encode_and_pack(inputs,'hybrid')
    count=profile.packed_accounting(packed)
    assert count['logical_bytes']>packed.tokens.numel()*packed.tokens.element_size()
