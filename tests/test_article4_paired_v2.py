import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('paired',Path(__file__).parents[1]/'scripts/v2/analyze_article4_paired_v2.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def fixture(ids=(2,1)):
    return dict(clip=np.repeat(np.array(ids)[:,None],4,axis=1),
                horizon=np.tile(np.arange(1,5),(2,1)),
                foreground_mse=np.array([[100,100,100,2.],[100,100,100,4.]]))


def test_pixel_weighting_is_not_clip_weighting_and_empty_is_explicit():
    out=module.foreground_scores([[1.,0],[4.,4],[9.,9]],[[1,0],[1,1],[0,0]])
    assert out['pixel_weighted']==3
    assert out['clip_mean']==pytest.approx(5/3)
    assert out['empty_clips']==1


def test_pairing_uses_ids_and_horizon_four_only():
    a=fixture(); b={k:v[::-1].copy() for k,v in a.items()}
    b['foreground_mse'][:,3]-=1
    out=module.paired(a,b,2,repetitions=100)
    assert out['flat_minus_hierarchy']==1
    assert out['adjusted_percentile_interval']==[1,1]


def test_unpaired_duplicate_and_missing_horizon_fail():
    with pytest.raises(ValueError,match='unpaired'): module.paired(fixture(),fixture((3,1)),1,100)
    with pytest.raises(ValueError,match='exactly'): module.primary(fixture((1,1)))
    a=fixture(); a['horizon'][:]=1
    with pytest.raises(ValueError,match='exactly'): module.primary(a)
    a=fixture(); a['horizon'][0,3]=3
    with pytest.raises(ValueError,match='exactly'): module.primary(a)
