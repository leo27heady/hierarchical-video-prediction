from pathlib import Path

import pytest

from hvp.config import Config, PredictorConfig, TopologyConfig, causal_output_length


def test_topology_derives_shapes():
    t = TopologyConfig()
    assert t.n_stages() == 3
    assert t.stage_spatial() == (16, 4, 1)   # bot, mid, top
    assert t.stage_temporal() == (9, 5, 3)
    t.validate()


def test_causal_output_length():
    assert causal_output_length(9) == 5
    assert causal_output_length(5) == 3


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Config.from_dict({"topology": {"image_size": 32, "bogus": 1}})


def test_from_dict_nested_and_tuples():
    cfg = Config.from_dict({"topology": {"stage_latent_ch": [16, 8, 8]}, "optim": {"lr": 1e-3}})
    assert cfg.topology.stage_latent_ch == (16, 8, 8)
    assert cfg.optim.lr == 1e-3
    cfg.validate()


def test_validate_rejects_bad_seq_len():
    with pytest.raises(ValueError):
        TopologyConfig(sequence_length=8).validate()


def test_all_yaml_configs_load_and_validate():
    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    yamls = sorted(cfg_dir.glob("*.yaml"))
    assert yamls, "no config yamls found"
    for p in yamls:
        Config.from_yaml(p).validate()


def test_predictor_per_stage_resolution_and_divisibility():
    pc = PredictorConfig(stage_dims=(192, 256, 256), stage_heads=(6, 8, 8), stage_layers=(4, 4, 3))
    dims, heads, layers = pc.resolved(3)
    assert dims == (192, 256, 256) and heads == (6, 8, 8) and layers == (4, 4, 3)
    with pytest.raises(ValueError):
        PredictorConfig(stage_dims=(100,), stage_heads=(7,)).resolved(1)  # 100 % 7 != 0


def test_per_stage_dims_cascade_builds_and_runs():
    import torch
    from hvp.predictor.cascade import HierarchicalCascade
    topo = TopologyConfig()
    pc = PredictorConfig(stage_dims=(48, 64, 64), stage_heads=(4, 4, 4), stage_layers=(2, 2, 1))
    cas = HierarchicalCascade(topo, pc)
    sp, tp = topo.stage_spatial(), topo.stage_temporal()
    z = [torch.randn(1, topo.stage_latent_ch[s], tp[s], sp[s], sp[s]) for s in range(3)]
    preds = cas(z)
    assert preds[1].shape == (1, tp[1], sp[1], sp[1], topo.stage_latent_ch[1])
