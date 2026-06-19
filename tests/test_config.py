import pytest

from hvp.config import Config, TopologyConfig, causal_output_length


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
