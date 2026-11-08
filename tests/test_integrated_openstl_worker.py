from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.v2.train_integrated_openstl import main


def _write_cache(root: Path) -> None:
    folder = root / "taxibj" / "memmap_v1"
    folder.mkdir(parents=True)
    rng = np.random.RandomState(7)
    for name, size in (("X_train", 10), ("Y_train", 10), ("X_test", 2), ("Y_test", 2)):
        array = np.lib.format.open_memmap(
            folder / f"{name}.npy", mode="w+", dtype=np.float32,
            shape=(size, 4, 2, 32, 32),
        )
        array[:] = rng.uniform(-1, 1, size=array.shape).astype(np.float32)
        array.flush()
        del array


def test_complete_cpu_worker_writes_resume_and_streaming_evidence(tmp_path, monkeypatch):
    import os
    import pytest
    if not os.environ.get('HVP_OPENSTL_ROOT'):
        pytest.skip('set HVP_OPENSTL_ROOT to opt into external framework tests')
    openstl = Path(os.environ.get('HVP_OPENSTL_ROOT', Path(__file__).resolve().parents[2] / "OpenSTL-integrated"))
    if not (openstl / "openstl/models/hvp_integrated_model.py").is_file():
        import pytest
        pytest.skip("optional OpenSTL integration checkout required; see integrations/openstl")
    data = tmp_path / "data"
    _write_cache(data)
    config = tmp_path / "smoke.py"
    config.write_text(
        "\n".join([
            "method = 'HVPIntegrated'",
            "d_model = 16",
            "stem_width = 8",
            "core_depth = 1",
            "decoder_depth = 1",
            "num_heads = 2",
            "mlp_ratio = 2",
            "hierarchy_mode = 'hierarchical'",
            "routing_mode = 'static'",
            "detail_quota = 4",
            "min_detail_per_frame = 1",
            "reversible_core = False",
            "position_frequencies = 2",
            "validation_fraction = 0.2",
            "memmap_cache = 'taxibj/memmap_v1'",
            "strict_memmap = True",
            "lr = 0.001",
            "batch_size = 2",
            "val_batch_size = 2",
            "epoch = 1",
            "sched = 'onecycle'",
            "clip_grad = 1.0",
        ]) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    arguments = argparse.Namespace(
        config=config,
        openstl=openstl,
        data_root=data,
        output=output,
        dataset="taxibj",
        seed=0,
        run_id="CPU_SMOKE",
        max_epochs=2,
        max_steps=2,
        batch_size=2,
        num_workers=0,
        precision="32-true",
        log_every=1,
        checkpoint_every=2,
        resume=True,
        wandb_project="unused",
        wandb_group="unused",
        wandb_mode="disabled",
        no_wandb=True,
    )
    assert main(arguments) == 0
    assert (output / "result.json").is_file()
    assert (output / "metrics.jsonl").is_file()
    assert (output / "checkpoints" / "last.ckpt").is_file()
    assert (output / "saved" / "streaming_metrics.json").is_file()
    assert (output / "panels" / "locked_test.png").is_file()
    # Remove the development step cap while preserving the original two-epoch
    # OneCycle schedule. The run must continue from step two to step eight.
    arguments.max_steps = None
    assert main(arguments) == 0
    checkpoint = __import__("torch").load(
        output / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False
    )
    assert checkpoint["global_step"] == 8
    # A real bounded worker must validate without calling Trainer.test at all.
    from lightning import Trainer
    def forbidden_test(*args, **kwargs):
        raise AssertionError('development must not evaluate the test set')
    monkeypatch.setattr(Trainer, 'test', forbidden_test)
    arguments.output = tmp_path / 'development'
    arguments.development = True
    arguments.max_steps = 2
    arguments.max_time = '00:00:01:00'
    assert main(arguments) == 0
    result = __import__('json').loads((arguments.output / 'result.json').read_text())
    assert result['test_evaluated'] is False
    assert result['validation_loss'] >= 0
    assert not (arguments.output / 'panels/locked_test.png').exists()
