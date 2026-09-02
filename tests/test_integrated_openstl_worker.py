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


def test_complete_cpu_worker_writes_resume_and_streaming_evidence(tmp_path):
    openstl = Path(__file__).resolve().parents[2] / "OpenSTL-integrated"
    assert openstl.is_dir()
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
