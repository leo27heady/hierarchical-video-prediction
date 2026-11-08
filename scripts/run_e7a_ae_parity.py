"""CPU numerical gate: explicit codec seed 0 preserves the legacy training path."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import hier_convae
from hvp_rev.provenance import atomic_write_json


def legacy_reference(path: str, *, steps: int, batch: int, lr: float) -> dict:
    torch.manual_seed(0)
    ae = hier_convae.HierConvAE().cpu()
    data = hier_convae.FramePool(path)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.05
    )
    for _ in range(steps):
        x = data.batch(batch)
        rc, rf, _, _ = ae(x)
        if torch.rand(()) < 0.5:
            loss = 4.0 * (F.l1_loss(rc, x) + F.mse_loss(rc, x))
        else:
            loss = F.l1_loss(rf, x) + F.mse_loss(rf, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return ae.state_dict()


def exact(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def main() -> None:
    torch.set_num_threads(1)
    prior_device = hier_convae.DEV
    hier_convae.DEV = "cpu"
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.RandomState(11)
            data_path = root / "frames.npy"
            np.save(data_path, rng.randint(0, 256, (42, 100, 64, 64), dtype=np.uint8))
            reference = legacy_reference(str(data_path), steps=2, batch=2, lr=3e-4)
            seed0_path = root / "seed0.pt"
            seed1_path = root / "seed1.pt"
            hier_convae.train(str(data_path), str(seed0_path), steps=2, batch=2, lr=3e-4, seed=0)
            hier_convae.train(str(data_path), str(seed1_path), steps=2, batch=2, lr=3e-4, seed=1)
            seed0 = torch.load(seed0_path, map_location="cpu", weights_only=False)
            seed1 = torch.load(seed1_path, map_location="cpu", weights_only=False)
            seed0_exact = exact(reference, seed0["model"])
            seed1_distinct = not exact(seed0["model"], seed1["model"])
            metadata_ok = (
                seed0["training"]["seed"] == 0 and seed1["training"]["seed"] == 1 and
                seed0["training"]["steps"] == seed1["training"]["steps"] == 2
            )
            if not seed0_exact:
                raise AssertionError("explicit seed 0 changed the legacy codec training path")
            if not seed1_distinct:
                raise AssertionError("codec seed 1 did not create a distinct trained state")
            if not metadata_ok:
                raise AssertionError("codec checkpoint training metadata is incomplete")
    finally:
        hier_convae.DEV = prior_device
    report = {
        "schema_version": 1,
        "status": "E7A_AE_PARITY_PASS",
        "device": "cpu",
        "updates": 2,
        "seed0_legacy_bit_exact": seed0_exact,
        "seed1_distinct": seed1_distinct,
        "checkpoint_metadata_valid": metadata_ok,
    }
    atomic_write_json("gates/e7a_ae_parity_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
