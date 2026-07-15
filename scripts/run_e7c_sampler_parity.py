"""CPU numerical parity gate for the explicit E7c sampler-seed interface."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json
from hvp_rev.rhvd import LAT, WIN, WindowDiffusion, WindowDiT, arm_cfg, rollout_crn_seed


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = arm_cfg("hier-min")
    assert mode == "diff" and not n_hist_f and not n_ctx_t and stride_c == stride_t == 1
    net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c).cpu().eval()
    sampler = WindowDiffusion(net)
    context_c = torch.randn(1, n_ctx_c, LAT, 4, 4)
    context_f = torch.randn(1, n_ctx_f, LAT, 8, 8)
    actions = torch.tensor([[1, 0, -1, 1]], dtype=torch.long)
    chunk, step = 3, 17
    legacy_seed = 100000 + 1000 * chunk + step

    torch.manual_seed(legacy_seed)
    old = sampler.sample(context_c, context_f, actions, n_steps=16)
    torch.manual_seed(rollout_crn_seed(0, chunk, step))
    new = sampler.sample(context_c, context_f, actions, n_steps=16)
    exact = all(torch.equal(left, right) for left, right in zip(old, new))
    if not exact:
        raise AssertionError("sampler_seed=0 changed the legacy NFE-16 numerical path")

    torch.manual_seed(rollout_crn_seed(1, chunk, step))
    alternate = sampler.sample(context_c, context_f, actions, n_steps=16)
    distinct = any(not torch.equal(left, right) for left, right in zip(old, alternate))
    if not distinct:
        raise AssertionError("sampler_seed=1 did not produce a distinct rollout-noise stream")

    report = {
        "schema_version": 1,
        "status": "SAMPLER_PARITY_PASS",
        "arm": "hier-min",
        "nfe": 16,
        "legacy_seed": legacy_seed,
        "explicit_seed": rollout_crn_seed(0, chunk, step),
        "bit_exact": exact,
        "alternate_seed_distinct": distinct,
        "device": "cpu",
    }
    atomic_write_json("gates/e7c_sampler_parity_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
