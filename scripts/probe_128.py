"""Memory/speed probe for the 128px (2->8->32) config before committing a multi-hour run.

Verifies the derived stage shapes, then runs a few real train steps at a few batch sizes,
reporting it/s and peak CUDA memory. Use it to pick batch_size and (if needed) a BOT spatial
window, then calibrate max_steps to a wall-clock target.

Run:  venv\\Scripts\\python.exe scripts\\probe_128.py --set predictor.stage_s_window=[8,0,0]
"""

from __future__ import annotations

import argparse
import time

import torch

from hvp.config import Config, apply_cli_overrides
from hvp.objectives import compute_oneshot_loss
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import build_tokenizer


def probe(cfg: Config, batch_sizes, n_steps=6):
    dev = torch.device("cuda")
    topo = cfg.topology
    print(f"image_size={topo.image_size}  stage_spatial={topo.stage_spatial()}  "
          f"stage_temporal={topo.stage_temporal()}  latent_ch={topo.stage_latent_ch}")
    bot_tokens = topo.stage_temporal()[0] * topo.stage_spatial()[0] ** 2
    print(f"BOT self-attention tokens = {bot_tokens}  (s_window={cfg.predictor.stage_s_window} "
          f"t_window={cfg.predictor.stage_t_window})")

    tok = build_tokenizer(cfg).to(dev).train()
    cas = HierarchicalCascade(topo, cfg.predictor).to(dev).train()
    n_params = sum(p.numel() for p in tok.parameters()) + sum(p.numel() for p in cas.parameters())
    print(f"params = {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(list(tok.parameters()) + list(cas.parameters()), lr=1e-4)

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = (torch.rand(bs, 3, topo.sequence_length, topo.image_size, topo.image_size, device=dev) * 2 - 1)
            t0 = None
            for i in range(n_steps):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    total, m, _, _ = compute_oneshot_loss(tok, cas, x, cfg, step=2000)  # past pred warmup
                total.backward()
                opt.step()
                if i == 0:
                    torch.cuda.synchronize(); t0 = time.time()   # exclude first (compile/alloc) step
            torch.cuda.synchronize()
            dt = time.time() - t0
            ips = (n_steps - 1) / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  bs={bs:3d}  ok   {ips:5.2f} it/s   peak {peak:5.2f} GB   "
                  f"loss={float(total):.3f}  -> ~{(1/ips):.2f}s/step")
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:3d}  OOM")
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/so3_128.yaml")
    ap.add_argument("--batch", type=int, nargs="*", default=[4, 8, 12, 16])
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config, args.set)
    cfg.validate()
    probe(cfg, args.batch)


if __name__ == "__main__":
    main()
