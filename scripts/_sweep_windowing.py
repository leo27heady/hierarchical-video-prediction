"""Scratch sweep: does starving fine-stage self-attention (temporal + spatial windows) force
the model to lean on the parent (cascade) — and does prediction survive it?

Per config: recon, per-stage effective rank, single-step changed-region PSNR for the model with
the cascade ON vs OFF (the parent-reliance signal) vs persistence vs the rotation null.
"""
import traceback

import torch
from torch.utils.data import DataLoader

from hvp.config import Config
from hvp.data import dataset_from_cfg
from hvp.eval.baselines import rotation_extrap
from hvp.eval.quality import changed_region_metrics
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import build_tokenizer
from hvp.train import fit_oneshot

BASE = ["predictor.pred_warmup_steps=400", "run.wandb=false", "run.val_every=100000"]
WIN = ["predictor.stage_t_window=[1,1,0]", "predictor.stage_s_window=[8,0,0]"]  # bot self+8x8, mid self, top full
WIDE = ["topology.tap_after=[0,2,4]", "topology.stage_latent_ch=[8,64,512]"]

CONFIGS = {
    "A full        16/4/1": BASE + ["optim.max_steps=800"],
    "B windowed    16/4/1": BASE + ["optim.max_steps=800"] + WIN,
    "C wide+window 32/8/2": BASE + ["optim.max_steps=500", "optim.batch_size=4"] + WIDE + WIN,
}


@torch.no_grad()
def gate(tok, cas, cfg, dev, max_batches=8):
    va = DataLoader(dataset_from_cfg(cfg, "val"), batch_size=12, num_workers=0)
    N = cfg.topology.n_stages()
    acc = {k: 0.0 for k in ("on", "off", "persist", "rot")}; nb = 0
    for i, b in enumerate(va):
        if i >= max_batches:
            break
        x = b["video"].to(dev)
        gt8, prev = x[:, :, 8:9], x[:, :, 7:8]
        z = tok.encoder(x)

        def dec(use_cascade):
            zp = [p.permute(0, 4, 1, 2, 3).contiguous() for p in cas(z, use_cascade=use_cascade)]
            return tok.decoder.decode_depth(zp, N).float()[:, :, 7:8]

        bs = x.shape[0]
        acc["on"] += changed_region_metrics(dec(True), gt8, prev)["changed_psnr"] * bs
        acc["off"] += changed_region_metrics(dec(False), gt8, prev)["changed_psnr"] * bs
        acc["persist"] += changed_region_metrics(prev, gt8, prev)["changed_psnr"] * bs
        acc["rot"] += changed_region_metrics(rotation_extrap(x[:, :, :8], 1), gt8, prev)["changed_psnr"] * bs
        nb += bs
    return {k: v / nb for k, v in acc.items()}


def run():
    dev = torch.device("cuda")
    rows = []
    for name, ov in CONFIGS.items():
        print(f"\n===== {name} =====", flush=True)
        try:
            cfg = Config.from_yaml("configs/base_64.yaml", ov)
            res = fit_oneshot(cfg)
            tok = build_tokenizer(cfg).to(dev).eval()
            cas = HierarchicalCascade(cfg.topology, cfg.predictor).to(dev).eval()
            blob = torch.load(res["ckpt"], map_location=dev, weights_only=False)
            tok.load_state_dict(blob["model"]); cas.load_state_dict(blob["cascade"])
            g = gate(tok, cas, cfg, dev)
            rows.append((name, res, g)); del tok, cas; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}"); traceback.print_exc(); rows.append((name, None, None))

    print("\n\n========= WINDOWING SWEEP (single-step changed-region, dB) =========")
    print(f"{'config':22s} {'recon':>6s} {'erank b/m/t':>16s} {'midlift':>7s} {'botlift':>7s} | "
          f"{'on':>5s} {'off':>5s} {'on-off':>6s} {'persist':>7s} {'rot':>5s}")
    for name, res, g in rows:
        if res is None:
            print(f"{name:22s}  (failed)"); continue
        er = f"{res['val/erank_s0']:.1f}/{res['val/erank_s1']:.1f}/{res['val/erank_s2']:.1f}"
        print(f"{name:22s} {res['val/recon_psnr']:6.2f} {er:>16s} "
              f"{res['val/cascade_lift_s1']:+7.3f} {res['val/cascade_lift_s0']:+7.3f} | "
              f"{g['on']:5.2f} {g['off']:5.2f} {g['on']-g['off']:+6.2f} {g['persist']:7.2f} {g['rot']:5.2f}")
    print("\non/off = cascade conditioning on vs off; on-off>0 and large = the parent is load-bearing.")


if __name__ == "__main__":
    run()
