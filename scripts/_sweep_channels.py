"""Scratch sweep: does redistributing latent channels toward the coarse stages help?
Trains a short run per stage_latent_ch config and reports recon / per-stage effective rank
(capacity actually used) / single-step changed-region pred vs persistence vs the rotation null."""
import torch
from torch.utils.data import DataLoader

from hvp.config import Config
from hvp.data import dataset_from_cfg
from hvp.eval.baselines import rotation_extrap
from hvp.eval.quality import changed_region_metrics
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import build_tokenizer
from hvp.train import fit_oneshot

STEPS = 800
CONFIGS = {
    "baseline   [16,8,8]":    [16, 8, 8],
    "modest     [32,64,32]":  [32, 64, 32],
    "extreme    [64,512,4096]": [64, 512, 4096],
}


@torch.no_grad()
def single_step_gate(tok, cas, cfg, dev, max_batches=8):
    va = DataLoader(dataset_from_cfg(cfg, "val"), batch_size=16, num_workers=0)
    N = cfg.topology.n_stages()
    mM = mP = mR = 0.0; nb = 0
    for i, b in enumerate(va):
        if i >= max_batches:
            break
        x = b["video"].to(dev)
        gt8, prev = x[:, :, 8:9], x[:, :, 7:8]
        z = tok.encoder(x)
        zp = [p.permute(0, 4, 1, 2, 3).contiguous() for p in cas(z, use_cascade=True)]
        model8 = tok.decoder.decode_depth(zp, N).float()[:, :, 7:8]
        rot = rotation_extrap(x[:, :, :8], 1)
        bs = x.shape[0]
        mM += changed_region_metrics(model8, gt8, prev)["changed_psnr"] * bs
        mP += changed_region_metrics(prev, gt8, prev)["changed_psnr"] * bs
        mR += changed_region_metrics(rot, gt8, prev)["changed_psnr"] * bs
        nb += bs
    return mM / nb, mP / nb, mR / nb


def run():
    dev = torch.device("cuda")
    rows = []
    for name, slc in CONFIGS.items():
        print(f"\n===== {name} =====", flush=True)
        cfg = Config.from_yaml("configs/base_64.yaml", [
            f"topology.stage_latent_ch={slc}",
            f"optim.max_steps={STEPS}",
            "predictor.pred_warmup_steps=400",
            "run.wandb=false", "run.val_every=100000",
        ])
        res = fit_oneshot(cfg)
        tok = build_tokenizer(cfg).to(dev).eval()
        cas = HierarchicalCascade(cfg.topology, cfg.predictor).to(dev).eval()
        blob = torch.load(res["ckpt"], map_location=dev, weights_only=False)
        tok.load_state_dict(blob["model"]); cas.load_state_dict(blob["cascade"])
        npar = sum(p.numel() for p in tok.parameters()) + sum(p.numel() for p in cas.parameters())
        mM, mP, mR = single_step_gate(tok, cas, cfg, dev)
        rows.append((name, slc, npar, res, mM, mP, mR))

    print("\n\n================ CHANNEL SWEEP RESULTS (800 steps, 64x64) ================")
    hdr = f"{'config':22s} {'par(M)':>6s} {'recon':>6s} {'erank b/m/t':>16s} {'midlift':>7s} | {'chg: model':>10s} {'persist':>7s} {'rotation':>8s}"
    print(hdr)
    for name, slc, npar, res, mM, mP, mR in rows:
        eb, em, et = res["val/erank_s0"], res["val/erank_s1"], res["val/erank_s2"]
        erank = f"{eb:.1f}/{em:.1f}/{et:.1f}"
        print(f"{name:22s} {npar/1e6:6.2f} {res['val/recon_psnr']:6.2f} {erank:>16s} "
              f"{res['val/cascade_lift_s1']:+7.3f} | {mM:10.2f} {mP:7.2f} {mR:8.2f}")
    print("\n(erank b/m/t against latent dims " + " / ".join(str(c) for c in CONFIGS['baseline   [16,8,8]']) +
          " etc. — full rank = all channels used; chg = changed-region PSNR, dB)")


if __name__ == "__main__":
    run()
