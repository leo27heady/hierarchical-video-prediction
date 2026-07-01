"""Calibration trainer for the per-pixel SpatialDLH. Goal of this run is NOT sharpness (the renderer handles
that) — it is to answer three questions: (a) does the per-location e-map SEGMENT the moving digits? (b) does the
tick rate calibrate to ~8-15%? (c) does open-loop fg-MSE MATCH the vanilla 2-level DLH long-horizon (no worse).
Logs tick/soft-rate/sKL; evals open-loop fg-MSE + lap_var + segmentation-correlation + an e-map strip.
Usage: venv/Scripts/python.exe -m hvp_dlh.train_spatial --iters 5000 --seq-len 50 --batch-size 32 --run-name sp_cal
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
from hvp_dlh.model_spatial import SpatialDLH, SpatialDLHConfig
dev = "cuda" if torch.cuda.is_available() else "cpu"

_lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=dev).view(1,1,3,3)
def lapv(im): return F.conv2d(im.reshape(-1,1,64,64), _lap, padding=1).var(dim=(1,2,3)).mean().item()


@torch.no_grad()
def seg_corr(model, obs):
    """Teacher-forced: correlation between the top-level per-location e-map and the frame-diff motion mask
    (downsampled to the 16x16 latent grid). corr>0 => the gate segments motion. Also returns tick-on-fg vs bg."""
    cfg = model.cfg; B, T = obs.shape[:2]
    feats = model._bottom_up(obs); state = model._init_state(B, dev)
    emaps, motion = [], []
    prev = obs[:, 0]
    for t in range(T):
        x_t = [feats[n][:, t] for n in range(cfg.levels)]
        _, state, info = model._step(state, x_t, mode="posterior")
        emaps.append(info["e"][-1])                                          # (B,1,16,16) top level
        d = (obs[:, t] - prev).abs()                                         # (B,1,64,64)
        motion.append(F.adaptive_avg_pool2d(d, cfg.latent_hw))              # -> (B,1,16,16)
        prev = obs[:, t]
    E = torch.stack(emaps, 1).reshape(-1).float()                           # all (t,loc)
    M = (torch.stack(motion, 1).reshape(-1) > 0.02).float()                  # motion mask at 16x16
    # pearson corr between tick and motion
    Ec, Mc = E - E.mean(), M - M.mean()
    corr = (Ec * Mc).sum() / (Ec.norm() * Mc.norm() + 1e-8)
    tick_on_motion = (E * M).sum() / M.sum().clamp(min=1)
    tick_on_static = (E * (1 - M)).sum() / (1 - M).sum().clamp(min=1)
    return corr.item(), tick_on_motion.item(), tick_on_static.item()


@torch.no_grad()
def evaluate(model, test, ctx=36, H=100, n=16, bon=3):
    obs = test[:n, :ctx+H].to(dev); gt = obs[:, ctx:]; fg = (gt > 0.1)
    best = torch.full((n,), 1e9); keep = torch.zeros_like(gt.cpu())
    for _ in range(bon):
        p, _, _ = model.open_loop(obs, ctx, H); p = p.cpu()
        for c in range(n):
            m = (((p[c]-gt[c].cpu())**2)[fg[c].cpu()].mean()).item()
            if m < best[c]: best[c] = m; keep[c] = p[c]
    keep = keep.to(dev)
    fgc = lambda t: (((keep[:,t]-gt[:,t])**2)*fg[:,t].float()).sum().item()/fg[:,t].float().sum().clamp(min=1).item()
    corr, t_mot, t_stat = seg_corr(model, test[:8, :60].to(dev))
    return {"fg_all": (((keep-gt)**2)*fg.float()).sum().item()/fg.float().sum().clamp(min=1).item(),
            "fg@25": fgc(24), "fg@50": fgc(49), "fg@100": fgc(99), "lap": lapv(keep),
            "seg_corr": corr, "tick_motion": t_mot, "tick_static": t_stat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000); ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--force-tick-iters", type=int, default=3000)
    ap.add_argument("--beta-warmup-iters", type=int, default=4000); ap.add_argument("--lambda-e", type=float, default=1e-2)
    ap.add_argument("--persist-margin", type=float, default=0.0); ap.add_argument("--target-tick-rate", type=float, default=0.12)
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--run-name", default="sp_cal"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = SpatialDLHConfig(seq_len=a.seq_len, batch_size=a.batch_size, force_tick_iters=a.force_tick_iters,
                           beta_warmup_iters=a.beta_warmup_iters, lambda_e=a.lambda_e,
                           persist_margin=a.persist_margin, target_tick_rate=a.target_tick_rate)
    out = ROOT / "logs_spatial"; out.mkdir(exist_ok=True)
    model = SpatialDLH(cfg).to(dev)
    print(f"[{a.run_name}] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | seq{cfg.seq_len} B{cfg.batch_size} "
          f"ft{cfg.force_tick_iters} bw{cfg.beta_warmup_iters} lam_e{cfg.lambda_e} margin{cfg.persist_margin}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    train = np.load(str(CW/"data_mmnist"/"train.npy")); test = np.load(str(CW/"data_mmnist"/"test.npy"))
    test_t = torch.from_numpy(test.astype(np.float32)/255).unsqueeze(2)
    hist = []
    for step in range(1, a.iters+1):
        beta = cfg.beta_max * min(1.0, step / max(1, cfg.beta_warmup_iters))
        ft = step <= cfg.force_tick_iters
        idx = np.random.randint(0, train.shape[0], cfg.batch_size)
        obs = torch.from_numpy(train[idx, :cfg.seq_len].astype(np.float32)/255).unsqueeze(2).to(dev)
        loss, met = model.loss(obs, beta=beta, force_tick=ft)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        if step % a.log_every == 0:
            print(f"[{step:5d}] nll {met['nll'].item():7.1f} sKL {[round(x,2) for x in met['skl_per_level'].tolist()]} "
                  f"tick {[round(x,3) for x in met['e_per_level'].tolist()]} soft {met['soft_tick_rate'].item():.3f} "
                  f"beta {beta:.2f} ft {int(ft)}", flush=True)
        if step % a.eval_every == 0 or step == a.iters:
            model.eval(); ev = evaluate(model, test_t); model.train(); ev["step"] = step; hist.append(ev)
            print(f"  EVAL[{step}] fg_all {ev['fg_all']:.3f} (@25 {ev['fg@25']:.3f} @50 {ev['fg@50']:.3f} @100 {ev['fg@100']:.3f}) "
                  f"lap {ev['lap']:.4f} | SEG corr {ev['seg_corr']:.3f} tick_motion {ev['tick_motion']:.3f} "
                  f"tick_static {ev['tick_static']:.3f}", flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "step": step},
                       out / f"{a.run_name}_last.pt")
            (out / f"{a.run_name}_hist.json").write_text(json.dumps(hist, indent=1))
    print(f"[{a.run_name}] DONE", flush=True)


if __name__ == "__main__":
    main()
