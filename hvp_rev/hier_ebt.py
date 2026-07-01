"""Minimal MilliVid+EBT: CONSISTENCY-COUPLED hierarchical energy predictor (red-team-corrected design).

Two latent levels (coarse 4x4, fine 8x8). Total energy over a JOINT candidate (y_c, y_f):
    E = E_coarse(ctx_c, y_c) + E_fine(ctx_f, y_c, y_f) + lam * E_cons(y_c, y_f)
where E_cons penalizes disagreement between the coarse latent and a learned down-projection of the fine latent -- the
cross-level back-coupling that makes this a genuine constraint-satisfaction problem (fine must agree with coarse; coarse
gets nudged by fine), the one mechanism the red-team said could beat single-scale saturation. Inference = JOINT energy
descent over (y_c, y_f). S2 recipe (replay buffer + no-detach + final loss). D2 diagnostic GREEN-lit this (oracle-coarse
improved fine prediction 75%).

Controlled baselines (the honest comparison the red-team required):
  - hier-regression : feed-forward predict coarse from ctx, then fine from ctx+predicted-coarse (NO energy) -> isolates
    hierarchy-as-inductive-bias from energy-as-mechanism (the critical baseline).
  - single-scale EBT is hvp_rev/ebt_stage1 (matched descent budget).
Metrics stratified easy/hard; decoded SAMPLE IMAGES saved each eval.

  venv/Scripts/python.exe -m hvp_rev.hier_ebt --ae logs_rhcp/hier_ae.pt --steps 8000
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.hier_convae import load as load_hier, LAT
from hvp_rev import long_eval as LE
from hvp_rev.viz import save_samples

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class HierData:
    def __init__(self, ae, clip_path, n_ctx=6, k=4, n_clips=2500, max_t=100, n_val=300):
        clips = np.load(clip_path)[:n_clips]
        N, T = clips.shape[0], min(max_t, clips.shape[1])
        x = torch.from_numpy(clips[:, :T].astype(np.float32) / 127.5 - 1.0).unsqueeze(2)
        zc, zf = [], []
        with torch.no_grad():
            for t in range(T):
                c, f = ae.encode(x[:, t].to(DEV))
                zc.append(c.cpu()); zf.append(f.cpu())
        self.zc, self.zf = torch.stack(zc, 1), torch.stack(zf, 1)     # (N,T,LAT,4,4),(N,T,LAT,8,8)
        self.x = x                                                    # keep frames for sample viz
        self.N, self.T, self.n_ctx, self.k, self.n_val = N, T, n_ctx, k, n_val

    def batch(self, B, train=True, return_frames=False):
        hi, lo = (self.N - self.n_val, 0) if train else (self.N, self.N - self.n_val)
        ci = torch.randint(lo, hi, (B,)); ti = torch.randint(self.n_ctx, self.T - self.k, (B,))
        cc = torch.stack([self.zc[ci[b], ti[b] - self.n_ctx:ti[b]].reshape(-1, 4, 4) for b in range(B)]).to(DEV)
        cf = torch.stack([self.zf[ci[b], ti[b] - self.n_ctx:ti[b]].reshape(-1, 8, 8) for b in range(B)]).to(DEV)
        tc = torch.stack([self.zc[ci[b], ti[b] + self.k - 1] for b in range(B)]).to(DEV)
        tf = torch.stack([self.zf[ci[b], ti[b] + self.k - 1] for b in range(B)]).to(DEV)
        lastf = torch.stack([self.zf[ci[b], ti[b] - 1] for b in range(B)]).to(DEV)
        change = (tf - lastf).flatten(1).norm(dim=1)
        if return_frames:
            gt_frame = torch.stack([self.x[ci[b], ti[b] + self.k - 1] for b in range(B)]).to(DEV)
            last_frame = torch.stack([self.x[ci[b], ti[b] - 1] for b in range(B)]).to(DEV)
            return cc, cf, tc, tf, change, last_frame, gt_frame
        return cc, cf, tc, tf, change


def _net(cin, cout, hw, w=128):
    return nn.Sequential(nn.Conv2d(cin, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
                         nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU())


class HierEnergy(nn.Module):
    def __init__(self, n_ctx, lam=1.0):
        super().__init__()
        self.lam = lam
        self.cnet = _net(LAT * (n_ctx + 1), 0, 4); self.chead = nn.Linear(128, 1)
        self.fnet = _net(LAT * (n_ctx + 2), 0, 8); self.fhead = nn.Linear(128, 1)
        self.down = nn.Conv2d(LAT, LAT, 1)                              # fine->coarse projection for consistency
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def alpha(self):
        return self.log_alpha.exp().clamp(min=1e-4, max=200.0)

    def energy(self, cc, cf, y_c, y_f):
        e_c = self.chead(self.cnet(torch.cat([cc, y_c], 1)).mean(dim=(2, 3)))
        yc_up = F.interpolate(y_c, size=(8, 8), mode="nearest")
        e_f = self.fhead(self.fnet(torch.cat([cf, yc_up, y_f], 1)).mean(dim=(2, 3)))
        e_cons = ((self.down(F.avg_pool2d(y_f, 2)) - y_c) ** 2).mean(dim=(1, 2, 3), keepdim=True).squeeze(-1).squeeze(-1)
        return e_c + e_f + self.lam * e_cons


def descend_hier(model, cc, cf, n_steps, tc=None, tf=None, mode="s1", langevin=0.0, randomize=False, train=False,
                 y_init=None, scale=2.0):
    B = cc.shape[0]
    if y_init is None:
        y_c = torch.randn(B, LAT, 4, 4, device=DEV); y_f = torch.randn(B, LAT, 8, 8, device=DEV)
    else:
        y_c, y_f = y_init[0].clone(), y_init[1].clone()
    a0 = model.alpha(); traj, losses = [], []
    for _ in range(n_steps):
        y_c = y_c.requires_grad_(True); y_f = y_f.requires_grad_(True)
        E = model.energy(cc, cf, y_c, y_f)
        gc, gf = torch.autograd.grad(E.sum(), [y_c, y_f], create_graph=train)
        ac = a0 / scale + torch.rand_like(y_c) * (a0 * scale - a0 / scale) if randomize else a0
        af = a0 / scale + torch.rand_like(y_f) * (a0 * scale - a0 / scale) if randomize else a0
        y_c = y_c - ac * gc; y_f = y_f - af * gf
        if langevin > 0:
            y_c = y_c + langevin * torch.randn_like(y_c); y_f = y_f + langevin * torch.randn_like(y_f)
        traj.append(float(E.mean().detach()))
        if train and tc is not None and mode == "s1":
            losses.append(F.smooth_l1_loss(y_c, tc) + F.smooth_l1_loss(y_f, tf))
            y_c, y_f = y_c.detach(), y_f.detach()
        elif not train:
            y_c, y_f = y_c.detach(), y_f.detach()
    loss = None
    if train and tc is not None:
        loss = torch.stack(losses).mean() if mode == "s1" else (F.smooth_l1_loss(y_c, tc) + F.smooth_l1_loss(y_f, tf))
    return (y_c, y_f), traj, loss


def _metrics(ae, y_f, tf, decode=True):
    with torch.no_grad():
        p = ae.decode_fine(y_f); g = ae.decode_fine(tf)
    pp = ((p.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]; gg = ((g.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]
    fg = float(LE.fg_mse(pp, gg).mean())
    kk = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=p.dtype, device=p.device)[None, None]
    sh = float(F.conv2d((p.clamp(-1, 1) + 1) * 0.5, kk, padding=1).var().item())
    return fg, sh, (p if decode else None)


class ReplayBuffer:
    def __init__(self, size=1024): self.buf, self.size = [], size
    def sample(self, n):
        if len(self.buf) < n: return None
        idx = torch.randint(0, len(self.buf), (n,))
        return (torch.stack([self.buf[i][0] for i in idx]), torch.stack([self.buf[i][1] for i in idx]))
    def add(self, yc, yf):
        for a, b in zip(yc.detach().cpu(), yf.detach().cpu()): self.buf.append((a, b))
        self.buf = self.buf[-self.size:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="logs_rhcp/hier_ae.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n_ctx", type=int, default=6)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_mcmc", type=int, default=3)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    torch.manual_seed(0)
    ae = load_hier(args.ae)
    data = HierData(ae, "data/loop_train_3000x100.npy", n_ctx=args.n_ctx, k=args.k)
    ebt = HierEnergy(args.n_ctx).to(DEV)
    # hierarchical REGRESSION baseline (the critical control: hierarchy WITHOUT energy)
    reg_c = nn.Sequential(_net(LAT * args.n_ctx, 0, 4), nn.Conv2d(128, LAT, 1)).to(DEV)
    reg_f = nn.Sequential(_net(LAT * (args.n_ctx + 1), 0, 8), nn.Conv2d(128, LAT, 1)).to(DEV)
    oe = torch.optim.Adam([{"params": [p for n, p in ebt.named_parameters() if n != "log_alpha"], "lr": 3e-4},
                           {"params": [ebt.log_alpha], "lr": 3e-2, "weight_decay": 0.0}])
    orc = torch.optim.Adam(list(reg_c.parameters()) + list(reg_f.parameters()), lr=3e-4)
    rb = ReplayBuffer(1024)
    print(f"HIER-EBT (consistency-coupled) | n_ctx={args.n_ctx} k={args.k} steps={args.steps}", flush=True)
    for s in range(1, args.steps + 1):
        cc, cf, tc, tf, _ = data.batch(args.batch)
        B = cc.shape[0]
        # S2 EBT: replay warm-start + no-detach + final loss + randomized steps
        y0 = None
        rs = rb.sample(B // 2)
        if rs is not None:
            y0 = (torch.cat([rs[0].to(DEV), torch.randn(B - B // 2, LAT, 4, 4, device=DEV)]),
                  torch.cat([rs[1].to(DEV), torch.randn(B - B // 2, LAT, 8, 8, device=DEV)]))
        ns = args.n_mcmc + int(torch.randint(0, 3, (1,)))
        (yc, yf), _, le = descend_hier(ebt, cc, cf, ns, tc=tc, tf=tf, mode="s2", langevin=0.1, randomize=True,
                                       train=True, y_init=y0)
        oe.zero_grad(set_to_none=True); le.backward(); torch.nn.utils.clip_grad_norm_(ebt.parameters(), 1.0); oe.step()
        rb.add(yc, yf)
        # hier-regression baseline
        pc = reg_c(cc); pf = reg_f(torch.cat([cf, F.interpolate(pc.detach(), size=(8, 8), mode="nearest")], 1))
        lb = F.smooth_l1_loss(pc, tc) + F.smooth_l1_loss(pf, tf)
        orc.zero_grad(set_to_none=True); lb.backward(); orc.step()
        if s % 1000 == 0:
            print(f"  [step {s}] EBT {float(le.detach()):.4f}  hier-reg {float(lb.detach()):.4f}  alpha {float(ebt.alpha()):.1f}", flush=True)
        if s % 2000 == 0:   # periodic monitor: fg-MSE @ 8 steps + sample image (review in parallel)
            vcc, vcf, vtc, vtf, _, vlast, vgt = data.batch(256, train=False, return_frames=True)
            (vyc, vyf), _, _ = descend_hier(ebt, vcc, vcf, 8, langevin=0.05, train=False)
            vfg, vsh, vdec = _metrics(ae, vyf, vtf)
            print(f"      >> monitor@{s}: EBT fg-MSE {vfg:.4f} sharp {vsh:.4f}", flush=True)
            save_samples(f"logs_rhcp/hier_ebt_step{s}.png", [("last", vlast), ("EBT", vdec), ("GT", vgt)], n=8)

    # ---- eval: System-2 sweep (does think-longer help NOW with the coupled hierarchy?) + baselines + sample images ----
    cc, cf, tc, tf, change, lastf, gtf = data.batch(512, train=False, return_frames=True)
    med = change.median(); hard = change > med
    print(f"\n  HIER-EBT System-2 sweep (fine fg-MSE):  {'steps':>6}|{'all':>8}|{'hard':>8}|{'sharp':>8}")
    imgs = None
    for ns in [1, 2, 4, 8, 16, 32]:
        (yc, yf), _, _ = descend_hier(ebt, cc, cf, ns, langevin=0.05, train=False)
        fa, sh, dec = _metrics(ae, yf, tf)
        fh, _, _ = _metrics(ae, yf[hard], tf[hard], decode=False)
        print(f"                                          {ns:>6}|{fa:>8.4f}|{fh:>8.4f}|{sh:>8.4f}")
        if ns == 16:
            imgs = dec
    # baselines
    with torch.no_grad():
        pc = reg_c(cc); pf = reg_f(torch.cat([cf, F.interpolate(pc, size=(8, 8), mode="nearest")], 1))
    rf, rsh, rdec = _metrics(ae, pf, tf)
    pf0 = lastf  # persistence = last frame
    pp = ((pf0.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]; gg = ((gtf.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]
    persist_fg = float(LE.fg_mse(pp, gg).mean())
    print(f"\n  BASELINES (fine fg-MSE): hier-regression {rf:.4f} (sharp {rsh:.4f})  |  persistence {persist_fg:.4f}")
    save_samples("logs_rhcp/hier_ebt_samples.png",
                 [("last-ctx", lastf), ("hier-EBT(16)", imgs), ("hier-reg", rdec), ("GT", gtf)], n=8)
    print("  => review logs_rhcp/hier_ebt_samples.png (last-ctx | hier-EBT | hier-regression | GT)")


if __name__ == "__main__":
    main()
