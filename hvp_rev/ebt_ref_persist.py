"""NON-CIRCULAR DECISIVE TEST: reference EBT on the REAL panned next-frame task with PERSISTENCE
(most_recent_embedding) init instead of random_noise -- at BOTH train and eval.

Prior finding (ebt_ref_verify*.py): the reference energy landscape is CORRECT (E(target) lowest,
conditions on context, argmin-diag 8/8) but cold RANDN-init MCMC descent lands in the mean's basin
=> blank. Warm-start from near the target commits. But the copy-task warm start was CIRCULAR
(init == target).

Here we close the loop on the REAL task, where the target (frame t+4, viewport panned by the action)
genuinely DIFFERS from the last context frame. We set the descent init = the LAST CONTEXT FRAME's
latent (the reference's denoising_initial_condition="most_recent_embedding"), at train AND eval, and
ask: does persistence-init descent COMMIT to the actual panned target (d_tgt small, sharp), or does
it (a) just sit at persistence (d_tgt ~= d_persist), or (b) flee to the mean?

Fair baseline: a plain feed-forward Regressor (hvp_rev.step1_mech.Regressor) trained the same #steps.

    venv/Scripts/python.exe -u -m hvp_rev.ebt_ref_persist --steps 1800 --batch 32 --bf16
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import torch.nn as nn

from hvp_rev.ebt_reference_control import RefEBT, RefTrainer, calibrate_alpha, _rms, DEV, TOKDIM
from hvp_rev import hier_convae, viz
from hvp_rev.step1_mech import (ActionHierData, decode01, fg_sharp, fg_mse_px_frames,
                                Regressor, train_regressor, LAT)


def log(*a): print(*a, flush=True)


# ============================================================================================
# RefEBT variant: MCMC descent initialised from a PROVIDED persistence tensor (most_recent_embedding)
# rather than randn. Everything else (transformer, schedule, langevin, truncate_mcmc) is unchanged.
# ============================================================================================
class RefEBTPersist(RefEBT):
    def forward(self, real_original, actions, learning, replay_pred=None, no_randomness=True,
                persist_init=None):
        B, S, _ = real_original.shape
        alpha = torch.clamp(self.alpha, min=1e-4)
        if not no_randomness and self.randomize_step_scale != 1:
            expanded = alpha.expand(B, S, 1)
            low, high = alpha / self.randomize_step_scale, alpha * self.randomize_step_scale
            alpha = low + torch.rand_like(expanded) * (high - low)
        lang = torch.clamp(self.langevin_std, min=1e-6)
        real_down = self.encoder_down_projection(real_original)

        # ---- INIT = most_recent_embedding (persistence) instead of random_noise ----
        if persist_init is not None:
            orig_pred = persist_init.clone()                       # (B,S,1280)
        else:
            orig_pred = torch.randn_like(real_original)
        if replay_pred is not None:
            orig_pred[B - replay_pred.shape[0]:] = replay_pred

        mcmc_steps = []
        for step in range(self.mcmc_num_steps):
            if not no_randomness and self.randomize_num_steps > 0:
                if step == (self.mcmc_num_steps - 1):
                    min_steps = 1 if self.randomize_num_steps_min == 0 else self.randomize_num_steps_min
                    repeats = int(torch.randint(min_steps, self.randomize_num_steps + 2, (1,)).item())
                    mcmc_steps.extend([step] * repeats)
                else:
                    mcmc_steps.append(step)
            elif no_randomness and self.randomize_num_steps > 0:
                if step == (self.mcmc_num_steps - 1):
                    mcmc_steps.extend([step] * (self.randomize_num_steps + 1))
                else:
                    mcmc_steps.append(step)
            else:
                mcmc_steps.append(step)

        pred_list, energy_list = [], []
        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                orig_pred = orig_pred.requires_grad_().reshape(B, S, TOKDIM)
                pred_down = self.encoder_down_projection(orig_pred)
                pred_down = self._add_action(pred_down, actions)
                if float(self.langevin_std) != 0:
                    pred_down = pred_down + torch.randn_like(pred_down.detach()) * lang
                all_emb = torch.cat([real_down, pred_down], dim=1)
                energy = self.transformer(all_emb, start_pos=0, mcmc_step=mcmc_step)
                energy = energy.reshape(-1, 1)
                energy_list.append(energy)
                cg = bool(learning) and (i == len(mcmc_steps) - 1)
                grad = torch.autograd.grad([energy.sum()], [orig_pred], create_graph=cg)[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise ValueError("NaN/Inf grad in MCMC")
                orig_pred = orig_pred - alpha * grad
                pred_list.append(orig_pred)
        return pred_list, energy_list


# ---- persistence-init tensor for a batch: last-context frame standardized token, broadcast over slots ----
def make_persist_init(m: RefEBT, seq_full):
    """seq_full: (B, n_ctx+1, 1280). real_in slots predict frames 1..n; init each predicted slot with
    the PREVIOUS real frame's token (most_recent_embedding: predicted-frame j starts from real frame j).
    For the LAST (target) slot this is the last context frame -> persistence."""
    real_in = seq_full[:, :-1]                 # (B,n_ctx,1280) frames 0..n-1
    return real_in.clone()                     # slot j initialised from real frame j (its most-recent)


# ============================================================================================
# Persistence-init trainer (mirrors RefTrainer but threads persist_init through the model call).
# ============================================================================================
class PersistTrainer(RefTrainer):
    def train_step(self, ctx_c, ctx_f, tgt_c, tgt_f, actions):
        full = self._full_seq(ctx_c, ctx_f, tgt_c, tgt_f)          # (B,n+1,1280)
        if self.use_replay:
            real_in, replay_pred, gt, replay_acts = self._get_batch_with_actions(full, actions)
            acts = actions if replay_acts is None else torch.cat([actions, replay_acts], dim=0)
            persist = make_persist_init(self.m, torch.cat([real_in, gt[:, -1:]], dim=1)) \
                if replay_pred is None else None
            # when replay is active the combined batch mixes fresh+replay; persistence init only
            # cleanly defined for fresh rows -> build from real_in directly (it IS frames 0..n-1)
            persist = real_in.clone()
            pred_list, energy_list = self.m(real_in, acts, learning=True, replay_pred=replay_pred,
                                            no_randomness=False, persist_init=persist)
            self._update_with_actions(full.detach(), pred_list[-1].detach(), actions.detach())
        else:
            real_in = full[:, :-1]; gt = full[:, 1:]
            persist = real_in.clone()
            pred_list, energy_list = self.m(real_in, actions, learning=True, no_randomness=False,
                                            persist_init=persist)
        recon = self.crit(pred_list[-1], gt)
        return recon


# ============================================================================================
# EVAL PROBE from persistence init.
# ============================================================================================
@torch.no_grad()
def probe_persist(m: RefEBTPersist, test, ae, reg, tag):
    b = test.batch(64, train=False, return_frames=True, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    full = m._seq_from_ctx(cc, cf, tc, tf)
    real_in = full[:, :-1]
    gt_seq = full[:, 1:]
    tgt_tok = full[:, -1]                                          # standardized true panned target
    last_ctx_tok = m._flat_std(cc[:, -1], cf[:, -1])              # persistence token
    mean_tok = tgt_tok.mean(0, keepdim=True).expand_as(tgt_tok).contiguous()

    # distances (reference geometry, all in standardized token space)
    d_persist_tgt = _rms(last_ctx_tok - tgt_tok)                   # how far target is from persistence
    d_mean_tgt = _rms(mean_tok - tgt_tok)                          # how far target is from the mean

    # E(target) vs E(mean) vs E(persistence) [last-frame energy]
    def E_last(cand_last):
        cand = gt_seq.clone(); cand[:, -1] = cand_last
        with torch.enable_grad():
            e = m.energy_of(real_in, cand, a, mcmc_step=0)
        return float(e[:, -1].mean())
    Et, Em, Ep = E_last(tgt_tok), E_last(mean_tok), E_last(last_ctx_tok)
    lo = min([("target", Et), ("mean", Em), ("persist", Ep)], key=lambda x: x[1])[0]

    # ---- persistence-init eval descent (langevin off) ----
    saved = float(m.langevin_std); m.langevin_std.data.fill_(0.0)
    saved_alpha = float(m.alpha)
    real_down = m.encoder_down_projection(real_in)

    def descend_persist(alpha, nsteps):
        orig = real_in.clone().detach()                          # slot j init from real frame j
        with torch.enable_grad():
            for _ in range(nsteps):
                orig = orig.detach().requires_grad_(True)
                pd = m._add_action(m.encoder_down_projection(orig), a)
                all_emb = torch.cat([real_down.detach(), pd], dim=1)
                e = m.transformer(all_emb, start_pos=0, mcmc_step=0)
                g = torch.autograd.grad(e.sum(), orig)[0]
                orig = orig - alpha * g
        return orig[:, -1].detach()

    # sweep eval (alpha, steps): persistence init may need a SMALL alpha to not overshoot its basin
    log(f"  [persist-init descent sweep] (d_tgt / d_persist / sharp); d(persist,tgt)={_rms(last_ctx_tok-tgt_tok):.3f}")
    best = None
    for al in (0.3, 1.0, 3.0, saved_alpha):
        for ns in (2, 8, 32):
            pl = descend_persist(al, ns)
            dt = _rms(pl - tgt_tok); dp = _rms(pl - last_ctx_tok)
            _, zf_ = m._unflat(pl); sh_ = fg_sharp(decode01(ae, zf_), decode01(ae, tf))
            log(f"      alpha {al:>6.2f} steps {ns:>2}: d_tgt {dt:.3f}  d_persist {dp:.3f}  sharp {sh_:.4f}")
            # "best" = sharpest that also beats persistence distance-to-target
            score = sh_ - (0.5 if dt > _rms(last_ctx_tok - tgt_tok) else 0.0)
            if best is None or score > best[0]:
                best = (score, al, ns, pl)
    _, best_al, best_ns, pred_last = best
    log(f"  --> best persist-init descent: alpha {best_al:.2f} steps {best_ns} (used for reported numbers)")
    m.langevin_std.data.fill_(saved)

    d_tgt = _rms(pred_last - tgt_tok)
    d_mean = _rms(pred_last - mean_tok)
    d_persist = _rms(pred_last - last_ctx_tok)                    # did it just stay at persistence?

    # decode EBT prediction + persistence + regression + GT
    _, pred_zf = m._unflat(pred_last)
    ebt01 = decode01(ae, pred_zf)
    gt01 = decode01(ae, tf)
    last01 = decode01(ae, cf[:, -1])
    ebt_sharp = fg_sharp(ebt01, gt01)
    ebt_fgmse = float(fg_mse_px_frames(ebt01, gt01).mean())
    pers_sharp = fg_sharp(last01, gt01)
    pers_fgmse = float(fg_mse_px_frames(last01, gt01).mean())

    # regression baseline on the SAME batch
    zc0 = torch.zeros(cc.shape[0], LAT, 4, 4, device=DEV); zf0 = torch.zeros(cc.shape[0], LAT, 8, 8, device=DEV)
    reg.eval()
    yc, yf = reg(cc, cf, a, zc0, zf0)
    reg01 = decode01(ae, yf)
    reg_sharp = fg_sharp(reg01, gt01)
    reg_fgmse = float(fg_mse_px_frames(reg01, gt01).mean())
    reg_tok = m._flat_std(yc, yf)
    reg_dtgt = _rms(reg_tok - tgt_tok)

    log(f"\n=== PROBE [{tag}] REAL panned task, PERSISTENCE INIT ===")
    log(f"  geometry: d(persist,target)={d_persist_tgt:.3f}  d(mean,target)={d_mean_tgt:.3f}  "
        f"(target genuinely != persistence)")
    log(f"  energy:   E(target) {Et:+.3f} | E(mean) {Em:+.3f} | E(persist) {Ep:+.3f} -> LOWEST={lo} "
        f"{'[GOOD]' if lo=='target' else '[BAD]'}")
    log(f"  EBT(persist-init) pred: d_tgt {d_tgt:.3f} | d_mean {d_mean:.3f} | d_persist {d_persist:.3f}")
    log(f"     interpretation: {'COMMITS toward target' if d_tgt < 0.9*d_persist_tgt and d_tgt < d_mean else ('SITS AT PERSISTENCE' if d_persist < 0.3*d_persist_tgt else 'FLEES TO MEAN' if d_mean < d_tgt and d_mean < d_persist else 'ambiguous')}")
    log(f"  SHARPNESS (GT-roundtrip ~0.25):  EBT {ebt_sharp:.4f} | persist {pers_sharp:.4f} | "
        f"reg {reg_sharp:.4f}")
    log(f"  fg-MSE-to-GT:                    EBT {ebt_fgmse:.4f} | persist {pers_fgmse:.4f} | "
        f"reg {reg_fgmse:.4f}")
    log(f"  d_tgt(token):                    EBT {d_tgt:.3f} | persist {d_persist_tgt:.3f} | "
        f"reg {reg_dtgt:.3f}")

    os.makedirs("logs_rhcp", exist_ok=True)
    viz.save_samples("logs_rhcp/ebt_ref_realtask_persist.png",
                     [("last-frame", last01 * 2 - 1), ("EBT persist-init", ebt01 * 2 - 1),
                      ("regression", reg01 * 2 - 1), ("GT", gt01 * 2 - 1)], n=8)
    return dict(Et=Et, Em=Em, Ep=Ep, lo=lo, d_tgt=d_tgt, d_mean=d_mean, d_persist=d_persist,
                d_persist_tgt=d_persist_tgt, d_mean_tgt=d_mean_tgt,
                ebt_sharp=ebt_sharp, ebt_fgmse=ebt_fgmse, pers_sharp=pers_sharp,
                pers_fgmse=pers_fgmse, reg_sharp=reg_sharp, reg_fgmse=reg_fgmse, reg_dtgt=reg_dtgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--reg_steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--langevin", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    import hvp_rev.step1_mech as M
    M._BF16 = bool(args.bf16)
    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=6, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=6, k=4, n_val=250)
    b0 = data.batch(1024, train=True)
    sc = _rms(b0["tgt_c"] - b0["tgt_c"].mean()); sf = _rms(b0["tgt_f"] - b0["tgt_f"].mean())
    log(f"[standardize] coarse std {sc:.4f}  fine std {sf:.4f}")

    m = RefEBTPersist(d=args.d, n_layers=args.depth, n_heads=args.heads, n_frames_ctx=6, k=4,
                      sc=sc, sf=sf, mcmc_num_steps=1, randomize_num_steps=2, randomize_num_steps_min=2,
                      randomize_step_scale=2.0, langevin=args.langevin, alpha=1.0).to(DEV)
    k_eval = m.randomize_num_steps + 1
    bcal = data.batch(64, train=True)
    a_auto, gmag = calibrate_alpha(m, bcal["ctx_c"], bcal["ctx_f"], bcal["tgt_c"], bcal["tgt_f"],
                                   bcal["actions"], k_eval)
    m.alpha.data.fill_(a_auto if args.alpha == 0 else args.alpha)
    log(f"[EBT-persist] params {sum(p.numel() for p in m.parameters())/1e6:.2f}M | alpha {float(m.alpha):.3f}")

    # ---- fair baseline regressor (same task) ----
    log(f"\n-- training regressor baseline ({args.reg_steps} steps) --")
    reg = Regressor(n_ctx=6, k=4, d=args.d).to(DEV)
    train_regressor(reg, data, args.reg_steps, args.batch, val_every=1000)

    # ---- probe BEFORE EBT training (persistence init, untrained energy) ----
    m.eval(); probe_persist(m, test, ae, reg, "EBT-untrained"); m.train()

    # ---- train the reference EBT with PERSISTENCE INIT at train time ----
    log(f"\n-- training reference EBT (persistence init) ({args.steps} steps) --")
    trainer = PersistTrainer(m, use_replay=True, batch=args.batch)
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01); warm = 200
    run = 0.0
    for s in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = 4e-4 * min(1.0, s / warm)
        bt = data.batch(args.batch, train=True)
        loss = trainer.train_step(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            run += float(loss.detach())
        if s % 300 == 0:
            log(f"  [EBT-persist {s}/{args.steps}] smooth_l1 {run/300:.4f}"); run = 0.0

    m.eval(); r = probe_persist(m, test, ae, reg, "EBT-persist-final")

    # ---- verdict ----
    log("\n" + "=" * 88)
    log("=== VERDICT: persistence-init reference EBT on the REAL panned task ===")
    log("=" * 88)
    commits = (r["d_tgt"] < 0.9 * r["d_persist_tgt"] and r["ebt_sharp"] >= 0.08
               and r["d_tgt"] < r["d_mean"])
    sits = r["d_persist"] < 0.3 * r["d_persist_tgt"]
    if commits:
        verdict = ("EBT VIABLE HERE: persistence-init descent COMMITS to the sharp panned target "
                   f"(d_tgt {r['d_tgt']:.3f} < d(persist,tgt) {r['d_persist_tgt']:.3f}, sharp "
                   f"{r['ebt_sharp']:.3f}); vs regression sharp {r['reg_sharp']:.3f} d_tgt {r['reg_dtgt']:.3f}")
    elif r["ebt_sharp"] >= 0.08 and sits:
        verdict = ("EBT JUST RETURNS PERSISTENCE: output is sharp but == last frame "
                   f"(d_persist {r['d_persist']:.3f} tiny, d_tgt {r['d_tgt']:.3f} ~= d(persist,tgt) "
                   f"{r['d_persist_tgt']:.3f}); no generative gain over copying the last frame")
    elif r["ebt_sharp"] < 0.08:
        verdict = (f"EBT STILL COLLAPSES: even from persistence init the decode is blank "
                   f"(sharp {r['ebt_sharp']:.3f}), descent flees the init")
    else:
        verdict = (f"AMBIGUOUS: d_tgt {r['d_tgt']:.3f} d_mean {r['d_mean']:.3f} "
                   f"d_persist {r['d_persist']:.3f} sharp {r['ebt_sharp']:.3f}")
    log("  " + verdict)
    log(f"  grid -> logs_rhcp/ebt_ref_realtask_persist.png "
        f"[last-frame | EBT persist-init | regression | GT]")
    log("=" * 88)


if __name__ == "__main__":
    main()
