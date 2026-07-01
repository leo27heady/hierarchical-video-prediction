"""DECISIVE CONTROL: run the *reference* EBT (Gladstone/Alonso EBT repo) architecture + exact S2
recipe on OUR loopmom latents, to settle whether the mean-collapse we see is OUR reimplementation's
fault or an EBT-method limitation at our scale.

FAITHFULNESS (what is genuinely THEIRS vs approximated):
  THEIRS, verbatim (imported unchanged from C:/Users/leoni/Documents/projects/EBT):
    * model.ar_ebt_time_embed.EBTTimeConcat  -- the whole transformer: RMSNorm blocks, RoPE,
      causal+superdiagonal AR attention, prepended mcmc-step time-embedding TOKEN, bias-free
      Linear(dim,1) energy head, per-predicted-frame scalar energy. NOT reimplemented.
    * The EBT_VID.forward MCMC descent logic, transcribed line-for-line below (corrupt->randn init,
      per-step alpha randomize, langevin add, cat(real,pred), autograd.grad wrt ORIGINAL (pre-projection)
      embeddings, no_mcmc_detach single graph, truncate_mcmc final-step-only recon, step-count
      randomize with min, eval expansion).  Same control flow as EBT_VID.forward / forward_loss_wrapper.
    * model.replay_buffer.CausalReplayBuffer -- imported unchanged.
    * Recipe values from job_scripts/vid/pretrain/ebt_s2_vae.sh:
      random_noise init, langevin=1, truncate_mcmc, no_mcmc_detach, mcmc_replay_buffer,
      randomize_mcmc_step_size_scale=2, randomize_mcmc_num_steps=2 (min 2), mcmc_num_steps=1,
      SmoothL1 recon, lr 4e-4, batch 32, grad-clip 1.0, weight_decay 0.01, warmup.
      alpha (mcmc_step_size): their 3000 is calibrated to SD-VAE grad scale; we calibrate to OUR
      latent/grad scale by THEIR OWN principle (alpha*|gradE|*nsteps ~ init->target move on unit latents)
      and also allow forcing a literal value.  <-- flagged as the one recipe number we retune.

  APPROXIMATED / adapted to our data (unavoidable, flagged):
    * Tokenization: reference tokenizes each FRAME as one D-dim token (flattened SD-VAE latent, 3136->D
      via encoder_down_projection).  We do the SAME frame-as-token scheme: each frame's (coarse 16*4*4 +
      fine 16*8*8)=1280-dim standardized latent -> one token via a learned Linear(1280,D) that plays the
      EXACT role of their encoder_down_projection (grad is taken wrt the 1280-d ORIGINAL embedding, like
      theirs).  Descent is in original 1280-d latent space; un-standardize+split back to (coarse,fine)
      for decode.  This is the faithful frame-as-token mapping.
    * Actions: the reference video task has no actions; ours needs 4 bridging actions in {-1,0,1}. We add
      a learned action embedding (sum of 4 per-slot action embeddings) into the transformer's input token
      embeddings for the predicted (candidate) frames only, after down-projection.  Minimal, flagged.

PROBE (mirror hvp_rev/ebt_fixed.py::probe): E(target) vs E(mean-latent) vs E(randn); which is LOWEST
(target=good, mean=collapsed); cold-init eval prediction distance to target vs to mean; decoded fg-sharp
(GT~0.25, <0.08 collapsed). Saves GT-vs-pred PNG to logs_rhcp/ebt_reference_*.png.

  venv/Scripts/python.exe -u -m hvp_rev.ebt_reference_control --steps 2000 --bf16
"""
from __future__ import annotations
import argparse, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- make the genuine reference repo importable (stubs shim its heavy deps) ---
_STUBS = os.environ.get("EBT_STUBS")
_EBT_REPO = os.environ.get("EBT_REPO", "C:/Users/leoni/Documents/projects/EBT")
if _STUBS and _STUBS not in sys.path:
    sys.path.insert(0, _STUBS)
if _EBT_REPO not in sys.path:
    sys.path.insert(0, _EBT_REPO)

from model.ar_ebt_time_embed import EBTTimeConcat            # THEIRS, unchanged
from model.model_utils import EBTModelArgs                   # THEIRS, unchanged
from model.replay_buffer import CausalReplayBuffer           # THEIRS, unchanged

from hvp_rev import hier_convae, viz
from hvp_rev.step1_mech import ActionHierData, decode01, fg_sharp

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAT = 16
CDIM = LAT * 4 * 4     # 256  coarse flat
FDIM = LAT * 8 * 8     # 1024 fine flat
TOKDIM = CDIM + FDIM   # 1280 per-frame original embedding dim (like reference 3136)


def _rms(t):
    return float(t.pow(2).mean().sqrt())


# ============================================================================================
# Reference-faithful wrapper: builds per-frame tokens, runs THEIR transformer + THEIR MCMC loop.
# ============================================================================================
class RefEBT(nn.Module):
    def __init__(self, d=256, n_layers=6, n_heads=8, n_frames_ctx=6, k=4, sc=0.1377, sf=0.2439,
                 mcmc_num_steps=1, randomize_num_steps=2, randomize_num_steps_min=2,
                 randomize_step_scale=2.0, langevin=1.0, alpha=1.0, ffn_mult=None):
        """ffn_mult: FFN width multiplier. The original 2k-step control ran None (=1x) -- a real
        capacity gap vs the paper's train_model.py default of 4.0; the scaled runs pass 4.0."""
        super().__init__()
        self.d = d
        self.n_ctx = n_frames_ctx
        self.k = k
        self.sc, self.sf = float(sc), float(sf)
        # transformer input sequence has (n_ctx real frames) + (n_ctx predicted frames); +1 for time tok.
        # reference max_seq_len = context_length+1 (+1 more for time_embed).  context_length here = n_ctx.
        args = EBTModelArgs(dim=d, n_layers=n_layers, n_heads=n_heads,
                            max_seq_len=n_frames_ctx + 2, weight_initialization="xavier",
                            ebt_norm="rms", ffn_dim_multiplier=ffn_mult, ebt_act_func="silu",
                            weight_initialization_gain=1.0)
        # max_mcmc_steps must cover the highest mcmc_step index used (randomize expands final landscape)
        self.transformer = EBTTimeConcat(params=args, max_mcmc_steps=max(mcmc_num_steps, 2))
        # encoder_down_projection analog: 1280-d original frame embedding -> d (THEIR encoder_down_projection)
        self.encoder_down_projection = nn.Linear(TOKDIM, d, bias=True)
        nn.init.xavier_normal_(self.encoder_down_projection.weight)
        nn.init.zeros_(self.encoder_down_projection.bias)
        # action embedding added to predicted (candidate) tokens after down-projection (approx, flagged)
        self.action_embed = nn.Embedding(3 * k, d)   # 4 slots x {-1,0,1}
        nn.init.normal_(self.action_embed.weight, std=0.02)

        # --- reference recipe hyperparams ---
        self.mcmc_num_steps = mcmc_num_steps
        self.randomize_num_steps = randomize_num_steps
        self.randomize_num_steps_min = randomize_num_steps_min
        self.randomize_step_scale = randomize_step_scale
        # alpha (mcmc_step_size), fixed like S2 (mcmc_step_size_learnable is NOT set in ebt_s2_vae.sh)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=False)
        self.langevin_std = nn.Parameter(torch.tensor(float(langevin)), requires_grad=False)

    # ---- standardize (their x0.18215): operate in unit-std latent space ----
    def _flat_std(self, zc, zf):
        # zc:(B,16,4,4) zf:(B,16,8,8) -> (B, 1280) standardized frame token (original embedding)
        B = zc.shape[0]
        c = (zc / self.sc).reshape(B, -1)
        f = (zf / self.sf).reshape(B, -1)
        return torch.cat([c, f], dim=1)

    def _unflat(self, tok):
        # (B,1280) standardized -> (zc,zf) UN-standardized
        B = tok.shape[0]
        c = tok[:, :CDIM].reshape(B, LAT, 4, 4) * self.sc
        f = tok[:, CDIM:].reshape(B, LAT, 8, 8) * self.sf
        return c, f

    def _seq_from_ctx(self, ctx_c, ctx_f, tgt_c, tgt_f):
        """Build the (B, n_ctx+1, 1280) real frame-token sequence [ctx0..ctx_{n-1}, tgt]."""
        B, n = ctx_c.shape[0], self.n_ctx
        toks = []
        for i in range(n):
            toks.append(self._flat_std(ctx_c[:, i], ctx_f[:, i]))
        toks.append(self._flat_std(tgt_c, tgt_f))                  # target slot at the end
        return torch.stack(toks, dim=1)                            # (B, n+1, 1280)

    def _add_action(self, down_embeds_pred, actions):
        """down_embeds_pred: (B, S, d) predicted-token embeddings; add action embedding (approx)."""
        B, S, d = down_embeds_pred.shape
        # sum over the 4 action slots; offset each slot into its own block of the embedding table
        idx = (actions + 1) + 3 * torch.arange(self.k, device=actions.device).view(1, -1)  # (B,k) in [0,3k)
        a_emb = self.action_embed(idx).sum(dim=1)                  # (B, d)
        return down_embeds_pred + a_emb.unsqueeze(1)               # broadcast over predicted tokens

    # ============================================================================
    # MCMC descent -- transcribed from EBT_VID.forward (their S2 control flow).
    #   real_original: (B, S, 1280) original real embeddings input (frames 0..n-1)
    #   returns predicted_embeddings_list, predicted_energies_list (both in ORIGINAL 1280-d space)
    # ============================================================================
    def forward(self, real_original, actions, learning, replay_pred=None, no_randomness=True):
        B, S, _ = real_original.shape
        alpha = torch.clamp(self.alpha, min=1e-4)
        if not no_randomness and self.randomize_step_scale != 1:
            expanded = alpha.expand(B, S, 1)
            low, high = alpha / self.randomize_step_scale, alpha * self.randomize_step_scale
            alpha = low + torch.rand_like(expanded) * (high - low)   # per-token alpha (their randomize)
        lang = torch.clamp(self.langevin_std, min=1e-6)

        real_down = self.encoder_down_projection(real_original)      # (B,S,d)

        # corrupt_embeddings: random_noise init (THEIR denoising_initial_condition='random_noise')
        orig_pred = torch.randn_like(real_original)                 # (B,S,1280) cold randn in unit space
        if replay_pred is not None:
            orig_pred[B - replay_pred.shape[0]:] = replay_pred

        # build mcmc_steps schedule (their randomize_mcmc_num_steps logic; final-landscape expansion)
        mcmc_steps = []
        for step in range(self.mcmc_num_steps):
            if not no_randomness and self.randomize_num_steps > 0:
                if step == (self.mcmc_num_steps - 1):
                    min_steps = 1 if self.randomize_num_steps_min == 0 else self.randomize_num_steps_min
                    repeats = int(torch.randint(min_steps, self.randomize_num_steps + 2, (1,)).item())
                    mcmc_steps.extend([step] * repeats)
                else:
                    mcmc_steps.append(step)
            elif no_randomness and self.randomize_num_steps > 0:      # eval: max steps on final landscape
                if step == (self.mcmc_num_steps - 1):
                    mcmc_steps.extend([step] * (self.randomize_num_steps + 1))
                else:
                    mcmc_steps.append(step)
            else:
                mcmc_steps.append(step)

        pred_list, energy_list = [], []
        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                # no_mcmc_detach=True (S2): DO NOT detach -> single long graph
                orig_pred = orig_pred.requires_grad_().reshape(B, S, TOKDIM)
                pred_down = self.encoder_down_projection(orig_pred)   # (B,S,d)
                pred_down = self._add_action(pred_down, actions)      # inject actions (approx)

                if float(self.langevin_std) != 0:
                    pred_down = pred_down + torch.randn_like(pred_down.detach()) * lang

                all_emb = torch.cat([real_down, pred_down], dim=1)    # (B, 2S, d)
                energy = self.transformer(all_emb, start_pos=0, mcmc_step=mcmc_step)  # (B,S,1)
                energy = energy.reshape(-1, 1)
                energy_list.append(energy)

                # truncate_mcmc=True: create_graph only on final step (S2)
                cg = bool(learning) and (i == len(mcmc_steps) - 1)
                grad = torch.autograd.grad([energy.sum()], [orig_pred], create_graph=cg)[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise ValueError("NaN/Inf grad in MCMC")

                a = alpha if isinstance(alpha, torch.Tensor) and alpha.dim() == 0 else alpha
                orig_pred = orig_pred - a * grad
                pred_list.append(orig_pred)
        return pred_list, energy_list

    # ---- energy of an EXPLICIT candidate (for the probe), at a given mcmc_step ----
    def energy_of(self, real_original, cand_original, actions, mcmc_step=0):
        real_down = self.encoder_down_projection(real_original)
        pred_down = self.encoder_down_projection(cand_original)
        pred_down = self._add_action(pred_down, actions)
        all_emb = torch.cat([real_down, pred_down], dim=1)
        energy = self.transformer(all_emb, start_pos=0, mcmc_step=mcmc_step)  # (B,S,1)
        return energy  # per predicted frame


# ============================================================================================
# Loss wrapper (their forward_loss_wrapper + CausalReplayBuffer), truncate_mcmc final-step recon.
# ============================================================================================
class RefTrainer:
    def __init__(self, model: RefEBT, use_replay=True, replay_size=192, replay_pct=0.5, batch=32):
        self.m = model
        self.crit = nn.SmoothL1Loss(beta=1.0)                 # their reconstruction_criterion
        self.use_replay = use_replay
        if use_replay:
            self.buf = CausalReplayBuffer(max_size=replay_size, sample_size=int(batch * replay_pct))

    def _full_seq(self, ctx_c, ctx_f, tgt_c, tgt_f):
        # (B, n+1, 1280): frames [ctx0..ctx_{n-1}, tgt].  real_input = [:-1], gt = [1:].
        return self.m._seq_from_ctx(ctx_c, ctx_f, tgt_c, tgt_f)

    def train_step(self, ctx_c, ctx_f, tgt_c, tgt_f, actions):
        full = self._full_seq(ctx_c, ctx_f, tgt_c, tgt_f)     # (B, n+1, 1280)
        if self.use_replay:
            # Faithful replay: get_batch samples records with their CO-STORED actions so replayed
            # embeddings keep the actions they were produced under (fixes the action/embedding mismatch).
            real_in, replay_pred, gt, replay_acts = self._get_batch_with_actions(full, actions)
            acts = actions if replay_acts is None else torch.cat([actions, replay_acts], dim=0)
            pred_list, energy_list = self.m(real_in, acts, learning=True,
                                            replay_pred=replay_pred, no_randomness=False)
            self._update_with_actions(full.detach(), pred_list[-1].detach(), actions.detach())
        else:
            real_in = full[:, :-1]
            gt = full[:, 1:]
            pred_list, energy_list = self.m(real_in, actions, learning=True, no_randomness=False)

        # truncate_mcmc: only final step recon
        recon = self.crit(pred_list[-1], gt)
        return recon

    # ---- CausalReplayBuffer with actions co-stored per record (faithful indexing) ----
    def _get_batch_with_actions(self, all_data, fresh_actions):
        buf = self.buf.buffer
        fresh_inputs = all_data[:, :-1]
        fresh_gt = all_data[:, 1:]
        if not buf:
            return fresh_inputs, None, fresh_gt, None
        import random
        n = min(self.buf.sample_size, len(buf))
        samples = random.sample(buf, n)
        replay_inputs = torch.cat([s["all_data"][:, :-1] for s in samples], dim=0)
        replay_gt = torch.cat([s["all_data"][:, 1:] for s in samples], dim=0)
        replay_pred = torch.cat([s["predicted_outputs"] for s in samples], dim=0)
        replay_acts = torch.cat([s["actions"] for s in samples], dim=0)
        combined_inputs = torch.cat([fresh_inputs, replay_inputs], dim=0)
        combined_gt = torch.cat([fresh_gt, replay_gt], dim=0)
        return combined_inputs, replay_pred, combined_gt, replay_acts

    def _update_with_actions(self, all_data, final_pred, actions):
        for i in range(all_data.size(0)):
            self.buf.add({
                "all_data": all_data[i].unsqueeze(0),
                "predicted_outputs": final_pred[i].unsqueeze(0),
                "actions": actions[i].unsqueeze(0),
            })


# ============================================================================================
# PROBE  (mirror hvp_rev/ebt_fixed.py::probe)
# ============================================================================================
@torch.no_grad()
def probe(m: RefEBT, trainer, test, ae, tag, k_eval_steps, probe_mcmc_step=0):
    """probe_mcmc_step: which landscape's energy to probe (S2: 0 is the only one; S1 has 2 --
    pass mcmc_num_steps-1 to probe the FINAL landscape, the one inference descends last)."""
    b = test.batch(64, train=False, return_frames=True, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    full = m._seq_from_ctx(cc, cf, tc, tf)            # (B, n+1, 1280) standardized
    real_in = full[:, :-1]                            # frames 0..n-1
    gt_seq = full[:, 1:]                              # frames 1..n (last = target)

    # candidates for the FINAL predicted token (the target frame):
    tgt_tok = full[:, -1]                             # (B,1280) standardized target
    mean_tok = tgt_tok.mean(0, keepdim=True).expand_as(tgt_tok).contiguous()
    randn_tok = torch.randn_like(tgt_tok)

    def E_last(cand_last):
        # build a candidate sequence = gt for tokens 1..n-1, candidate for the last (target) token
        cand = gt_seq.clone()
        cand[:, -1] = cand_last
        with torch.enable_grad():
            e = m.energy_of(real_in, cand, a, mcmc_step=probe_mcmc_step)   # (B, S, 1)
        return float(e[:, -1].mean())                        # energy of the TARGET predicted frame

    Et = E_last(tgt_tok)
    Em = E_last(mean_tok)
    Er = E_last(randn_tok)
    lo = min([("target", Et), ("mean", Em), ("randn", Er)], key=lambda x: x[1])[0]

    # cold-init eval descent (langevin 0 at eval -> temporarily zero it), get predicted target token
    saved_lang = float(m.langevin_std)
    m.langevin_std.data.fill_(0.0)
    with torch.enable_grad():
        pred_list, _ = m(real_in, a, learning=False, no_randomness=True)
    m.langevin_std.data.fill_(saved_lang)
    pred_last = pred_list[-1][:, -1].detach()                # (B,1280) predicted target token
    d_tgt = _rms(pred_last - tgt_tok)
    d_mean = _rms(mean_tok - tgt_tok)

    # decode predicted target -> fine latent -> pixels
    _, pred_zf = m._unflat(pred_last)
    pred01 = decode01(ae, pred_zf)
    gt01 = decode01(ae, tf)
    sh = fg_sharp(pred01, gt01)

    print(f"  [{tag}] landscape E(last frame): tgt {Et:+.3f} | mean {Em:+.3f} | randn {Er:+.3f} "
          f"-> LOWEST={lo} {'[GOOD]' if lo=='target' else '[BAD]'}  | cold ns{k_eval_steps} "
          f"d_tgt {d_tgt:.3f} (mean->tgt {d_mean:.3f}) | fg-sharp {sh:.4f} (GT~0.25) "
          f"{'[COLLAPSED]' if sh < 0.08 else '[HAS STRUCTURE]'}", flush=True)
    os.makedirs("logs_rhcp", exist_ok=True)
    viz.save_samples(f"logs_rhcp/ebt_reference_{tag}.png",
                     [("GT", gt01 * 2 - 1), ("REF-EBT", pred01 * 2 - 1)], n=8)
    return lo, sh, Et, Em, Er, d_tgt, d_mean


def calibrate_alpha(m: RefEBT, cc, cf, tc, tf, a, k_eval_steps):
    """alpha s.t. alpha*|gradE|*nsteps ~ init->target move (~sqrt2 in unit space), THEIR principle."""
    full = m._seq_from_ctx(cc, cf, tc, tf)
    real_in = full[:, :-1]
    B, S, _ = real_in.shape
    cand = torch.randn_like(full[:, 1:]).requires_grad_(True)
    e = m.energy_of(real_in, cand, a, mcmc_step=0)
    g = torch.autograd.grad(e.sum(), cand)[0]
    gmag = _rms(g)
    target_move = 1.4
    alpha = target_move / (max(k_eval_steps, 1) * max(gmag, 1e-6))
    return alpha, gmag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.0, help="0 => auto-calibrate to our latent scale")
    ap.add_argument("--langevin", type=float, default=1.0)
    ap.add_argument("--no_replay", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)

    import hvp_rev.step1_mech as M
    M._BF16 = bool(args.bf16)

    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=6, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=6, k=4, n_val=250)
    b0 = data.batch(1024, train=True)
    sc = _rms(b0["tgt_c"] - b0["tgt_c"].mean())
    sf = _rms(b0["tgt_f"] - b0["tgt_f"].mean())
    print(f"[standardize] coarse std {sc:.4f}  fine std {sf:.4f} -> unit space (their x0.18215 analog)",
          flush=True)

    m = RefEBT(d=args.d, n_layers=args.depth, n_heads=args.heads, n_frames_ctx=6, k=4,
               sc=sc, sf=sf, mcmc_num_steps=1, randomize_num_steps=2, randomize_num_steps_min=2,
               randomize_step_scale=2.0, langevin=args.langevin, alpha=1.0).to(DEV)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    # eval expands final landscape to randomize_num_steps+1 = 3 descent steps
    k_eval_steps = m.randomize_num_steps + 1

    bcal = data.batch(64, train=True)
    a_auto, gmag = calibrate_alpha(m, bcal["ctx_c"], bcal["ctx_f"], bcal["tgt_c"], bcal["tgt_f"],
                                   bcal["actions"], k_eval_steps)
    m.alpha.data.fill_(a_auto if args.alpha == 0 else args.alpha)
    print(f"[model] REF-EBT (genuine EBTTimeConcat) params {n_params:.2f}M | init |gradE|={gmag:.4f} "
          f"auto-alpha={a_auto:.3f} -> using alpha={float(m.alpha):.3f} (fixed, S2) | langevin={args.langevin} "
          f"| replay={'off' if args.no_replay else 'on'} | eval-steps {k_eval_steps}", flush=True)

    trainer = RefTrainer(m, use_replay=(not args.no_replay), batch=args.batch)
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01)
    warm = 200

    m.eval(); probe(m, trainer, test, ae, "step0", k_eval_steps); m.train()
    run = 0.0
    for s in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = 4e-4 * min(1.0, s / warm)
        bt = data.batch(args.batch, train=True)
        loss = trainer.train_step(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            opt.step()
            run += float(loss.detach())
        if s % 300 == 0:
            print(f"\n[train {s}/{args.steps}] smooth_l1 {run/300:.4f} grad-norm {gn:.3f}", flush=True)
            run = 0.0
            m.eval(); probe(m, trainer, test, ae, f"step{s}", k_eval_steps); m.train()
    print("\nREF-EBT CONTROL DONE -> logs_rhcp/ebt_reference_step*.png (GT | REF-EBT)", flush=True)


if __name__ == "__main__":
    main()
