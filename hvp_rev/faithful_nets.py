"""Faithful-mechanism (small-scale) MilliVid-DiT denoiser + EBT scalar-energy transformer
for a 2-level latent video predictor.

Both nets share ONE tokenization + ONE joint self-attention over all levels' tokens -- this
mirrors MilliVid's KEY property that fine conditions on coarse via a single attention sequence
(NOT upsample-and-concat) and EBT's KEY property that context + candidate tokens live in one
sequence and only candidate energies are returned.

Interface matches the frozen AE in hvp_rev/hier_convae.py: LAT=16, grayscale, per frame a
coarse (B,16,4,4) and a fine (B,16,8,8) latent. Context = n_ctx past frames' latents.

Run the smoke test with the venv (base python lacks torch):
    venv/Scripts/python.exe -m hvp_rev.faithful_nets

======================================================================================
REAL-REPO MECHANISMS MIRRORED (file:line references)
======================================================================================
MilliVid DiT (reference checkout: source/components/dit_per_token_c.py):
  - modulate(x, shift, scale) = x*(1+scale)+shift                              (dit_per_token_c.py:60-65)
  - DiTBlock adaLN-Zero: SiLU->Linear(d,6d) chunked into
    shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp; gates init 0    (dit_per_token_c.py:84-110)
  - Unpatchify final adaLN-Zero: SiLU->Linear(d,2d)->shift,scale, then
    modulate(norm_final(x)) and a zero-init Linear(d,out)                      (dit_per_token_c.py:145-173)
  - TimestepEmbedder: t*1000, sinusoidal(embedding_size) -> MLP(d)            (dit_per_token_c.py:181-215)
MilliVid conditioning (reference checkout: source/components/adaptivity.py):
  - posemb added to x (flhw sin/cos), and c = time_embed(t) + level_embed(l)
    + action_encoder(actions), i.e. per-token conditioning summed            (adaptivity.py:186-197)
  - level_embedding = nn.Embedding(num_levels, d)                             (adaptivity.py:163)
MilliVid joint attention (reference checkout: source/model/model_video_universal.py):
  - context tokens at noise-level 0, denoised tokens at t; roles==DENOISED
    get the diffusion time, context gets t=0; ALL share one attention seq    (model_video_universal.py:123-148, 200)
  - loss only on DENOISED tokens (we only unpatchify/return target tokens)    (model_video_universal.py:143-148)
EBT (reference checkout: model/vid/ebt.py and ar_ebt_adaln.py):
  - transformer(all_embeddings) with all=cat(context, candidate); energies
    reshaped to (-1,1); grad wrt candidate via autograd.grad(create_graph=learning) (ebt.py:117-129)
  - descent step  x <- x - alpha * grad(E)                                    (ebt.py:139)
  - per-MCMC-step embedding: nn.Embedding(max_mcmc_steps, dim); adaLN blocks
    modulated by time_embeddings; FinalLayer = Linear(dim,1,bias=False)       (ar_ebt_adaln.py:492,506-513,553-561)
  - only the candidate half of energies returned: energies[:, half:]          (ar_ebt_adaln.py:609 / ebt.py:120)

INTERFACE CAVEATS / SIMPLEST-FAITHFUL CHOICES (documented for the harness):
  - The real EBT uses a hand-rolled causal + super-diagonal attention so each candidate token
    only sees context + itself (autoregressive). Here the prediction target is a SINGLE future
    frame (all target tokens co-occur), so we use plain full self-attention with a role/level
    -aware token set instead. This is the "simplest faithful" reading of "one joint attention
    sequence, only candidate energies returned". No causal mask is needed because there is one
    denoised frame, not a rolled-out sequence.
  - Context tokens carry role=context and NO diffusion-time / mcmc-step modulation (their adaLN
    conditioning uses a fixed "noise level 0" / step-0 vector), mirroring MilliVid's t=0 for
    context and EBT's context half; target tokens carry role=target and are modulated by
    embed(t)+embed(actions) (DiT) or embed(step)+embed(actions) (EBT).
  - Actions (B,k) in {-1,0,1} are shifted +1 -> {0,1,2}, embedded per-action, summed, and added
    to the conditioning vector c (a single global-per-batch action code broadcast to all target
    tokens). MilliVid uses a per-token ActionEncoder; we use one shared action code, which is the
    faithful minimal version for "the k future actions condition the denoise".
  - v-prediction: DiT returns v for both target levels (harness supplies the noise schedule).
  - EnergyTransformer.energy returns E:(B,1) = SUM over candidate tokens of per-token scalar
    energy, and is twice-differentiable wrt yc,yf (final Linear has NO zero-init so gradients
    are non-trivial from step 0; adaLN gates in the energy blocks are also non-zero-init for the
    same reason -- a zero-init energy head would give an identically-zero energy landscape).
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

# EBT needs second-order gradients (create_graph=True) through attention, but fused SDPA kernels
# (flash/mem-efficient) implement no double-backward. So by DEFAULT we use fast fused SDPA, and only
# the ONE forward that will be double-backprop'd (EBT's final training descent step) switches to an
# explicit-softmax path via double_backward_attention(). This makes reg/diff training, ALL eval, and
# EBT's detached descent steps ~3-5x faster than an always-explicit-softmax attention, while keeping
# EBT's second-order training exact.
_DOUBLE_BACKWARD = False


@contextmanager
def double_backward_attention():
    """Within this context Attention uses explicit softmax (twice-differentiable). Wrap ONLY the
    forward whose graph will be double-backprop'd (EBT's final training descent step)."""
    global _DOUBLE_BACKWARD
    prev = _DOUBLE_BACKWARD
    _DOUBLE_BACKWARD = True
    try:
        yield
    finally:
        _DOUBLE_BACKWARD = prev

LAT = 16  # matches hvp_rev/hier_convae.py


# ----------------------------------------------------------------------------------------
# Shared building blocks (mirrors MilliVid dit_per_token_c.py)
# ----------------------------------------------------------------------------------------
def modulate(x, shift, scale):
    # dit_per_token_c.py:60-65
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Continuous time in [0,1] -> d-dim embedding. Mirrors dit_per_token_c.py:181-215."""

    def __init__(self, hidden_size, embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.embedding_size = embedding_size
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t = t * 1000  # timesteps designed for [0,1000]
        half = self.embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.linspace(0, 1, half, dtype=torch.float32, device=t.device)
        )
        args = t[..., None].float() * frequencies
        args = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(args)


class Attention(nn.Module):
    """Plain multi-head full self-attention (QK-RMSNorm as in MilliVid dit_per_token_c.py:21-52,
    but without the bf16 autocast dance so it stays second-differentiable on CPU+CUDA)."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B,S,H,hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B,H,S,hd)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Fast fused SDPA by default; explicit softmax ONLY when a double-backward is coming (EBT's
        # final training step, flagged via double_backward_attention()) since fused kernels lack it.
        # Both compute softmax(QK^T/sqrt(hd))V identically (SDPA's default scale == head_dim**-0.5).
        if _DOUBLE_BACKWARD:
            # Force fp32 softmax: a no-op in fp32, a numerics guard under bf16 autocast (the softmax
            # denominator / exp are unstable in bf16). scores/matmuls may be bf16 under autocast; only
            # the softmax normalization is promoted, then cast back to v's dtype for attn @ v.
            scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = scores.float().softmax(dim=-1).to(v.dtype)
            x = attn @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, S, -1)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """adaLN-Zero DiT block with PER-TOKEN conditioning c. Mirrors dit_per_token_c.py:68-110.

    zero_gate=True zero-inits the adaLN modulation (as in MilliVid) so the block starts as
    identity -- correct for a denoiser trained from a residual. For the energy net we set
    zero_gate=False so the (untrained) energy landscape is non-degenerate.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, zero_gate=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)
        if zero_gate:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


# ----------------------------------------------------------------------------------------
# Shared tokenizer: (coarse,fine) latents -> tokens with level/frame/role embeddings.
# Mirrors MilliVid patchify (Linear latent_channels->d) + posemb_flhw + level_embedding
# + role, all summed as per-token conditioning (adaptivity.py:186-197).
# ----------------------------------------------------------------------------------------
class LatentTokenizer(nn.Module):
    COARSE_HW = 4  # (B,LAT,4,4) -> 16 tokens
    FINE_HW = 8    # (B,LAT,8,8) -> 64 tokens
    TOP_HW = 2     # (B,LAT,2,2) -> 4 tokens  [hier3 depth-3 extension; level id 2]

    def __init__(self, lat, d, n_ctx, k, n_levels=2):
        super().__init__()
        self.d = d
        self.n_ctx = n_ctx
        self.patchify = nn.Linear(lat, d)                 # project LAT channels -> d
        # n_levels=2 (default) keeps the module state-dict BIT-IDENTICAL to every existing
        # checkpoint; n_levels=3 (hier3 arms only) adds level id 2 = TOP (2x2) + its posemb.
        self.level_embedding = nn.Embedding(n_levels, d)  # 0=coarse, 1=fine, (2=top)
        self.role_embedding = nn.Embedding(2, d)          # 0=context, 1=target
        # frame-index / temporal posemb: n_ctx context frames get index 0..n_ctx-1, target = n_ctx
        self.frame_embedding = nn.Embedding(n_ctx + 1, d)
        # spatial posemb per cell, per level (learned; simplest faithful stand-in for flhw sin/cos)
        self.coarse_pos = nn.Parameter(torch.zeros(self.COARSE_HW ** 2, d))
        self.fine_pos = nn.Parameter(torch.zeros(self.FINE_HW ** 2, d))
        nn.init.normal_(self.coarse_pos, std=0.02)
        nn.init.normal_(self.fine_pos, std=0.02)
        if n_levels >= 3:
            self.top_pos = nn.Parameter(torch.zeros(self.TOP_HW ** 2, d))
            nn.init.normal_(self.top_pos, std=0.02)
        # action conditioning: k future actions in {-1,0,1} -> {0,1,2}
        self.action_embedding = nn.Embedding(3, d)
        self.k = k

    def _cells(self, z):
        # z:(B,LAT,H,W) -> tokens:(B,H*W,LAT)
        B, C, H, W = z.shape
        return z.reshape(B, C, H * W).transpose(1, 2)

    def tokenize_level(self, z, level, role, frame_idx):
        # returns x (patch-projected latents + spatial posemb), and role/level/frame ids for c
        B = z.shape[0]
        tok = self.patchify(self._cells(z))                       # (B,N,d)
        # (N,d); level 2 = top exists only on 3-level tokenizers -- branch lazily
        pos = self.coarse_pos if level == 0 else (self.fine_pos if level == 1 else self.top_pos)
        tok = tok + pos[None]
        N = tok.shape[1]
        dev = z.device
        level_id = torch.full((B, N), level, dtype=torch.long, device=dev)
        role_id = torch.full((B, N), role, dtype=torch.long, device=dev)
        frame_id = torch.full((B, N), frame_idx, dtype=torch.long, device=dev)
        return tok, level_id, role_id, frame_id

    def action_code(self, actions):
        # actions:(B,k) int in {-1,0,1} -> summed embedding (B,d)
        return self.action_embedding(actions + 1).sum(dim=1)

    def cond(self, level_id, role_id, frame_id, cond_ctx, cond_tgt, action_code):
        """Build per-token conditioning c (adaptivity.py:194): level + frame + role +
        (time/step for target OR fixed noise-level-0/step-0 for context) + action(target only)."""
        c = (
            self.level_embedding(level_id)
            + self.frame_embedding(frame_id)
            + self.role_embedding(role_id)
        )
        is_tgt = (role_id == 1)[..., None].float()
        # target tokens: cond_tgt (embed(t)+embed(actions)); context tokens: cond_ctx (level-0)
        c = c + is_tgt * cond_tgt[:, None, :] + (1 - is_tgt) * cond_ctx[:, None, :]
        c = c + is_tgt * action_code[:, None, :]
        return c


def _build_sequence(tok, ctx_c, ctx_f, yc, yf, n_ctx):
    """Assemble ONE token sequence [ctx tokens (both levels); target coarse; target fine] plus
    concatenated level/role/frame ids. Mirrors the single joint-attention sequence of MilliVid
    (context at noise-0) + EBT (context ++ candidate).

    ASYMMETRIC PER-LEVEL CONTEXT: the per-level context lengths come from the tensor shapes
    (n_ctx_c = ctx_c.shape[1], n_ctx_f = ctx_f.shape[1], with n_ctx_f <= n_ctx_c). The fine frames
    are the LAST n_ctx_f frames of the coarse window, so fine frame j carries frame-index
    n_ctx_c - n_ctx_f + j (temporal alignment preserved). The symmetric case reproduces the
    original per-frame interleaved order exactly. Target tokens carry frame index n_ctx_c; the
    tokenizer's frame_embedding (sized n_ctx+1 at construction) must cover n_ctx_c+1 entries.
    The n_ctx arg is kept for signature stability (counts come from shapes)."""
    n_ctx_c, n_ctx_f = ctx_c.shape[1], ctx_f.shape[1]
    assert n_ctx_f <= n_ctx_c, (n_ctx_c, n_ctx_f)
    f0 = n_ctx_c - n_ctx_f                     # first coarse frame that ALSO has a fine token
    xs, lvs, rls, fms = [], [], [], []
    for f in range(n_ctx_c):
        x, lv, rl, fm = tok.tokenize_level(ctx_c[:, f], level=0, role=0, frame_idx=f)
        xs.append(x); lvs.append(lv); rls.append(rl); fms.append(fm)
        if f >= f0:
            x, lv, rl, fm = tok.tokenize_level(ctx_f[:, f - f0], level=1, role=0, frame_idx=f)
            xs.append(x); lvs.append(lv); rls.append(rl); fms.append(fm)
    # target tokens (role=target, frame index = n_ctx_c)
    xc, lvc, rlc, fmc = tok.tokenize_level(yc, level=0, role=1, frame_idx=n_ctx_c)
    xf, lvf, rlf, fmf = tok.tokenize_level(yf, level=1, role=1, frame_idx=n_ctx_c)
    n_tgt_c, n_tgt_f = xc.shape[1], xf.shape[1]
    xs += [xc, xf]; lvs += [lvc, lvf]; rls += [rlc, rlf]; fms += [fmc, fmf]
    x = torch.cat(xs, dim=1)
    level_id = torch.cat(lvs, dim=1)
    role_id = torch.cat(rls, dim=1)
    frame_id = torch.cat(fms, dim=1)
    return x, level_id, role_id, frame_id, n_tgt_c, n_tgt_f


# ----------------------------------------------------------------------------------------
# DiTDenoiser (MilliVid-style v-prediction denoiser)
# ----------------------------------------------------------------------------------------
class DiTDenoiser(nn.Module):
    def __init__(self, lat=16, n_ctx=6, d=256, depth=6, heads=8, k=4):
        super().__init__()
        self.lat, self.n_ctx, self.d, self.k = lat, n_ctx, d, k
        self.tok = LatentTokenizer(lat, d, n_ctx, k)
        self.time_embedding = TimestepEmbedder(d)
        self.blocks = nn.ModuleList(
            [DiTBlock(d, heads, zero_gate=True) for _ in range(depth)]
        )
        # per-level unpatchify heads (adaLN-Zero final layer, dit_per_token_c.py:145-173)
        self.norm_c = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm_f = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_c_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final_f_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head_c = nn.Linear(d, lat)
        self.head_f = nn.Linear(d, lat)
        for ada in (self.final_c_ada, self.final_f_ada):
            nn.init.constant_(ada[-1].weight, 0)
            nn.init.constant_(ada[-1].bias, 0)
        for head in (self.head_c, self.head_f):
            nn.init.constant_(head.weight, 0)
            nn.init.constant_(head.bias, 0)

    def forward(self, yc, yf, ctx_c, ctx_f, t, actions):
        # yc:(B,16,4,4) yf:(B,16,8,8) noisy target; ctx_*:(B,n_ctx,16,H,W) clean; t:(B,); actions:(B,k)
        B = yc.shape[0]
        x, level_id, role_id, frame_id, n_tgt_c, n_tgt_f = _build_sequence(
            self.tok, ctx_c, ctx_f, yc, yf, self.n_ctx
        )
        action_code = self.tok.action_code(actions)
        # target conditioning = embed(t); context conditioning = embed(t=0) (noise level 0)
        cond_tgt = self.time_embedding(t)
        cond_ctx = self.time_embedding(torch.zeros(B, device=yc.device, dtype=t.dtype))
        c = self.tok.cond(level_id, role_id, frame_id, cond_ctx, cond_tgt, action_code)

        for blk in self.blocks:
            x = blk(x, c)

        # split target tokens off the tail and unpatchify each level back to latents
        tgt = x[:, -(n_tgt_c + n_tgt_f):]
        c_tgt = c[:, -(n_tgt_c + n_tgt_f):]
        xc, xf = tgt[:, :n_tgt_c], tgt[:, n_tgt_c:]
        cc, cf = c_tgt[:, :n_tgt_c], c_tgt[:, n_tgt_c:]

        sc, scl = self.final_c_ada(cc).chunk(2, dim=-1)
        vc = self.head_c(modulate(self.norm_c(xc), sc, scl))       # (B,16,LAT)
        sf, sfl = self.final_f_ada(cf).chunk(2, dim=-1)
        vf = self.head_f(modulate(self.norm_f(xf), sf, sfl))       # (B,64,LAT)

        hw_c = self.tok.COARSE_HW
        hw_f = self.tok.FINE_HW
        vc = vc.transpose(1, 2).reshape(B, self.lat, hw_c, hw_c)
        vf = vf.transpose(1, 2).reshape(B, self.lat, hw_f, hw_f)
        return vc, vf


# ----------------------------------------------------------------------------------------
# EnergyTransformer (EBT-style scalar-energy transformer)
# ----------------------------------------------------------------------------------------
class EnergyTransformer(nn.Module):
    def __init__(self, lat=16, n_ctx=6, d=256, depth=6, heads=8, k=4, max_mcmc_steps=64):
        super().__init__()
        self.lat, self.n_ctx, self.d, self.k = lat, n_ctx, d, k
        self.tok = LatentTokenizer(lat, d, n_ctx, k)
        # per-MCMC-step embedding (EBT time_embed, ar_ebt_adaln.py:553) -- important for stability
        self.step_embedding = nn.Embedding(max_mcmc_steps, d)
        nn.init.normal_(self.step_embedding.weight, std=0.02)
        # NOTE zero_gate=False: an untrained energy net must have a non-degenerate landscape,
        # otherwise every energy is identically 0 and grad wrt candidate vanishes.
        self.blocks = nn.ModuleList(
            [DiTBlock(d, heads, zero_gate=False) for _ in range(depth)]
        )
        self.norm_final = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.energy_head = nn.Linear(d, 1, bias=False)  # per-token scalar energy (ar_ebt_adaln.py:506)

    def energy(self, ctx_c, ctx_f, yc, yf, step, actions):
        # ctx_*: clean context; yc,yf: CANDIDATE target latents (require grad); step:(B,) long; actions:(B,k)
        B = yc.shape[0]
        x, level_id, role_id, frame_id, n_tgt_c, n_tgt_f = _build_sequence(
            self.tok, ctx_c, ctx_f, yc, yf, self.n_ctx
        )
        action_code = self.tok.action_code(actions)
        # target conditioning = embed(step); context conditioning = embed(step=0)
        cond_tgt = self.step_embedding(step)
        cond_ctx = self.step_embedding(torch.zeros(B, dtype=torch.long, device=yc.device))
        c = self.tok.cond(level_id, role_id, frame_id, cond_ctx, cond_tgt, action_code)

        for blk in self.blocks:
            x = blk(x, c)

        # per-token scalar energy on ONLY the candidate (target) tokens, then sum -> E:(B,1)
        tgt = x[:, -(n_tgt_c + n_tgt_f):]
        c_tgt = c[:, -(n_tgt_c + n_tgt_f):]
        shift, scale = self.final_ada(c_tgt).chunk(2, dim=-1)
        tgt = modulate(self.norm_final(tgt), shift, scale)
        per_tok_e = self.energy_head(tgt)              # (B, n_tgt, 1)
        E = per_tok_e.sum(dim=1)                        # (B,1)
        return E


# ----------------------------------------------------------------------------------------
# __main__ smoke test
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    def run(device):
        print(f"\n===== SMOKE on {device} =====", flush=True)
        torch.manual_seed(0)
        B, n_ctx, k = 8, 6, 4
        dev = torch.device(device)

        ctx_c = torch.randn(B, n_ctx, LAT, 4, 4, device=dev)
        ctx_f = torch.randn(B, n_ctx, LAT, 8, 8, device=dev)
        actions = torch.randint(-1, 2, (B, k), device=dev)

        # ---- DiTDenoiser ----
        dit = DiTDenoiser(lat=LAT, n_ctx=n_ctx, k=k).to(dev)
        n_dit = sum(p.numel() for p in dit.parameters())
        print(f"DiTDenoiser params: {n_dit/1e6:.2f}M")
        yc = torch.randn(B, LAT, 4, 4, device=dev)
        yf = torch.randn(B, LAT, 8, 8, device=dev)
        t = torch.rand(B, device=dev)
        vc, vf = dit(yc, yf, ctx_c, ctx_f, t, actions)
        assert vc.shape == (B, LAT, 4, 4), vc.shape
        assert vf.shape == (B, LAT, 8, 8), vf.shape
        assert torch.isfinite(vc).all() and torch.isfinite(vf).all()
        print(f"  DiT out shapes OK: {tuple(vc.shape)}, {tuple(vf.shape)}; finite OK")

        # ---- EnergyTransformer ----
        ebt = EnergyTransformer(lat=LAT, n_ctx=n_ctx, k=k).to(dev)
        n_ebt = sum(p.numel() for p in ebt.parameters())
        print(f"EnergyTransformer params: {n_ebt/1e6:.2f}M")
        yc = torch.randn(B, LAT, 4, 4, device=dev).requires_grad_(True)
        yf = torch.randn(B, LAT, 8, 8, device=dev).requires_grad_(True)
        step = torch.zeros(B, dtype=torch.long, device=dev)
        # the forward that will be double-backprop'd must use the explicit-softmax attention path
        # (fused SDPA has no double-backward) -- same wrapper the real training path uses.
        with double_backward_attention():
            E = ebt.energy(ctx_c, ctx_f, yc, yf, step, actions)
        assert E.shape == (B, 1), E.shape
        assert torch.isfinite(E).all()
        print(f"  Energy shape OK: {tuple(E.shape)}; finite OK")

        # 1st-order grad wrt candidate, with create_graph=True (2nd-order enabled)
        g = torch.autograd.grad(E.sum(), [yc, yf], create_graph=True)
        assert torch.isfinite(g[0]).all() and torch.isfinite(g[1]).all()
        # SECOND backward THROUGH g -> proves create_graph/2nd-order works end-to-end
        gnorm = g[0].pow(2).sum() + g[1].pow(2).sum()
        gnorm.backward()
        # verify some model param received a 2nd-order gradient
        got_2nd = any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in ebt.parameters()
        )
        assert got_2nd, "no 2nd-order gradient reached model params"
        print("  2nd-order grad (create_graph) end-to-end OK")

        # descend 3 steps yc <- yc - 1e-2*g just to prove it RUNS (untrained; monotonicity not required)
        yc2 = torch.randn(B, LAT, 4, 4, device=dev)
        yf2 = torch.randn(B, LAT, 8, 8, device=dev)
        energies = []
        for s in range(3):
            yc2 = yc2.detach().requires_grad_(True)
            yf2 = yf2.detach().requires_grad_(True)
            st = torch.full((B,), s, dtype=torch.long, device=dev)
            Es = ebt.energy(ctx_c, ctx_f, yc2, yf2, st, actions)
            gg = torch.autograd.grad(Es.sum(), [yc2, yf2], create_graph=False)
            yc2 = yc2 - 1e-2 * gg[0]
            yf2 = yf2 - 1e-2 * gg[1]
            energies.append(float(Es.sum().detach()))
        assert all(math.isfinite(e) for e in energies)
        print(f"  3-step descent runs OK; E trace: {[round(e,3) for e in energies]}")
        return n_dit, n_ebt

    n_dit, n_ebt = run("cpu")
    if torch.cuda.is_available():
        run("cuda")
    else:
        print("\n(CUDA not available; ran CPU only)")

    print(f"\nFinal param counts: DiTDenoiser={n_dit/1e6:.2f}M  EnergyTransformer={n_ebt/1e6:.2f}M")
    print("SMOKE OK")
