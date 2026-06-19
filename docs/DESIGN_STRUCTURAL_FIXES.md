# Design notes — the two "structural fixes" are redesigns, not quick patches

> 2026-06-19. Written while executing the review's fix list. After implementing the motion-baseline
> gate, the harder SO(3) data, and the GMM head, a careful look shows the remaining two fixes
> (untrained-last-position, multi-rate temporal alignment) are **inherent to the fixed-window
> single-step factorization** — they cannot be patched without changing that factorization, so they
> are scheduled as deliberate design work rather than shipped as fragile hacks.

## Fix #4 — the rollout feeds back an UNTRAINED last position

**Why it isn't a one-liner.** The predictive loss supervises `pred[:, :-1]` (positions `0..T-2`
predict frames `1..T-1`, which exist in the clip). Position `T-1` predicts frame `T` — *one past the
window* — which has no target, so it's unsupervised and regresses to a low-norm mean (verified
magnitude 0.48 vs 1.09). Rollout (`free_run`) advances by exactly this position (`recon[:,:,-1]`).

This is **unavoidable for any fixed-length causal window**: the only position that predicts the
*future* frame (the one rollout needs) is always the last, and it's always one-past-the-window, so it
can never have an in-window target. Lengthening the training clip (T→13) just moves the wall — the
last position of the new window is still unsupervised. `T≡1 (mod 4)` also rules out T=10/14.

**Proper fixes (pick one, gate it):**
1. **Variable-length windows.** Train on T frames but let inference use a *shorter* window so the
   predicted frame comes from an interior (supervised) position. Requires the cascade to support
   variable T (rebuild `self_mask`, the per-frame RoPE grid, and the repeated spatial bias per length,
   or make them length-agnostic). Cleanest long-term; ~1 day.
2. **Anti-drift / self-forcing training (roadmap P2).** Train *with* a 1+-step rollout: feed back the
   model's own last-position prediction, re-encode, and add a consistency loss so that position is
   shaped by how it's actually used. This both supervises it and closes the exposure-bias gap — so #4
   is best absorbed into P2 rather than fixed in isolation.

**Recommendation:** fold #4 into P2 (anti-drift). It is the same defect (train-what-you-test) and a
standalone patch would be thrown away when P2 lands.

## Fix #5 — multi-rate temporal shift + off-distribution decode

**Why it isn't a one-liner.** Each stage predicts "+1 at its native rate": ~1 real frame at BOT (T9),
~2 at MID (T5), ~4 at TOP (T3). `decode_depth` fuses the predicted stack by *grid position*, so the
coarse stages contribute structure for a frame further in the future than BOT (probe: depth-1 lands
~frame 11.7, depth-3 ~10.1, true next = 9). And the decoder was only ever trained on *real* (aligned)
stacks, so decoding any predicted stack is off-distribution.

This is **inherent to temporal downsampling**: frame `t+1` (a single real frame) has no native MID/TOP
latent when those stages only exist every 2/4 frames — so "predict frame t+1's latent decomposition"
is not even well-defined at the coarse stages. You cannot align the per-stage "+1" to a common real
frame without either interpolating coarse latents or changing what the coarse stages predict.

**Proper fixes (pick one, gate it):**
1. **BOT-only next-frame prediction; coarse stages as current-time conditioners.** Only BOT predicts
   the next frame; MID/TOP supply *current* (un-shifted) context via cross-attention. Decode then fuses
   BOT(next) with MID/TOP(current) — still a mild rate mismatch, but no coarse over-shoot and the
   coarse stages are used in-distribution. The `+0.000/+0.03` cascade lifts already hint the coarse
   stages act mostly as conditioners, so this may lose little. Medium change to the cascade contract.
2. **Decoder fine-tune on predicted stacks.** Add a term that trains the decoder to render the
   *predicted* (shifted) stack against the true next frame, removing the OOD-decode cost. Additive, but
   the fused shifted stack has no single clean pixel target (BOT→t+1 vs coarse→t+2..t+4), so the target
   definition is itself ambiguous — weaker than (1).
3. **Common-real-time prediction with coarse-latent interpolation.** Predict every stage's latent for
   the *same* future frame, interpolating the coarse latent in time. Most faithful to "hierarchy", most
   work, and reintroduces a temporal-interpolation approximation.

**Recommendation:** prototype (1) (BOT-predicts, coarse-conditions) behind a `predict_mode` flag and
A/B it against the current symmetric-shift cascade on the SO(3) data — it is the cleanest and directly
testable. Gate: `decode(z_pred)` best-matches GT frame `t+1` (not `t+1.7`), and changed-region rollout
improves.

## Status of the review's fix list

| # | item | status |
|---|---|---|
| gates | motion baselines + grad guard + doc honesty | ✅ shipped + tested + committed |
| data | harder SO(3) dataset (defeats the 2D-warp null: 25→9 dB) | ✅ shipped + gated + committed |
| #3 | GMM distributional head, wired end-to-end + sampled rollout | ✅ shipped + tested + committed |
| #6 | re-run capacity/topology/windowing + L2-vs-GMM on SO(3) | ◻ in progress (needs the SO(3) runs) |
| #4 | untrained last position | ▣ designed — fold into P2 (anti-drift), not a standalone patch |
| #5 | multi-rate temporal alignment | ▣ designed — prototype BOT-predicts/coarse-conditions behind a flag |
