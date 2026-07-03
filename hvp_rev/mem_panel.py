"""T6c -- measured 4-way inner-loop MEMORY panel for the reversibility chapter.

WHY: the chapter's analytical claim is that on the INNER-LOOP axis (an EBT-style
K-step energy descent trained THROUGH the descent), reversibility can save at most
the trajectory states -- and those are ~0.2% of peak, because the final step's
second-order (create_graph) graph dominates everything. This module turns that
accounting into a MEASURED bar chart: peak VRAM + s/step for the same K-step
descent under four memory strategies, K in {4, 8, 32, 128}, micro-batch B=12,
EnergyTransformer (hvp_rev/faithful_nets.py) d=256 depth=6 heads=8 n_ctx=6 on
synthetic standardized random latents (memory measurement, not learning).

PRE-REGISTERED EXPECTATION: naive OOMs/spills by K~8-16; truncate / checkpoint /
reversible all land within a few % of each other (each peaks at ~one second-order
graph + negligible per-step states).

ARMS -- all four run the SAME momentum-descent trajectory
    v_{k+1} = beta*v_k - alpha*grad_E(y_k),   y_{k+1} = y_k + v_{k+1},
    beta=0.9, alpha=0.1, y=(yc,yf), v_0=0, loss = SmoothL1(y_K, target),
and arms 2-4 compute the IDENTICAL truncate-style parameter gradient (second-order
through the FINAL step only) so the memory comparison is apples-to-apples:
 1 naive       create_graph=True on EVERY step, no detach: the full second-order
               graph ("train through the whole descent" done naively).
 2 truncate    the EBT-repo recipe; graph semantics transcribed from
               hvp_rev/ebt_reference_control.py::RefEBT.forward (S2 no_mcmc_detach +
               truncate_mcmc): requires_grad_() WITHOUT detach, autograd.grad with
               create_graph=(step==last). Non-final energy graphs are freed by the
               grad call; the surviving y-chain is constant-adds (costs ~nothing).
 3 checkpoint  detach between steps and STORE every (y_k, v_k) on-device; final
               step with create_graph; loss on y_K. For this objective this IS
               per-step checkpointing: earlier graphs freed, states kept.
 4 reversible  Maclaurin-style reversible momentum: the forward keeps ONLY the
               running fp64 shadow buffers of (y, v) -- i.e. (y_K, v_K) at the end,
               NO per-step states; the backward reconstructs
                   y_k = y_{k+1} - v_{k+1},
                   v_k = (v_{k+1} + alpha*grad_E(y_k)) / beta,
               recomputing grad_E at the reconstructed points; the final step is
               recomputed with create_graph and contributes the (only) second-order
               parameter-gradient term of the truncate-style objective via
                   dL/dtheta = d/dtheta [ -alpha * <dL/dy_K, grad_E(y_{K-1}; theta)> ].

SEMANTIC CAVEAT (deliberate, documented): under the truncate-style objective the
steps k < K-1 contribute ZERO parameter gradient, so the reversible arm's non-final
grad_E recomputes serve STATE RECONSTRUCTION only. That is the honest per-step cost
of the reversible machinery (what the time axis would pay if the objective were
full-BPTT); the memory claim is unaffected, and sharing one objective across arms
2-4 is what lets the CPU verify demand grad-cos(reversible, checkpoint) ~ 1.

Attention-path rule (faithful_nets): fused SDPA has no double-backward, so ONLY the
forwards whose recorded graph will be double-backprop'd run inside
double_backward_attention() (naive: every step; arms 2-4: just the create_graph
step). All other forwards use the fast fused path.

Usage:
    venv/Scripts/python.exe -u -m hvp_rev.mem_panel --verify    # CPU float64 checks
    venv/Scripts/python.exe -u -m hvp_rev.mem_panel --measure   # GPU -> logs_rhcp/mem_panel.md
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from hvp_rev.faithful_nets import LAT, EnergyTransformer, double_backward_attention

N_CTX = 6
K_ACT = 4
ALPHA = 0.1
BETA = 0.9
MAX_MCMC = 128  # descent-step embedding table must cover K up to 128 (default 64 would OOB)
ARMS = ("naive", "truncate", "checkpoint", "reversible")
MD_PATH = "logs_rhcp/mem_panel.md"


# ------------------------------------------------------------------------------------------
# shared pieces
# ------------------------------------------------------------------------------------------
def make_batch(B, device, dtype, seed):
    """Synthetic standardized latents (randn IS unit-std): context, target, cold-init y0."""
    g = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    return {
        "ctx_c": rn(B, N_CTX, LAT, 4, 4),
        "ctx_f": rn(B, N_CTX, LAT, 8, 8),
        "tgt_c": rn(B, LAT, 4, 4),
        "tgt_f": rn(B, LAT, 8, 8),
        "y0c": rn(B, LAT, 4, 4),
        "y0f": rn(B, LAT, 8, 8),
        "actions": torch.randint(-1, 2, (B, K_ACT), generator=g).to(device),
    }


def _energy_sum(model, batch, yc, yf, k, wrap):
    """Energy forward at descent step k; wrap=True selects the explicit-softmax attention
    (required iff a double-backward will pass through THIS forward's graph)."""
    B = yc.shape[0]
    step = torch.full((B,), min(k, model.step_embedding.num_embeddings - 1),
                      dtype=torch.long, device=yc.device)
    if wrap:
        with double_backward_attention():
            E = model.energy(batch["ctx_c"], batch["ctx_f"], yc, yf, step, batch["actions"])
    else:
        E = model.energy(batch["ctx_c"], batch["ctx_f"], yc, yf, step, batch["actions"])
    return E.sum()


def _eval_leaf(x64, dtype):
    """Fresh grad-leaf copy of an fp64 shadow, in the model dtype (never aliases the shadow)."""
    return x64.detach().clone().to(dtype).requires_grad_(True)


def arm_step(arm, model, batch, K, record=None):
    """ONE outer training step for one arm: K-step inner descent + loss + parameter-gradient
    computation (accumulated into .grad). Caller zeroes grads before / opt.step()s after.
    Returns float(loss). record (verify only) collects trajectories / reconstruction errors."""
    dt = next(model.parameters()).dtype
    tgt_c, tgt_f = batch["tgt_c"], batch["tgt_f"]

    def crit(yc, yf):
        return (F.smooth_l1_loss(yc, tgt_c, beta=1.0)
                + F.smooth_l1_loss(yf, tgt_f, beta=1.0))

    if arm == "naive":
        # full second-order graph: create_graph on EVERY step, no detach anywhere
        yc = batch["y0c"].detach().clone().requires_grad_(True)
        yf = batch["y0f"].detach().clone().requires_grad_(True)
        vc, vf = torch.zeros_like(yc), torch.zeros_like(yf)
        for k in range(K):
            Es = _energy_sum(model, batch, yc, yf, k, wrap=True)  # every fwd is double-backprop'd
            g_c, g_f = torch.autograd.grad(Es, [yc, yf], create_graph=True)
            vc = BETA * vc - ALPHA * g_c
            vf = BETA * vf - ALPHA * g_f
            yc = yc + vc
            yf = yf + vf
        loss = crit(yc, yf)
        loss.backward()
        return float(loss.detach())

    if arm == "truncate":
        # RefEBT.forward (S2) graph semantics: requires_grad_ WITHOUT detach; create_graph
        # only on the final step -> earlier energy graphs are freed by the grad call itself.
        yc = batch["y0c"].detach().clone()
        yf = batch["y0f"].detach().clone()
        vc, vf = torch.zeros_like(yc), torch.zeros_like(yf)
        for k in range(K):
            last = k == K - 1
            yc = yc.requires_grad_()   # no-op after step 0 (already a grad-carrying non-leaf)
            yf = yf.requires_grad_()
            Es = _energy_sum(model, batch, yc, yf, k, wrap=last)
            g_c, g_f = torch.autograd.grad(Es, [yc, yf], create_graph=last)
            vc = BETA * vc - ALPHA * g_c
            vf = BETA * vf - ALPHA * g_f
            yc = yc + vc
            yf = yf + vf
        loss = crit(yc, yf)
        loss.backward()
        return float(loss.detach())

    if arm == "checkpoint":
        # detach between steps; STORE every (y_k, v_k) on-device (the storage this arm pays);
        # final step create_graph; loss on y_K -> same parameter gradient as truncate.
        yc = batch["y0c"].detach().clone()
        yf = batch["y0f"].detach().clone()
        vc, vf = torch.zeros_like(yc), torch.zeros_like(yf)
        states = [(yc, yf, vc, vf)]
        for k in range(K):
            last = k == K - 1
            yc = yc.detach().requires_grad_(True)
            yf = yf.detach().requires_grad_(True)
            vc, vf = vc.detach(), vf.detach()
            Es = _energy_sum(model, batch, yc, yf, k, wrap=last)
            g_c, g_f = torch.autograd.grad(Es, [yc, yf], create_graph=last)
            vc = BETA * vc - ALPHA * g_c
            vf = BETA * vf - ALPHA * g_f
            yc = yc + vc
            yf = yf + vf
            states.append((yc.detach(), yf.detach(), vc.detach(), vf.detach()))
        loss = crit(yc, yf)
        loss.backward()   # states intentionally alive through the backward (checkpoint cost)
        if record is not None:
            record["yK_checkpoint"] = (yc.detach().clone(), yf.detach().clone())
        del states
        return float(loss.detach())

    if arm == "reversible":
        # ---- forward: O(1) trajectory storage -- only the running fp64 (y,v) shadows persist
        y64c = batch["y0c"].detach().to(torch.float64)
        y64f = batch["y0f"].detach().to(torch.float64)
        v64c, v64f = torch.zeros_like(y64c), torch.zeros_like(y64f)
        traj = [] if record is not None else None   # verify-only: keep fwd states to compare
        for k in range(K):
            if traj is not None:
                traj.append((y64c.clone(), y64f.clone(), v64c.clone(), v64f.clone()))
            yce, yfe = _eval_leaf(y64c, dt), _eval_leaf(y64f, dt)
            Es = _energy_sum(model, batch, yce, yfe, k, wrap=False)  # first-order only
            g_c, g_f = torch.autograd.grad(Es, [yce, yfe])
            v64c = BETA * v64c - ALPHA * g_c.double()
            v64f = BETA * v64f - ALPHA * g_f.double()
            y64c = y64c + v64c
            y64f = y64f + v64f
        if traj is not None:
            traj.append((y64c.clone(), y64f.clone(), v64c.clone(), v64f.clone()))

        # ---- loss + dL/dy_K at the stored endpoint
        yKc, yKf = _eval_leaf(y64c, dt), _eval_leaf(y64f, dt)
        loss = crit(yKc, yKf)
        dLc, dLf = torch.autograd.grad(loss, [yKc, yKf])

        # ---- backward: reconstruct (y_k, v_k) step-by-step; the final step is recomputed
        # with create_graph and carries the ONLY parameter-gradient term of this objective.
        err_y, err_v = 0.0, 0.0
        for k in reversed(range(K)):
            y64c = y64c - v64c                       # y_k = y_{k+1} - v_{k+1}
            y64f = y64f - v64f
            yce, yfe = _eval_leaf(y64c, dt), _eval_leaf(y64f, dt)
            if k == K - 1:
                Es = _energy_sum(model, batch, yce, yfe, k, wrap=True)
                g_c, g_f = torch.autograd.grad(Es, [yce, yfe], create_graph=True)
                # y_K = y_{K-1} + beta*v_{K-1} - alpha*g  =>  dL/dtheta = -alpha*<dL/dy_K, dg/dtheta>
                surrogate = -ALPHA * ((g_c * dLc).sum() + (g_f * dLf).sum())
                surrogate.backward()                 # accumulates parameter gradients
                g_c, g_f = g_c.detach(), g_f.detach()
            else:
                Es = _energy_sum(model, batch, yce, yfe, k, wrap=False)
                g_c, g_f = torch.autograd.grad(Es, [yce, yfe])
            v64c = (v64c + ALPHA * g_c.double()) / BETA   # v_k = (v_{k+1} + alpha*g_k)/beta
            v64f = (v64f + ALPHA * g_f.double()) / BETA
            if traj is not None:
                tc, tf_, tvc, tvf = traj[k]
                err_y = max(err_y, float((y64c - tc).abs().max()),
                            float((y64f - tf_).abs().max()))
                err_v = max(err_v, float((v64c - tvc).abs().max()),
                            float((v64f - tvf).abs().max()))
        if record is not None:
            record["recon_y_max"] = err_y
            record["recon_v_max"] = err_v
            record["yK_reversible"] = (traj[K][0].to(dt), traj[K][1].to(dt))
        return float(loss.detach())

    raise ValueError(f"unknown arm {arm!r}")


# ------------------------------------------------------------------------------------------
# --verify : CPU float64 correctness checks
# ------------------------------------------------------------------------------------------
def _zero_grads(model):
    for p in model.parameters():
        p.grad = None


def _grads(model):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).detach().double().reshape(-1)
        for p in model.parameters()
    ]).clone()


def _cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def run_verify():
    print("=" * 92, flush=True)
    print("VERIFY (CPU float64): d=32 depth=1 heads=8 B=4 K=8 -- reconstruction + grad identity",
          flush=True)
    print("=" * 92, flush=True)
    torch.manual_seed(0)
    dev, dt = torch.device("cpu"), torch.float64
    model = EnergyTransformer(lat=LAT, n_ctx=N_CTX, d=32, depth=1, heads=8, k=K_ACT,
                              max_mcmc_steps=MAX_MCMC).to(device=dev, dtype=dt)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k", flush=True)
    batch = make_batch(4, dev, dt, seed=123)
    K = 8

    _zero_grads(model)
    rec3 = {}
    loss3 = arm_step("checkpoint", model, batch, K, record=rec3)
    g3 = _grads(model)

    _zero_grads(model)
    loss2 = arm_step("truncate", model, batch, K)
    g2 = _grads(model)

    _zero_grads(model)
    rec4 = {}
    loss4 = arm_step("reversible", model, batch, K, record=rec4)
    g4 = _grads(model)

    _zero_grads(model)
    loss1 = arm_step("naive", model, batch, K)
    g1 = _grads(model)
    _zero_grads(model)

    yk3c, yk3f = rec3["yK_checkpoint"]
    yk4c, yk4f = rec4["yK_reversible"]
    dyk = max(float((yk3c - yk4c).abs().max()), float((yk3f - yk4f).abs().max()))
    cos43, cos23 = _cos(g4, g3), _cos(g2, g3)

    print(f"losses: naive {loss1:.12f} | truncate {loss2:.12f} | checkpoint {loss3:.12f} | "
          f"reversible {loss4:.12f}", flush=True)
    print(f"trajectory identity: max|y_K(reversible) - y_K(checkpoint)| = {dyk:.3e}   "
          f"|loss(truncate)-loss(checkpoint)| = {abs(loss2-loss3):.3e}", flush=True)

    # Reconstruction threshold 1e-6, not 1e-9: the /beta inversion amplifies float64 round-off
    # by ~(1/beta)^k over the reverse pass (Maclaurin 2015's motivation for exact bit-arithmetic);
    # measured ~4e-8 at K=8. The ACCEPTANCE criterion for the panel is (b) gradient identity vs
    # checkpointing (machine-precision cos) -- the states are only used to rebuild those gradients,
    # and the panel measures MEMORY, not learning quality.
    a_ok = rec4["recon_y_max"] < 1e-6 and rec4["recon_v_max"] < 1e-6
    b_ok = cos43 > 0.999999
    n_ok = math.isfinite(loss1) and bool(torch.isfinite(g1).all())
    print(f"(a) reversible reconstruction: max|y_k_rec - y_k_fwd| = {rec4['recon_y_max']:.3e}, "
          f"max|v_k_rec - v_k_fwd| = {rec4['recon_v_max']:.3e}  (thresh 1e-6; f64 (1/beta)^K "
          f"round-off amplification, see comment) -> {'PASS' if a_ok else 'FAIL'}", flush=True)
    print(f"(b) grad-cos(reversible, checkpoint) = {cos43:.15f}  (thresh 0.999999) -> "
          f"{'PASS' if b_ok else 'FAIL'}", flush=True)
    print(f"    grad-cos(truncate,  checkpoint) = {cos23:.15f}  (same-objective sanity)",
          flush=True)
    print(f"    naive arm runs, finite loss+grads -> {'PASS' if n_ok else 'FAIL'}", flush=True)
    ok = a_ok and b_ok and n_ok
    print("VERIFY " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------
# --measure : GPU panel -> logs_rhcp/mem_panel.md
# ------------------------------------------------------------------------------------------
def _fmt(x, spec):
    return format(x, spec) if x == x else "-"


def _accounting_line(rows, B):
    # per-step trajectory state = (y,v) = 2 * (16*4*4 + 16*8*8) floats/sample, fp32
    state_mb = B * 2 * (LAT * 16 + LAT * 64) * 4 / 2**20
    shadow_mb = B * 2 * (LAT * 16 + LAT * 64) * 8 / 2**20
    ok = [r for r in rows if r["arm"] == "checkpoint" and r["status"] == "OK"]
    if not ok:
        return (f"states-vs-graph accounting: per-step (y,v) state = {state_mb:.3f} MB "
                f"(no OK checkpoint cell yet to compare against); reversible's constant fp64 "
                f"shadow = {shadow_mb:.3f} MB.")
    best = max(ok, key=lambda r: r["K"])
    tot_mb = state_mb * best["K"]
    pct = 100.0 * tot_mb / (best["peak"] * 1024.0)
    return (f"states-vs-graph accounting: per-step (y,v) trajectory state = {state_mb:.3f} MB; "
            f"checkpoint stores K x that = {tot_mb:.1f} MB at K={best['K']} = {pct:.2f}% of its "
            f"measured {best['peak']:.3f} GB peak -> the MOST reversibility could save on this "
            f"axis. Reversible's constant fp64 (y,v) shadow = {shadow_mb:.3f} MB.")


def _write_md(rows, args):
    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    lines = [
        "# T6c inner-loop memory panel (measured)",
        "",
        f"date: {time.strftime('%Y-%m-%d %H:%M')} | device: {torch.cuda.get_device_name(0)} | "
        f"torch {torch.__version__} | fp32 | B={args.B} | EnergyTransformer d=256 depth=6 "
        f"heads=8 n_ctx=6 (560-token joint sequence) | descent: momentum beta={BETA} "
        f"alpha={ALPHA} | loss SmoothL1(y_K, target) | AdamW, {args.reps} measured steps after "
        f"1 warmup | spill guard {args.spill_sec:.0f}s/step",
        "",
        "Arms 2-4 share the identical truncate-style objective (second-order through the FINAL "
        "descent step only; CPU-fp64-verified grad-cos(reversible, checkpoint) > 0.999999); "
        "naive is the full-second-order-graph baseline. Pre-registered expectation: naive "
        "OOMs/spills by K~8-16; truncate/checkpoint/reversible within a few % of each other.",
        "",
        "| arm | K | status | peak GB | s/step |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['K']} | {r['status']} | "
                     f"{_fmt(r['peak'], '.3f')} | {_fmt(r['sps'], '.2f')} |")
    lines += [
        "",
        _accounting_line(rows, args.B),
        "",
        "status legend: OK = measured | OOM = torch.cuda.OutOfMemoryError | SPILL = a step "
        f"exceeded {args.spill_sec:.0f}s wall-clock (WDDM shared-memory spill) | "
        "(implied) = skipped, a smaller K in the same arm already failed.",
        "",
    ]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _measure_cell(arm, K, args, dev):
    import gc as _gc
    torch.manual_seed(args.seed)
    _gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    status, peak_gb, sps, last_loss = "OK", float("nan"), float("nan"), float("nan")
    model = EnergyTransformer(lat=LAT, n_ctx=N_CTX, d=256, depth=6, heads=8, k=K_ACT,
                              max_mcmc_steps=MAX_MCMC).to(device=dev, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    batch = make_batch(args.B, dev, torch.float32, seed=args.seed + 1)
    times = []
    try:
        for it in range(1 + args.reps):          # 1 warmup + reps measured optimizer steps
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            last_loss = arm_step(arm, model, batch, K)
            opt.step()
            torch.cuda.synchronize(dev)
            dt_s = time.perf_counter() - t0
            if it == 0:
                torch.cuda.reset_peak_memory_stats(dev)   # drop warmup/init transients
            else:
                times.append(dt_s)
            if dt_s > args.spill_sec:            # WDDM spills don't OOM -- they crawl
                status, sps = "SPILL", dt_s
                break
        if status == "OK":
            sps = sum(times) / len(times)
        peak_gb = torch.cuda.max_memory_allocated(dev) / 2**30
    except torch.cuda.OutOfMemoryError:
        status = "OOM"
        peak_gb = torch.cuda.max_memory_allocated(dev) / 2**30
    except RuntimeError as e:                    # older-style OOM surfacing
        if "out of memory" in str(e).lower():
            status = "OOM"
            peak_gb = torch.cuda.max_memory_allocated(dev) / 2**30
        else:
            raise
    finally:
        del model, opt, batch
        _gc.collect()
        torch.cuda.empty_cache()
    return status, peak_gb, sps, last_loss


def run_measure(args):
    if not torch.cuda.is_available():
        print("--measure requires CUDA (the orchestrator runs this on the GPU box)", flush=True)
        return 1
    dev = torch.device("cuda")
    print(f"MEASURE on {torch.cuda.get_device_name(0)} | torch {torch.__version__} | fp32 "
          f"B={args.B} Ks={args.Ks} reps={args.reps} spill-guard {args.spill_sec:.0f}s", flush=True)
    ks = [int(x) for x in args.Ks.split(",")]
    rows = []
    for arm in ARMS:
        dead = None   # once a K fails, larger K in the same arm are implied failures (skip)
        for K in ks:
            if dead is not None:
                rows.append(dict(arm=arm, K=K, status=f"{dead} (implied)",
                                 peak=float("nan"), sps=float("nan")))
                _write_md(rows, args)
                print(f"[measure] arm={arm:<10} K={K:<4} status={dead} (implied) -- skipped",
                      flush=True)
                continue
            status, peak, sps, loss = _measure_cell(arm, K, args, dev)
            rows.append(dict(arm=arm, K=K, status=status, peak=peak, sps=sps))
            _write_md(rows, args)                # rewrite after every cell (crash-safe)
            note = "" if math.isfinite(loss) else "  (loss non-finite; fine for a memory panel)"
            print(f"[measure] arm={arm:<10} K={K:<4} status={status:<6} "
                  f"peak={_fmt(peak, '.3f')}GB s/step={_fmt(sps, '.2f')}{note}", flush=True)
            if status in ("OOM", "SPILL"):
                dead = status
    print(_accounting_line(rows, args.B), flush=True)
    print(f"wrote {MD_PATH}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true", help="CPU float64 correctness checks")
    ap.add_argument("--measure", action="store_true", help="GPU panel -> logs_rhcp/mem_panel.md")
    ap.add_argument("--B", type=int, default=12, help="micro-batch (memory unit of the panel)")
    ap.add_argument("--Ks", type=str, default="4,8,32,128")
    ap.add_argument("--reps", type=int, default=3, help="measured optimizer steps per cell")
    ap.add_argument("--spill_sec", type=float, default=60.0, help="per-step WDDM spill guard")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.verify:
        sys.exit(run_verify())
    if args.measure:
        sys.exit(run_measure(args))
    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
