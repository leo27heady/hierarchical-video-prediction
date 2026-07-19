"""OpenSTL Moving-MNIST bridge -- the appendix CREDIBILITY ANCHOR: our WindowDiT harness on the
community-standard 10->10 Moving-MNIST protocol, scored with the exact OpenSTL metric convention.

================================================================================================
PROTOCOL (standard OpenSTL / PredRNN Moving-MNIST benchmark, 10 context -> 10 predicted)
================================================================================================
* TEST  = the canonical `mnist_test_seq.npy` (20, 10000, 64, 64) uint8 from the sibling OpenSTL
  repo (path in hvp_rev.eval_adapters.OPENSTL_TEST). Frames 0-9 = context, 10-19 = target.
  NEVER trained on; never used for model selection.
* TRAIN = on-the-fly-style Moving MNIST frozen into ONE fixed file with a RECORDED seed
  (GEN_SEED = 0): data/mmnist_bridge_train_10000x20.npy -- two 28x28 digits from OpenSTL's
  train-images-idx3-ubyte.gz, standard bounce dynamics (step_length 0.1, uniform start position
  and heading). Exact port of OpenSTL openstl/datasets/dataloader_moving_mnist.py
  (get_random_trajectory + generate_moving_mnist), with python `random` swapped for a seeded
  numpy RandomState so the file is reproducible. A zero-filled *_actions.npy is written
  alongside: MMNIST has NO actions; WindowDiT's action embedding takes {-1,0,+1}, so every
  frame gets the constant "no-op" action 0.
* METRICS = hvp_rev.eval_adapters (bit-verified vs openstl.core.metrics): pixel-SUM MSE / MAE,
  per-frame-mean SSIM / PSNR on [0,1]; plus full-frame LPIPS (no fg mask) when `lpips` is
  importable (alexnet backbone; cached weights required offline). Every eval table carries the
  AE-ROUND-TRIP CEILING row (encode+decode the GT target frames) -- the honesty device: a
  latent-space predictor cannot beat its own frozen AE -- and a persistence row (repeat frame 9).
* MODEL = the UNMODIFIED WindowDiT / WindowDiffusion from hvp_rev.rhvd (4-frame windows,
  v-pred cosine diffusion, DDIM-16 at eval, EMA 0.999). 10 predicted frames = 3 window steps
  (4+4+2: the final window predicts 4 frames and the last 2 are cropped). Bridge arms are
  registered into rhvd.ARMS at import time -- rhvd.py itself is NOT edited:
    mm-flat : n_ctx_c=0,  n_ctx_f=10  (640 ctx tokens)  -- flat baseline row
    mm-hier : n_ctx_c=10, n_ctx_f=6   (544 ctx tokens)  -- hierarchical comparison row
  The AE is a fresh HierConvAE trained on bridge train frames only -> logs_rhcp/hier_ae_mmnist.pt.

================================================================================================
LEADERBOARD REFERENCE NUMBERS (for the appendix table)
================================================================================================
VERIFIED locally in the sibling repo: OpenSTL docs/en/model_zoos/video_benchmarks.md
(10->10 MMNIST, OpenSTL's own unified retraining, Adam + OneCycle, single GPU):
    200  epochs: ConvLSTM-S    29.80 MSE / 90.64 MAE / 0.9288 SSIM / 22.10 PSNR
                 PredRNN       23.97 / 72.82 / 0.9462 / 23.28
                 PredRNN++     22.06 / 69.58 / 0.9509 / 23.65
                 SimVP+gSTA-S  26.69 / 77.19 / 0.9402  (PSNR inconsistent in the doc itself:
                                38.35 in the main table vs 22.78 in the MetaVP table -- cite MSE/
                                MAE/SSIM only, or resolve against the released log file)
    2000 epochs: ConvLSTM-S    22.41 / 73.07 / 0.9480
                 PredRNN++     14.07 / 48.91 / 0.9698
                 SimVP+gSTA-S  15.05 / 49.80 / 0.9675
Classic paper-reported numbers often quoted (ConvLSTM ~103.3, PredRNN ~56.8, SimVP ~23.8 MSE)
are from the SimVP (CVPR'22) paper's table under the ORIGINAL training recipes -- they are NOT
in the local OpenSTL docs [VERIFY against the SimVP paper before citing; prefer the OpenSTL
numbers above, which we verified locally and whose convention our metrics are bit-matched to].

================================================================================================
RUN COMMANDS (from the repo root; venv python)
================================================================================================
  # 0) smoke chain (GPU, ~4 min: gen 400 seqs, AE 200 steps, both arms 200 steps, eval 32 seqs)
  venv/Scripts/python.exe -u -m hvp_rev.mmnist_bridge --stage smoke

  # 1) generate the fixed train set (10k x 20 frames, seed 0) + zero-actions file
  venv/Scripts/python.exe -u -m hvp_rev.mmnist_bridge --stage gen
  # 2) train the frozen hier AE on bridge train frames (7k steps)
  venv/Scripts/python.exe -u -m hvp_rev.mmnist_bridge --stage ae --ae_steps 7000
  # 3) train the arms (B16 x 15k steps each; checkpoint/resume-safe via rhvd.train)
  venv/Scripts/python.exe -u -m hvp_rev.mmnist_bridge --stage train --arm mm-flat --steps 15000 --bf16
  venv/Scripts/python.exe -u -m hvp_rev.mmnist_bridge --stage train --arm mm-hier --steps 15000 --bf16
  # 4) full 10->10 eval on the canonical 10000-sequence test set (per arm)
  venv/Scripts/python.exe -u -m hvp_rev.mmnist_bridge --stage eval --arm mm-flat --steps 15000
  venv/Scripts/python.exe -u -m hvp_rev.mmnist_bridge --stage eval --arm mm-hier --steps 15000
"""
from __future__ import annotations

import argparse
import gzip
import os
import time

import numpy as np
import torch

import hvp_rev.eval_adapters as EA
import hvp_rev.rhvd as R
import hvp_rev.step1_mech as M
from hvp_rev import hier_convae, viz
from hvp_rev.faithful_nets import LAT
from hvp_rev.step1_mech import ActionHierData, decode01

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GEN_SEED = 0
MNIST_GZ = os.environ.get(
    "HVP_MNIST_GZ", "data/external/openstl/moving_mnist/train-images-idx3-ubyte.gz"
)
AE_PATH = "logs_rhcp/hier_ae_mmnist.pt"

# Bridge arms, registered into rhvd.ARMS at import (rhvd.py untouched; keys are new).
BRIDGE_ARMS = {
    # name: (n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f) -- same 5-tuple as rhvd.ARMS
    "mm-flat": (0, 10, "diff", 1, 0),
    "mm-hier": (10, 6, "diff", 1, 0),
}
R.ARMS.update(BRIDGE_ARMS)


def log(*a):
    print(*a, flush=True)


def train_path(n_train):
    return f"data/mmnist_bridge_train_{n_train}x20.npy"


def stats_path(ae_path):
    return ae_path + ".bridge_stats.pt"


# ================================================================================================
# 1) DATA: fixed-seed port of OpenSTL's on-the-fly Moving-MNIST generator.
# ================================================================================================
def _load_mnist_digits():
    assert os.path.exists(MNIST_GZ), f"missing MNIST train images: {MNIST_GZ}"
    with gzip.open(MNIST_GZ, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)


def _trajectory(rng, T, image_size=64, digit_size=28, step_length=0.1):
    """Port of OpenSTL MovingMNIST.get_random_trajectory (dataloader_moving_mnist.py:103-145):
    uniform start (x,y) in [0,1]^2, uniform heading theta, constant speed step_length, reflecting
    walls via the bounce_x/bounce_y sign trick, position updated BEFORE the first store."""
    canvas = image_size - digit_size
    x, y = rng.random_sample(), rng.random_sample()
    theta = rng.random_sample() * 2 * np.pi
    v_y, v_x = np.sin(theta), np.cos(theta)
    sy, sx = np.zeros(T), np.zeros(T)
    bx, by = 1, 1
    for i in range(T):
        y += by * v_y * step_length
        x += bx * v_x * step_length
        if x <= 0:
            x = 0
            bx = -bx
        if x >= 1.0:
            x = 1.0
            bx = -bx
        if y <= 0:
            y = 0
            by = -by
        if y >= 1.0:
            y = 1.0
            by = -by
        sy[i], sx[i] = y, x
    return (canvas * sy).astype(np.int32), (canvas * sx).astype(np.int32)


def gen_clips(n_seq, T=20, seed=GEN_SEED, num_digits=2):
    """(n_seq, T, 64, 64) uint8 -- port of generate_moving_mnist (num_digits=2, black bg),
    np.maximum digit compositing, one seeded RandomState for the WHOLE file (recorded seed)."""
    rng = np.random.RandomState(seed)
    digits = _load_mnist_digits()
    clips = np.zeros((n_seq, T, 64, 64), np.uint8)
    for s in range(n_seq):
        for _ in range(num_digits):
            ty, tx = _trajectory(rng, T)
            img = digits[rng.randint(digits.shape[0])]
            for i in range(T):
                reg = clips[s, i, ty[i]:ty[i] + 28, tx[i]:tx[i] + 28]
                np.maximum(reg, img, out=reg)
    return clips


def run_gen(n_train, seed=GEN_SEED):
    path = train_path(n_train)
    if os.path.exists(path):
        log(f"[gen] {path} already exists -- skipping (delete it to regenerate)")
        return path
    t0 = time.time()
    clips = gen_clips(n_train, T=20, seed=seed)
    os.makedirs("data", exist_ok=True)
    np.save(path, clips)
    acts = np.zeros(clips.shape[:2], np.int8)          # MMNIST has no actions -> constant no-op 0
    np.save(path.replace(".npy", "_actions.npy"), acts)
    log(f"[gen] wrote {path} {clips.shape} uint8 (seed {seed}, OpenSTL dynamics) "
        f"+ zero actions file in {time.time()-t0:.1f}s; pixel mean {clips.mean():.2f}")
    return path


# ================================================================================================
# 2) Data layer: reuse ActionHierData unchanged; chunk only the encode (N=10k frames/timestep).
# ================================================================================================
class _ChunkedEnc:
    """VRAM guard: ActionHierData encodes ALL clips' frames for one timestep in a single conv
    pass; at N=10000 (vs 3000 on loopmom) that transient is ~3-5GB, so route .encode through
    fixed-size chunks. Everything else (decode_fine, attribute writes like _lat_stats) proxies
    to the wrapped AE, and build_data() re-sets _lat_stats on the REAL AE afterwards."""

    def __init__(self, ae, chunk=2048):
        self._ae, self._chunk = ae, chunk

    def encode(self, x):
        zs = [self._ae.encode(x[i:i + self._chunk]) for i in range(0, x.shape[0], self._chunk)]
        return torch.cat([z[0] for z in zs]), torch.cat([z[1] for z in zs])

    def __getattr__(self, k):
        return getattr(self.__dict__["_ae"], k)


def build_data(ae, path, n_val=500, ae_path=None):
    """ActionHierData on the bridge train file (zero actions). W/speed defaults are irrelevant:
    they only feed the pan-geometry hardness labels, which the bridge never queries (rhvd.train
    uses window_batch, and the bridge has its own eval). Caches the [STD-FIX] stats next to the
    AE so --stage eval can standardize without re-encoding the train set."""
    data = ActionHierData(_ChunkedEnc(ae), path, path.replace(".npy", "_actions.npy"),
                          n_ctx=6, k=4, n_val=n_val)
    ae._lat_stats = data.stats                          # the wrapper got it; the real AE needs it
    if ae_path is not None:
        torch.save({"stats": data.stats, "train_path": path}, stats_path(ae_path))
        log(f"[data] cached latent stats -> {stats_path(ae_path)}")
    return data


def load_stats(ae, ae_path, n_train):
    """Set ae._lat_stats from the cache (or rebuild it from the train file if missing)."""
    sp = stats_path(ae_path)
    if os.path.exists(sp):
        st = torch.load(sp, map_location="cpu", weights_only=False)
        ae._lat_stats = tuple(st["stats"])
        log(f"[stats] loaded {sp} (train={st.get('train_path')})")
        return
    log(f"[stats] no cache at {sp} -- rebuilding from the train set (one-time encode)")
    build_data(ae, train_path(n_train), ae_path=ae_path)


# ================================================================================================
# 3) TRAIN: AE (hier_convae CLI equivalent) + arms (rhvd.train, unmodified).
# ================================================================================================
def run_ae(n_train, ae_path, ae_steps):
    path = train_path(n_train)
    assert os.path.exists(path), f"train file missing -- run --stage gen first ({path})"
    t0 = time.time()
    hier_convae.train(path, ae_path, steps=ae_steps)
    log(f"[ae] {ae_steps} steps in {time.time()-t0:.1f}s -> {ae_path}")


def run_train(args):
    M._BF16 = bool(args.bf16)
    ae = hier_convae.load(args.ae_path)
    data = build_data(ae, train_path(args.n_train), n_val=min(500, args.n_train // 4),
                      ae_path=args.ae_path)
    t0 = time.time()
    # test=data: rhvd.train uses its `test` arg only for periodic sample grids, drawn from
    # val_idx = held-out tail clips of the GENERATED train file -- the canonical OpenSTL test
    # set stays untouched until --stage eval.
    R.train(args.arm, data, data, ae, args.steps, args.batch, args.seed,
            val_every=args.val_every, save_every=args.save_every, grid_every=args.grid_every)
    dt = time.time() - t0
    log(f"[train {args.arm}] {args.steps} steps in {dt/60:.1f} min ({dt/max(1,args.steps):.3f}s/step "
        f"incl. data-encode overhead) -> logs_rhcp/rhvd_{args.arm}_s{args.seed}_t{args.steps}.pt")


# ================================================================================================
# 4) EVAL: 10->10 on the canonical test set, OpenSTL convention + ceiling/persistence rows.
# ================================================================================================
_LPIPS_NET = None


def _lpips_mean(pred, true, batch=256):
    """Full-frame LPIPS (alex) between (N,T,1,64,64) [0,1] arrays; None if lpips unavailable."""
    global _LPIPS_NET
    try:
        import lpips
        if _LPIPS_NET is None:
            _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    except Exception as e:  # noqa: BLE001  (no package / no cached weights offline)
        log(f"  (LPIPS skipped: {type(e).__name__}: {e})")
        return None
    p = torch.from_numpy(pred.reshape(-1, 1, 64, 64))
    g = torch.from_numpy(true.reshape(-1, 1, 64, 64))
    vals = []
    with torch.no_grad():
        for i in range(0, p.shape[0], batch):
            a = (p[i:i + batch].to(DEV) * 2 - 1).repeat(1, 3, 1, 1)
            b = (g[i:i + batch].to(DEV) * 2 - 1).repeat(1, 3, 1, 1)
            vals.append(_LPIPS_NET(a, b).view(-1).float().cpu())
    return float(torch.cat(vals).mean())


@torch.no_grad()
def bridge_eval(arm, ae, steps, seed=0, nfe=16, n_eval=10000, chunk=50, tag=""):
    """AR-roll 3 windows (frames 10-13, 14-17, 18-21 -> crop to 18-19) from the 10 GT context
    frames of each canonical test sequence; score in the OpenSTL convention via eval_adapters."""
    n_ctx_c, n_ctx_f, mode, _, _ = R.ARMS[arm]
    assert os.path.exists(EA.OPENSTL_TEST), \
        f"canonical mnist_test_seq.npy MISSING at {EA.OPENSTL_TEST} -- protocol requires it, STOP"
    raw = np.load(EA.OPENSTL_TEST, mmap_mode="r")       # (20, 10000, 64, 64) uint8
    assert raw.shape[0] == 20 and raw.shape[2:] == (64, 64), f"unexpected layout {raw.shape}"
    n_eval = min(n_eval, raw.shape[1])
    assert getattr(ae, "_lat_stats", None) is not None, "call load_stats() before bridge_eval()"
    mu_c, sd_c, mu_f, sd_f = [t.to(DEV) for t in ae._lat_stats]

    ckpt = f"logs_rhcp/rhvd_{arm}_s{seed}_t{steps}.pt"
    st = torch.load(ckpt, map_location=DEV, weights_only=False)
    net = R.WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
    net.load_state_dict(st["ema"])                      # eval on the EMA weights
    net.eval()
    wrap = R.WindowDiffusion(net) if mode == "diff" else R.WindowRegression(net)
    log(f"[eval {arm}] ckpt {ckpt} @ step {st['step']} | EMA | nfe {nfe} | {n_eval} test seqs")

    T_pred, base = 10, net.base                          # base == 10 for both bridge arms
    pred = np.empty((n_eval, T_pred, 1, 64, 64), np.float32)
    true = np.empty_like(pred)
    rt = np.empty_like(pred)                             # AE round-trip ceiling
    pers = np.empty_like(pred)                           # persistence (repeat frame 9)
    grid_rows = None
    t0 = time.time()
    for i0 in range(0, n_eval, chunk):
        sl = np.asarray(raw[:, i0:i0 + chunk])           # (20,B,64,64) -- mmap slice, then convert
        clips01 = EA.to_ntchw(sl, time_axis=0)           # (B,20,1,64,64) float [0,1]
        B = clips01.shape[0]
        x = torch.from_numpy(clips01).to(DEV) * 2 - 1    # [-1,1], the AE's input range
        zc, zf = ae.encode(x[:, :10].reshape(B * 10, 1, 64, 64))
        zc = ((zc - mu_c) / sd_c).view(B, 10, LAT, 4, 4)
        zf = ((zf - mu_f) / sd_f).view(B, 10, LAT, 8, 8)
        buf_c = zc if n_ctx_c > 0 else torch.zeros(B, 0, LAT, 4, 4, device=DEV)
        buf_f = zf
        acts = torch.zeros(B, R.WIN, dtype=torch.long, device=DEV)   # constant no-op action
        frames = []
        for w in range(3):                               # windows at t=10,14,18 (last -> crop 2)
            torch.manual_seed(4242 + 1000 * (i0 // chunk) + w)       # CRN across arms
            ctx_c = buf_c[:, -n_ctx_c:] if n_ctx_c > 0 else buf_c
            yc, yf = wrap.sample(ctx_c, buf_f[:, -n_ctx_f:], acts, n_steps=nfe)
            if n_ctx_c > 0:
                buf_c = torch.cat([buf_c, yc], 1)[:, -base:]
            buf_f = torch.cat([buf_f, yf], 1)[:, -base:]
            for j in range(R.WIN):
                frames.append(decode01(ae, yf[:, j]).cpu())
        pred[i0:i0 + B] = torch.stack(frames, 1)[:, :T_pred].numpy()
        true[i0:i0 + B] = clips01[:, 10:20]
        _, zf_gt = ae.encode(x[:, 10:20].reshape(B * 10, 1, 64, 64))
        rt[i0:i0 + B] = ((ae.decode_fine(zf_gt).clamp(-1, 1) + 1) * 0.5) \
            .view(B, 10, 1, 64, 64).cpu().numpy()
        pers[i0:i0 + B] = np.repeat(clips01[:, 9:10], T_pred, axis=1)
        if grid_rows is None:
            grid_rows = []
            for h in (1, 5, 10):
                grid_rows += [(f"pred_h{h}", torch.from_numpy(pred[i0:i0 + B, h - 1])),
                              (f"GT_h{h}", torch.from_numpy(true[i0:i0 + B, h - 1]))]
            viz.save_samples(f"logs_rhcp/mmnist_bridge_{arm}_s{seed}_t{steps}{tag}.png",
                             grid_rows, n=8)
        if (i0 // chunk) % 20 == 0:
            log(f"  [eval {arm}] seq {i0}..{i0+B-1} / {n_eval}  ({time.time()-t0:.0f}s)")

    rows = [(arm, pred), ("AE-roundtrip", rt), ("persistence", pers)]
    results = {}
    log(f"\n{'='*88}\n  MMNIST BRIDGE 10->10 (OpenSTL convention; n={n_eval} canonical test seqs; "
        f"nfe={nfe})\n{'='*88}")
    log(f"  {'row':<14}{'MSE':>9}{'MAE':>10}{'SSIM':>9}{'PSNR':>8}{'LPIPS':>9}")
    for name, p in rows:
        m = EA.evaluate(p, true)
        m["lpips"] = _lpips_mean(p, true)
        results[name] = m
        lp = f"{m['lpips']:>9.4f}" if m["lpips"] is not None else f"{'n/a':>9}"
        log(f"  {name:<14}{m['mse']:>9.2f}{m['mae']:>10.2f}{m['ssim']:>9.4f}{m['psnr']:>8.2f}{lp}")
    # per-horizon pixel-sum MSE curve for the model row (appendix drift plot)
    mse_h = np.mean((pred - true) ** 2, axis=0).reshape(T_pred, -1).sum(axis=1)
    log("  per-frame MSE (pixel-sum): " + " ".join(f"h{h+1}:{v:.1f}" for h, v in enumerate(mse_h)))
    out = f"logs_rhcp/mmnist_bridge_eval_{arm}_s{seed}_t{steps}{tag}.npz"
    np.savez(out, mse_h=mse_h, n_eval=n_eval, nfe=nfe,
             **{f"{name.replace('-','_')}_{k}": (np.nan if v is None else v)
                for name, m in results.items() for k, v in m.items()})
    log(f"  saved {out}")
    return results


def run_eval(args):
    M._BF16 = bool(args.bf16)
    ae = hier_convae.load(args.ae_path)
    load_stats(ae, args.ae_path, args.n_train)
    bridge_eval(args.arm, ae, args.steps, seed=args.seed, nfe=args.nfe,
                n_eval=args.n_eval, chunk=args.eval_chunk, tag=args.eval_tag)


# ================================================================================================
# 5) SMOKE: end-to-end miniature chain (gen 400 -> AE 200 -> both arms 200 -> eval 32).
# ================================================================================================
def run_smoke(args):
    M._BF16 = bool(args.bf16)
    n_train, ae_path = 400, "logs_rhcp/hier_ae_mmnist_smoke.pt"
    log(f"=== MMNIST BRIDGE SMOKE | device {DEV} | bf16 {M._BF16} ===")
    times = {}
    t = time.time(); run_gen(n_train); times["gen"] = time.time() - t
    t = time.time(); run_ae(n_train, ae_path, ae_steps=200); times["ae"] = time.time() - t
    ae = hier_convae.load(ae_path)
    t = time.time()
    data = build_data(ae, train_path(n_train), n_val=100, ae_path=ae_path)
    times["encode"] = time.time() - t
    for arm in ("mm-flat", "mm-hier"):
        t = time.time()
        R.train(arm, data, data, ae, 200, 16, args.seed,
                val_every=50, save_every=200, grid_every=200)
        times[f"train {arm}"] = time.time() - t
    for arm in ("mm-flat", "mm-hier"):
        t = time.time()
        bridge_eval(arm, ae, 200, seed=args.seed, nfe=16, n_eval=32, chunk=16, tag="_smoke")
        times[f"eval {arm}"] = time.time() - t
    log("\n[smoke] wall-times: " + " | ".join(f"{k} {v:.1f}s" for k, v in times.items()))
    log("SMOKE OK")


# ================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["gen", "ae", "train", "eval", "smoke"])
    ap.add_argument("--arm", default="mm-flat", choices=list(BRIDGE_ARMS))
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n_train", type=int, default=10000, help="generated train sequences")
    ap.add_argument("--ae_path", default=AE_PATH)
    ap.add_argument("--ae_steps", type=int, default=7000)
    ap.add_argument("--n_eval", type=int, default=10000, help="canonical test seqs to score")
    ap.add_argument("--eval_chunk", type=int, default=50)
    ap.add_argument("--eval_tag", default="")
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--grid_every", type=int, default=2000)
    args = ap.parse_args()

    if args.stage == "gen":
        run_gen(args.n_train)
    elif args.stage == "ae":
        run_ae(args.n_train, args.ae_path, args.ae_steps)
    elif args.stage == "train":
        run_train(args)
    elif args.stage == "eval":
        run_eval(args)
    elif args.stage == "smoke":
        run_smoke(args)


if __name__ == "__main__":
    main()
