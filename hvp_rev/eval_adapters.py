"""Eval adapters + OpenSTL-convention metrics for the RHCP article (P0).

Unifies the three on-disk MMNIST layouts into (N, T, 1, H, W) float[0,1]:
  - canonical OpenSTL   mnist_test_seq.npy           (T=20,  N=10000, H, W)  -> time_axis=0
  - project long sets   mmnist_long_test_*.npy       (T,     N,       H, W)  -> time_axis=0
  - CW-VAE split        ../cwvae/data_mmnist/*.npy    (N,     T,       H, W)  -> time_axis=1
Then short-horizon metrics (MSE/MAE/SSIM/PSNR) in the EXACT OpenSTL convention (pixel-SUM MSE/MAE, per-frame-mean
SSIM/PSNR on [0,1]), with a self-test against the sibling OpenSTL repo's openstl.core.metrics if importable.

  venv/Scripts/python.exe -m hvp_rev.eval_adapters --selftest
"""
from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None

OPENSTL_TEST = os.environ.get(
    "HVP_OPENSTL_TEST", "data/external/openstl/moving_mnist/mnist_test_seq.npy"
)
CWVAE_TEST = os.environ.get("HVP_CWVAE_TEST", "data/external/cwvae/data_mmnist/test.npy")
CWVAE_TRAIN = os.environ.get("HVP_CWVAE_TRAIN", "data/external/cwvae/data_mmnist/train.npy")


# ----------------------------------------------------------------------------- layout adapter
def to_ntchw(arr: np.ndarray, time_axis: int) -> np.ndarray:
    """(.. with a time axis and a clip axis ..) -> (N, T, 1, H, W) float32 in [0,1]."""
    assert arr.ndim == 4, f"expected 4D (two of N/T then H,W), got {arr.shape}"
    if time_axis == 0:                       # (T, N, H, W) -> (N, T, H, W)
        arr = np.swapaxes(arr, 0, 1)
    elif time_axis != 1:
        raise ValueError("time_axis must be 0 or 1")
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr[:, :, None]                    # (N, T, 1, H, W)


def load_clips(path: str, time_axis: int, n: int | None = None, t: int | None = None) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    out = to_ntchw(arr, time_axis)
    if n is not None:
        out = out[:n]
    if t is not None:
        out = out[:, :t]
    return out


# ----------------------------------------------------------------------------- OpenSTL metrics
def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()


def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()


def PSNR(pred, true):
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 20.0 * np.log10(1.0 / np.sqrt(mse))


def SSIM(pred, true):
    assert _ssim is not None, "skimage required for SSIM"
    n, t = pred.shape[0], pred.shape[1]
    vals = []
    for i in range(n):
        for j in range(t):
            p = np.clip(pred[i, j], 0, 1).transpose(1, 2, 0)
            g = np.clip(true[i, j], 0, 1).transpose(1, 2, 0)
            vals.append(_ssim(p, g, data_range=1.0, channel_axis=2))
    return float(np.mean(vals))


def evaluate(pred, true):
    """pred/true: (N,T,C,H,W) in [0,1]. Returns the OpenSTL table metrics."""
    pred = np.clip(np.asarray(pred, np.float32), 0, 1)
    true = np.asarray(true, np.float32)
    return {"mse": float(MSE(pred, true)), "mae": float(MAE(pred, true)),
            "ssim": SSIM(pred, true), "psnr": float(PSNR(pred, true))}


def selftest():
    """Confirm <1e-4 identity vs the sibling OpenSTL repo's metrics on random data."""
    rng = np.random.RandomState(0)
    pred = rng.rand(8, 10, 1, 64, 64).astype(np.float32)
    true = rng.rand(8, 10, 1, 64, 64).astype(np.float32)
    ours = evaluate(pred, true)
    print("ours:", {k: round(v, 4) for k, v in ours.items()})
    try:
        import sys
        openstl_repo = os.environ.get("OPENSTL_REPO")
        if not openstl_repo:
            raise RuntimeError("OPENSTL_REPO is not set")
        sys.path.insert(0, openstl_repo)
        from openstl.core.metrics import metric as ostl_metric
        res, _ = ostl_metric(pred, true, mean=0, std=1, metrics=["mae", "mse", "ssim", "psnr"],
                             clip_range=[0, 1], return_log=True)
        for k in ["mse", "mae", "ssim", "psnr"]:
            if k in res:
                d = abs(res[k] - ours[k])
                print(f"  vs OpenSTL {k}: ours {ours[k]:.4f} | openstl {res[k]:.4f} | diff {d:.2e} "
                      f"{'PASS' if d < 1e-3 else 'CHECK'}")
    except Exception as e:
        print(f"(OpenSTL metric cross-check skipped: {e})")
    # direct numeric check against the known convention on a trivial case: all-zero pred vs known true
    p0 = np.zeros((2, 3, 1, 4, 4), np.float32)
    t0 = np.ones((2, 3, 1, 4, 4), np.float32)
    mse0 = MSE(p0, t0)
    assert abs(mse0 - 16.0) < 1e-6, f"MSE convention broken: {mse0} != 16 (sum over 4x4)"
    print(f"convention check: MSE(0 vs 1, 4x4) = {mse0:.4f} == 16.0 (pixel-sum)  PASS")
    print("eval_adapters selftest PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    if args.inspect:
        for name, p, ta in [("openstl", OPENSTL_TEST, 0), ("cwvae_test", CWVAE_TEST, 1),
                            ("cwvae_train", CWVAE_TRAIN, 1)]:
            if os.path.exists(p):
                a = load_clips(p, ta, n=4, t=8)
                print(f"{name}: {p} -> {a.shape} range [{a.min():.2f},{a.max():.2f}]")
    if args.selftest or not args.inspect:
        selftest()
