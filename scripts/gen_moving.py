"""Generate long Moving-MNIST / Moving-FashionMNIST clips (OpenSTL-matched protocol) for the 2nd-dataset drift check.

Protocol matches OpenSTL/CW-VAE exactly: 2 digits, step 0.1, bounce off walls (per-axis sign flip), max-composite,
digit 28px on a 64px canvas, random init position + unit-circle velocity. Deterministic (p_turn=0).

  venv/Scripts/python.exe scripts/gen_moving.py --src fmnist --frames 76 --n 2000 --out data/moving_fmnist_long_2000x76.npy
"""
from __future__ import annotations

import argparse
import gzip
import os

import numpy as np

SRC = {
    "mnist": "C:/Users/leoni/Documents/projects/OpenSTL/data/moving_mnist/train-images-idx3-ubyte.gz",
    "fmnist": "C:/Users/leoni/Documents/projects/OpenSTL/data/moving_fmnist/train-images-idx3-ubyte.gz",
}
DIGIT = 28


def load_idx3(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)


def traj(seq_len, image_size, rng, step=0.1):
    canvas = image_size - DIGIT
    x, y = rng.rand(), rng.rand()
    theta = rng.rand() * 2 * np.pi
    vx, vy = np.cos(theta), np.sin(theta)
    bx, by = 1.0, 1.0
    sx, sy = np.zeros(seq_len), np.zeros(seq_len)
    for i in range(seq_len):
        y += by * vy * step
        x += bx * vx * step
        if x <= 0: x = 0; bx = -bx
        if x >= 1: x = 1; bx = -bx
        if y <= 0: y = 0; by = -by
        if y >= 1: y = 1; by = -by
        sx[i], sy[i] = x, y
    return (canvas * sy).astype(np.int32), (canvas * sx).astype(np.int32)


def gen_clip(n_frames, image_size, digits, rng, num=2):
    out = np.zeros((n_frames, image_size, image_size), np.float32)
    for _ in range(num):
        d = digits[rng.randint(len(digits))]
        ys, xs = traj(n_frames, image_size, rng)
        for t in range(n_frames):
            out[t, ys[t]:ys[t] + DIGIT, xs[t]:xs[t] + DIGIT] = np.maximum(
                out[t, ys[t]:ys[t] + DIGIT, xs[t]:xs[t] + DIGIT], d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["mnist", "fmnist"], default="fmnist")
    ap.add_argument("--frames", type=int, default=76)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    digits = load_idx3(SRC[args.src]).astype(np.float32)   # (N,28,28) in [0,255]
    rng = np.random.RandomState(args.seed)
    arr = np.zeros((args.n, args.frames, args.size, args.size), np.uint8)
    for i in range(args.n):
        clip = gen_clip(args.frames, args.size, digits, rng, args.digits)
        arr[i] = np.clip(clip, 0, 255).astype(np.uint8)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.n}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, arr)                                  # (N, T, 64, 64) -> load with time_axis=1
    print(f"saved {args.out}  shape {arr.shape}  range [{arr.min()},{arr.max()}]")


if __name__ == "__main__":
    main()
