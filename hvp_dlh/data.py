"""Datasets for DLH — synthetic-first (per the plan; KTH real-world is skipped).

- MovingBall: DLH's toy stochasticity probe (Zakharov 2022). A ball bounces in a 64x64 frame and
  STOCHASTICALLY changes direction with probability p_turn each step. Generated on the fly (infinite),
  so no storage. This is the cheapest dataset and the one that exercises the dynamic clock.
- MovingMNIST: the cached numpy from the CW-VAE port (../cwvae/data_mmnist/{train,test}.npy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# --------------------------------------------------------------------------- Moving Ball
class MovingBallDataset(IterableDataset):
    """Infinite stream of bouncing-ball sequences. (T, C, 64, 64) float32 in [0,1].
    Two stochastic features at DIFFERENT timescales (the point of the 2-level test, paper §3.4):
      - position: velocity resampled with prob p_turn each step (FAST feature -> bottom level)
      - colour:   resampled with prob p_colour each step, only if channels>1 (SLOW feature -> top level,
                  matching the paper's colour-changing Moving Ball; gives the hierarchy a real job)."""

    def __init__(self, seq_len=100, size=64, radius=6, speed=4.0, p_turn=0.1, p_colour=0.05,
                 channels=1, seed=0):
        self.seq_len, self.size, self.radius = seq_len, size, radius
        self.speed, self.p_turn, self.p_colour, self.channels = speed, p_turn, p_colour, channels
        self.seed = seed

    def _rand_colour(self, rng):
        # saturated, well-separated colours so a change is a clear global event
        c = rng.uniform(0.2, 1.0, size=self.channels); c[rng.integers(self.channels)] = 1.0
        return c

    def _gen(self, rng):
        S, r = self.size, self.radius
        pos = rng.uniform(r, S - r, size=2)
        ang = rng.uniform(0, 2 * np.pi)
        vel = self.speed * np.array([np.cos(ang), np.sin(ang)])
        yy, xx = np.mgrid[0:S, 0:S]
        col = self._rand_colour(rng) if self.channels > 1 else np.array([1.0])
        frames = np.zeros((self.seq_len, self.channels, S, S), np.float32)
        for t in range(self.seq_len):
            if rng.random() < self.p_turn:                          # FAST: stochastic direction change
                ang = rng.uniform(0, 2 * np.pi)
                vel = self.speed * np.array([np.cos(ang), np.sin(ang)])
            if self.channels > 1 and rng.random() < self.p_colour:  # SLOW: stochastic colour change
                col = self._rand_colour(rng)
            pos = pos + vel
            for d in range(2):                                      # bounce off walls
                if pos[d] < r:
                    pos[d] = 2 * r - pos[d]; vel[d] = -vel[d]
                elif pos[d] > S - r:
                    pos[d] = 2 * (S - r) - pos[d]; vel[d] = -vel[d]
            mask = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2 <= r ** 2
            for c in range(self.channels):
                frames[t, c][mask] = col[c]
        return frames

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = np.random.default_rng(self.seed + 1000 * wid)
        while True:
            yield torch.from_numpy(self._gen(rng))


# --------------------------------------------------------------------------- Moving MNIST (cached npy)
class MovingMNISTDataset(Dataset):
    """Cached MMNIST clips. train.npy (N,100,64,64) uint8 -> random seq_len crops; test gives full clips."""

    def __init__(self, root, split="train", seq_len=100):
        arr = np.load(Path(root) / f"{split}.npy")                  # (N,T,64,64) uint8
        self.data = arr
        self.seq_len = seq_len
        self.split = split

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        clip = self.data[i]                                         # (T,64,64) uint8
        T = clip.shape[0]
        if self.split == "train" and T > self.seq_len:
            s = np.random.randint(0, T - self.seq_len + 1)
            clip = clip[s:s + self.seq_len]
        x = torch.from_numpy(clip.astype(np.float32) / 255.0).unsqueeze(1)   # (T,1,64,64)
        return x


# --------------------------------------------------------------------------- factory
def make_loaders(cfg):
    """Returns (train_iter_fn, test_dataset). train_iter_fn() yields a (B,T,C,H,W) batch."""
    if cfg.dataset == "moving_ball":
        bk = dict(channels=cfg.image_channels, p_turn=cfg.ball_p_turn, p_colour=cfg.ball_p_colour)
        ds = MovingBallDataset(seq_len=cfg.seq_len, seed=cfg.seed, **bk)
        dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=True)
        test = MovingBallDataset(seq_len=cfg.ctx_frames + cfg.eval_horizon, seed=cfg.seed + 777, **bk)
        return dl, test
    elif cfg.dataset == "mmnist":
        ds = MovingMNISTDataset(cfg.data_root, "train", cfg.seq_len)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
        test = MovingMNISTDataset(cfg.data_root, "test", cfg.ctx_frames + cfg.eval_horizon)
        return dl, test
    else:
        raise ValueError(f"dataset {cfg.dataset} not wired yet")
