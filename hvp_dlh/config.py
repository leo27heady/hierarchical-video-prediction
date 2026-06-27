"""Configuration for DLH. Every architectural and training knob is exposed here so the model can be
resized / reshaped without touching the modelling code (paper §A.1, Table 2 give the defaults)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence


@dataclass
class DLHConfig:
    # ---- data / observation ----
    dataset: str = "moving_ball"            # moving_ball | mmnist | mazes
    image_size: int = 64
    image_channels: int = 1                 # 1 for MMNIST/ball(gray), 3 for colour ball / mazes

    # ---- hierarchy (paper: 3 levels MMNIST/Mazes, 2 levels Moving Ball) ----
    levels: int = 2                         # N. level 0 = BOTTOM (always ticks), level N-1 = TOP
    stoch_dim: int = 20                     # |s^n|  (random state per level)
    det_dim: int = 200                      # |x^n| = |d^n| = |c^n|  (deterministic pathways)

    # ---- image conv encoder / decoder (4 conv + 4 transposed-conv, DeepMind/CW-VAE stack) ----
    conv_channels: Sequence[int] = (32, 64, 128, 256)   # encoder widths; decoder mirrors
    conv_kernel: int = 4
    enc_to_feat: bool = True                # project flattened conv -> det_dim (x^1)
    dec_out_activation: str = "tanh"        # tanh (lineage) | sigmoid | none ; targets in [0,1]

    # ---- per-level component MLPs / GRUs (paper Table 2) ----
    enc_mlp: Sequence[int] = (200,)         # f_enc:  x^n -> x^{n+1}  (bottom-up)
    dec_mlp: Sequence[int] = (200,)         # f_dec:  [s^n, c^n] -> c^{n-1}  (top-down)
    post_mlp: Sequence[int] = (40, 40, 40)  # q(s^n | x^n, c^n)
    prior_mlp: Sequence[int] = (40, 40, 40) # p(s^n | d^n, c^n)  (change prior)
    temporal_gru: int = 200                 # f_tem GRU hidden (d^n)
    prior_factor_gru: int = 200             # p(e^n | s^n_{<t}) GRU hidden
    separate_prior_factor_gru: bool = True  # faithful: dedicated GRU for the transition predictor

    # ---- stochastic head ----
    min_std: float = 0.1                    # softplus std floor (anti-collapse; tune per dataset)
    std_bias: float = 0.54                  # softplus input bias (CW-VAE/DLH-lineage init trick)

    # ---- decoder likelihood ----
    dec_stddev: float = 1.0                 # fixed Gaussian std of the image likelihood (per dataset)

    # ---- ELBO weighting ----
    beta_max: float = 1.0
    beta_warmup_iters: int = 10000          # linear 0 -> beta_max over the first N iters (paper §A.1)
    free_nats: float = 0.0                  # optional per-level state-KL floor (0 = off)
    transition_kl_weight: float = 1.0       # weight on the Bernoulli (transition) KL term
    force_tick_iters: int = 3000            # dense warmup: force ALL levels to tick (bootstrap the
                                            # upper levels out of the dead-unit cold start), then anneal
                                            # to the learned KL-selection. 0 = off (pure paper rule).

    # ---- optimisation (paper §A.1: Adam lr 5e-4, eps 1e-4) ----
    lr: float = 5e-4
    adam_eps: float = 1e-4
    grad_clip: float = 100.0
    seq_len: int = 100                      # training sequence length
    batch_size: int = 32                    # paper used 100 on a V100; 32 fits 16GB

    # ---- evaluation / rollout ----
    ctx_frames: int = 30                    # context frames before open-loop (paper: 30; CW-VAE: 36)
    eval_horizon: int = 100                 # frames predicted open-loop (300 MMNIST / 200 mazes / 100 ball)
    eval_samples: int = 10                  # best-of-N (paper final: 100; dev: 10)

    # ---- run / logging ----
    seed: int = 0
    device: str = "cuda"
    total_iters: int = 60000
    log_every: int = 100
    eval_every: int = 2000
    ckpt_every: int = 5000
    wandb_project: str = "dlh-repro"
    wandb_mode: str = "online"              # online | offline | disabled
    run_name: str = "ball-2lvl"
    logdir: str = "logs_dlh"
    data_root: str = "../cwvae/data_mmnist" # MMNIST numpy cache (train.npy/test.npy)

    def to_dict(self):
        return asdict(self)
