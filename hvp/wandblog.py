"""Thin wandb wrapper — a no-op when disabled or wandb is unavailable."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import numpy as np


def _flatten(cfg) -> dict:
    out = {}
    for k, v in asdict(cfg).items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


class WandbLogger:
    def __init__(self, cfg, run_cfg):
        self.enabled = bool(run_cfg.wandb)
        self.wandb = None
        if not self.enabled:
            return
        try:
            import wandb
        except Exception as e:  # pragma: no cover
            print(f"[wandb] unavailable ({e}); logging disabled")
            self.enabled = False
            return
        self.wandb = wandb
        wandb.init(
            project=run_cfg.wandb_project,
            name=run_cfg.wandb_name,
            mode=run_cfg.wandb_mode,
            config=_flatten(cfg),
        )

    def log(self, metrics: dict, step: int) -> None:
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def log_image_table(self, rows: List[Tuple[int, str, np.ndarray]], step: int, key: str = "predictions") -> None:
        """rows: (image_index, category['GT'|'Pred'|'Recon'], uint8 HxWx3 strip)."""
        if not self.enabled:
            return
        table = self.wandb.Table(columns=["image_index", "category", "image"])
        for idx, cat, img in rows:
            table.add_data(int(idx), cat, self.wandb.Image(img))
        self.wandb.log({key: table}, step=step)

    def finish(self) -> None:
        if self.enabled:
            self.wandb.finish()
