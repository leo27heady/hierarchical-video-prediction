"""Resumable checkpoints + full seeding (P0 repro spine).

A checkpoint stores everything needed to resume bit-for-bit-ish: model + cascade weights,
optimizer state, step, the full config dict, the data cache hashes, and all RNG states
(python / numpy / torch / cuda). Backward-compatible keys (`model`, `cascade`, `topo`) keep
eval/probe scripts working.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, *, tok, cascade, opt, step: int, cfg, extra: Optional[dict] = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": tok.state_dict(),
        "cascade": cascade.state_dict(),
        "topo": vars(cfg.topology),
        "config": asdict(cfg),
        "optimizer": opt.state_dict() if opt is not None else None,
        "step": int(step),
        "data": {"train_hash": cfg.data.train_hash, "val_hash": cfg.data.val_hash},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra:
        blob.update(extra)
    # atomic write: a kill DURING torch.save would otherwise corrupt the only survivor (last.pt).
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    return str(path)


def load_checkpoint(path, *, tok, cascade, opt=None, device="cpu", restore_rng: bool = True) -> int:
    """Restore states in-place; returns the saved step (resume continues at step+1)."""
    blob = torch.load(path, map_location=device, weights_only=False)
    tok.load_state_dict(blob["model"])
    cascade.load_state_dict(blob["cascade"])
    if opt is not None and blob.get("optimizer") is not None:
        opt.load_state_dict(blob["optimizer"])
    if restore_rng and "rng" in blob:
        r = blob["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        # RNG states must be CPU ByteTensors (map_location may have moved them to device)
        torch.set_rng_state(r["torch"].cpu().to(torch.uint8))
        if r.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in r["cuda"]])
    return int(blob.get("step", 0))
