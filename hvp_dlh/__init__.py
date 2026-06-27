"""Faithful PyTorch reproduction of the Dynamic Latent Hierarchy (DLH).

Zakharov, Guo, Fountas — "Long-horizon Video Prediction using a Dynamic Latent Hierarchy", ICLR 2023
(arXiv:2212.14376). No official code was released; this is a from-scratch port from the paper.
"""
from hvp_dlh.config import DLHConfig

__all__ = ["DLHConfig"]
