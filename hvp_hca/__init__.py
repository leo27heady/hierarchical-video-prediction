"""Contracts for the Path B hierarchy-allocation experiments.

The package is intentionally independent of the frozen ``hvp_rev`` evidence code.
It begins with configuration, state, and positional contracts; model code is added
only after these contracts and the registered experiment protocols are stable.
"""

from .config import ContextLevel, ExperimentConfig, load_experiment_config
from .state import (
    CoordinateFourierEncoding,
    HierarchicalState,
    TokenMetadata,
    pack_fixed_quota,
)

__all__ = [
    "ContextLevel",
    "CoordinateFourierEncoding",
    "ExperimentConfig",
    "HierarchicalState",
    "TokenMetadata",
    "load_experiment_config",
    "pack_fixed_quota",
]
