"""Context-package specifications and geometric reach calculations.

``ArmSpec`` records context and target interfaces separately. This distinction is essential for
the legacy stride-2 configuration, whose context stride and target interface change together.
``ContextGeometry`` is the evaluation-only representation used to derive pair-specific reach
populations.
"""
from hvp_rev.arm_registry import ArmSpec, FIXED_TARGET_CONTROLS, TOKENS_PER_FRAME
from hvp_rev.arm_registry import from_legacy_mapping, validate_fixed_target_pair
from hvp_rev.evaluation.reach import ContextGeometry

__all__ = [
    "ArmSpec",
    "ContextGeometry",
    "FIXED_TARGET_CONTROLS",
    "TOKENS_PER_FRAME",
    "from_legacy_mapping",
    "validate_fixed_target_pair",
]
