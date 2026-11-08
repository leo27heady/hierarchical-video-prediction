"""Typed context-package registry with context and target interfaces separated.

The legacy trainer couples coarse context stride to target stride.  This registry makes the
coupling explicit and defines the fixed-target controls needed to isolate context packaging.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


WIN = 4
TOKENS_PER_FRAME = {"top": 4, "coarse": 16, "fine": 64}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    coarse_context_slots: int
    fine_recent_slots: int
    head: str = "diff"
    coarse_context_stride: int = 1
    fine_history_slots: int = 0
    top_context_slots: int = 0
    top_context_stride: int = 1
    coarse_target_stride: int = 1
    top_target_stride: int = 1
    predict_coarse: bool = True
    predict_fine: bool = True
    predict_top: bool = False
    loss_weights: tuple[float, float, float] = (0.0, 1.0, 1.0)
    status: str = "measured"
    note: str = ""

    def __post_init__(self) -> None:
        if self.head not in {"diff", "reg"}:
            raise ValueError(f"invalid head {self.head!r}")
        for stride in (self.coarse_context_stride, self.top_context_stride,
                       self.coarse_target_stride, self.top_target_stride):
            if stride < 1 or WIN % stride:
                raise ValueError(f"target/context stride {stride} must be a positive divisor of WIN={WIN}")
        if self.predict_top and not self.top_context_slots:
            raise ValueError("top target requires a top context stream")

    @property
    def context_tokens(self) -> int:
        return (
            self.top_context_slots * TOKENS_PER_FRAME["top"]
            + self.coarse_context_slots * TOKENS_PER_FRAME["coarse"]
            + (self.fine_history_slots + self.fine_recent_slots) * TOKENS_PER_FRAME["fine"]
        )

    @property
    def target_slots(self) -> dict[str, int]:
        return {
            "top": WIN // self.top_target_stride if self.predict_top else 0,
            "coarse": WIN // self.coarse_target_stride if self.predict_coarse else 0,
            "fine": WIN if self.predict_fine else 0,
        }

    @property
    def target_tokens(self) -> int:
        slots = self.target_slots
        return sum(slots[level] * TOKENS_PER_FRAME[level] for level in slots)

    @property
    def total_tokens(self) -> int:
        return self.context_tokens + self.target_tokens

    @property
    def context_reach(self) -> int:
        coarse = self.coarse_context_slots * self.coarse_context_stride
        top = self.top_context_slots * self.top_context_stride
        hist = (self.fine_recent_slots + 2 + 4 * (self.fine_history_slots - 1)
                if self.fine_history_slots else 0)
        return max(coarse, top, self.fine_recent_slots, hist)

    def resolved(self) -> dict:
        result = asdict(self)
        result.update(
            context_tokens=self.context_tokens,
            target_tokens=self.target_tokens,
            total_tokens=self.total_tokens,
            context_reach=self.context_reach,
            target_slots=self.target_slots,
        )
        return result

    @classmethod
    def from_legacy(cls, name: str, value: Sequence) -> "ArmSpec":
        if len(value) not in {5, 7}:
            raise ValueError(f"legacy arm {name} has unsupported tuple length {len(value)}")
        n_c, n_f, head, stride_c, n_hist = value[:5]
        n_top, stride_top = value[5:] if len(value) == 7 else (0, 1)
        return cls(
            name=name,
            coarse_context_slots=n_c,
            fine_recent_slots=n_f,
            head=head,
            coarse_context_stride=stride_c,
            fine_history_slots=n_hist,
            top_context_slots=n_top,
            top_context_stride=stride_top,
            # This records the legacy confound rather than hiding it.
            coarse_target_stride=stride_c if n_c and stride_c > 1 else 1,
            top_target_stride=stride_top if n_top and stride_top > 1 else 1,
            predict_top=bool(n_top),
            loss_weights=(1.0 if n_top else 0.0, 1.0, 1.0),
            note="legacy target interface coupled to context stride",
        )


def from_legacy_mapping(mapping: Mapping[str, Sequence]) -> dict[str, ArmSpec]:
    return {name: ArmSpec.from_legacy(name, value) for name, value in mapping.items()}


# New causal controls.  They are not authorized GPU runs until the experiment ledger says so.
FIXED_TARGET_CONTROLS: dict[str, ArmSpec] = {
    "hier-s2-fixed-target": ArmSpec(
        name="hier-s2-fixed-target",
        coarse_context_slots=24,
        coarse_context_stride=2,
        fine_recent_slots=2,
        coarse_target_stride=1,
        status="prepared-not-authorized",
        note="B1 context stride with the four-slot coarse target interface held fixed",
    ),
    "flat-short-fixed-target": ArmSpec(
        name="flat-short-fixed-target",
        coarse_context_slots=0,
        fine_recent_slots=8,
        coarse_target_stride=1,
        status="prepared-not-authorized",
        note="512-token single-stream context comparator with the same four-slot target interface",
    ),
}


def validate_fixed_target_pair() -> None:
    hier = FIXED_TARGET_CONTROLS["hier-s2-fixed-target"]
    flat = FIXED_TARGET_CONTROLS["flat-short-fixed-target"]
    if hier.context_tokens != flat.context_tokens:
        raise AssertionError("fixed-target B1 controls are not context-token matched")
    if hier.target_slots != flat.target_slots or hier.loss_weights != flat.loss_weights:
        raise AssertionError("fixed-target B1 controls do not share the target interface")


validate_fixed_target_pair()
