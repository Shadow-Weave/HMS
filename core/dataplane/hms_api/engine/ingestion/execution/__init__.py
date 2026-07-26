"""Pure execution planning primitives for Retain."""

from .windowing import (
    FactRecordIdentity,
    FullWriteWindowPlan,
    WindowUnitResult,
    merge_window_unit_ids,
    plan_full_write_windows,
)

__all__ = [
    "FactRecordIdentity",
    "FullWriteWindowPlan",
    "WindowUnitResult",
    "merge_window_unit_ids",
    "plan_full_write_windows",
]
