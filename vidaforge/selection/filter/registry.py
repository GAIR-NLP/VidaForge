from __future__ import annotations

from typing import Any

from .aesthetic import AestheticFilter
from .base import FilterBase
from .motion import MotionFilter
from .optical import OpticalFilter
from .text import TextFilter


FILTERS: dict[str, type[FilterBase[Any]]] = {
    "optical": OpticalFilter,
    "motion": MotionFilter,
    "aesthetic": AestheticFilter,
    "text": TextFilter,
}
