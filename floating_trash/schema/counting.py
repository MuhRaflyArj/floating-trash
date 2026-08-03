"""Counting events and cumulative time series."""

from typing import List

from pydantic import BaseModel, Field


class CountEvent(BaseModel):
    """Record the first valid crossing for one track identity."""

    frame_index: int
    second: int
    track_id: int
    line_position: float
    cumulative_count: int


class CountSeries(BaseModel):
    """Store cumulative counts sampled once per second."""

    values: List[int] = Field(default_factory=list)

    @property
    def final_count(self) -> int:
        """Return the last cumulative count."""
        return self.values[-1] if self.values else 0
