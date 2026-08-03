"""Tracking observations and ordered sequences."""

from typing import List, Tuple

from pydantic import BaseModel, Field


class TrackObservation(BaseModel):
    """Represent one tracked bounding box in one frame."""

    frame_index: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    class_id: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        """Return the bounding-box center."""
        return self.x + self.width / 2.0, self.y + self.height / 2.0


class TrackingSequence(BaseModel):
    """Collect ordered observations and frame metadata."""

    name: str
    frame_width: int
    frame_height: int
    frame_count: int
    observations: List[TrackObservation] = Field(default_factory=list)
