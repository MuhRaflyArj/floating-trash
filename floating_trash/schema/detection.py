"""Detection objects passed from YOLO to a tracker."""

from typing import List

from pydantic import BaseModel


class Detection(BaseModel):
    """Represent one detector bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    def as_row(self) -> List[float]:
        """Return the BoxMOT detection row."""
        return [self.x1, self.y1, self.x2, self.y2, self.confidence, float(self.class_id)]
