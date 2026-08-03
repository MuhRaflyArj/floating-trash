"""Normalized line geometry and trajectory crossing."""

from typing import Tuple

from floating_trash.schema import TrackObservation


class CountingLine:
    """Represent one full-width horizontal line at a normalized height."""

    def __init__(self, normalized_position: float, frame_width: int, frame_height: int) -> None:
        """Convert a normalized height into frame coordinates."""
        self.normalized_position = normalized_position

        self.start = (0.0, normalized_position * frame_height)
        self.end = (float(frame_width), normalized_position * frame_height)

    def center(self, observation: TrackObservation) -> Tuple[float, float]:
        """Calculate the center used by the counting experiment."""
        return observation.center

    def crosses(
        self,
        previous: Tuple[float, float],
        current: Tuple[float, float],
    ) -> bool:
        """Test whether a trajectory segment changes side at this line."""
        line_y = self.start[1]

        within_width = max(previous[0], current[0]) >= self.start[0] and min(previous[0], current[0]) <= self.end[0]
        changes_side = (previous[1] - line_y) * (current[1] - line_y) <= 0.0
        moved = previous != current

        return within_width and changes_side and moved
