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
        prev_x, prev_y = previous
        curr_x, curr_y = current

        line_y = self.start[1]
        line_start_x = self.start[0]
        line_end_x = self.end[0]

        within_width = max(prev_x, curr_x) >= line_start_x and min(prev_x, curr_x) <= line_end_x
        changes_side = (prev_y - line_y) * (curr_y - line_y) <= 0.0
        moved = (prev_x, prev_y) != (curr_x, curr_y)

        return within_width and changes_side and moved
