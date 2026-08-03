"""Counted-ID memory and one-second cumulative count series."""

import math
from typing import List

from floating_trash.base import BaseCounter
from floating_trash.count import CountingLine
from floating_trash.schema import CountEvent, CountSeries, TrackObservation


class MultiLineCounter(BaseCounter):
    """Count each retained track identity once across all active lines."""

    def __init__(
        self,
        line_positions: List[float],
        frame_width: int,
        frame_height: int,
        fps: int,
    ) -> None:
        """Create normalized lines and empty full-video identity memory."""
        self.lines = [CountingLine(position, frame_width, frame_height) for position in line_positions]
        self.fps = fps

        self.previous_centers = {}
        self.counted_ids = set()
        self.events = []

    def update(self, observation: TrackObservation) -> None:
        """Consume one observation and register its first line crossing."""
        current = observation.center
        previous = self.previous_centers.get(observation.track_id)

        self.previous_centers[observation.track_id] = current

        if previous is None or observation.track_id in self.counted_ids:
            return

        for line in self.lines:
            if not line.crosses(previous, current):
                continue

            self.counted_ids.add(observation.track_id)

            self.events.append(
                CountEvent(
                    frame_index=observation.frame_index,
                    second=observation.frame_index // self.fps,
                    track_id=observation.track_id,
                    line_position=line.normalized_position,
                    cumulative_count=len(self.counted_ids),
                )
            )

            return

    def build_series(self, frame_count: int, fps: int) -> CountSeries:
        """Sample the cumulative count at one-second intervals."""
        seconds = max(1, math.ceil(frame_count / fps))
        events_by_second = {}

        for event in self.events:
            events_by_second.setdefault(min(event.second, seconds - 1), []).append(event)

        cumulative = 0
        values = []

        for second in range(seconds):
            cumulative += len(events_by_second.get(second, []))
            values.append(cumulative)

        return CountSeries(values=values)
