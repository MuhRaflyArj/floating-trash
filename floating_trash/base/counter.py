"""Abstract trajectory counter interface."""

from abc import ABC, abstractmethod

from floating_trash.schema import CountSeries, TrackObservation


class BaseCounter(ABC):
    """Convert tracked observations into a cumulative count."""

    @abstractmethod
    def update(self, observation: TrackObservation) -> None:
        """Consume one chronological tracking observation."""

    @abstractmethod
    def build_series(self, frame_count: int, fps: int) -> CountSeries:
        """Build the cumulative one-second count series."""
