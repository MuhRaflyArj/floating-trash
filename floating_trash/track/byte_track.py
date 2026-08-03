"""ByteTrack implementation for the thesis comparison."""

import numpy as np
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

from floating_trash.base import BaseTracker
from floating_trash.schema import TrackerConfig


class ByteTrackTracker(BaseTracker):
    """Associate detections with ByteTrack and fixed research settings."""

    def __init__(self, tracker_config: TrackerConfig) -> None:
        """Create ByteTrack with the selected association thresholds."""
        self.config = tracker_config

        self.tracker = ByteTrack(
            min_conf=0.05,
            track_thresh=self.config.track_threshold,
            match_thresh=self.config.match_threshold,
            track_buffer=300,
            frame_rate=self.config.fps,
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Associate detections in one chronological frame."""
        return self.tracker.update(detections, frame)
