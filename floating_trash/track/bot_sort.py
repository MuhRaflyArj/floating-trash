"""BoT-SORT implementation for the thesis comparison."""

import numpy as np
from boxmot.trackers.tracker_zoo import create_tracker

from floating_trash.base import BaseTracker
from floating_trash.schema import TrackerConfig


class BotSortTracker(BaseTracker):
    """Associate detections with BoT-SORT and fixed research settings."""

    def __init__(self, tracker_config: TrackerConfig) -> None:
        """Create BoT-SORT with the selected association thresholds."""
        self.config = tracker_config

        parameters = {
            "track_high_thresh": self.config.track_threshold,
            "track_low_thresh": 0.05,
            "new_track_thresh": self.config.track_threshold,
            "track_buffer": 300,
            "match_thresh": self.config.match_threshold,
            "frame_rate": self.config.fps,
            "with_reid": False,
            "cmc_method": None,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "fuse_first_associate": False,
        }

        self.tracker = create_tracker(
            tracker_type="botsort",
            reid_weights=None,
            device=self._device_name(),
            half=False,
            per_class=False,
            evolve_param_dict=parameters,
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Associate detections in one chronological frame."""
        return self.tracker.update(detections, frame)

    def _device_name(self) -> str:
        """Convert the compact YAML device into BoxMOT notation."""
        device = self.config.device.lower()

        if device == "cpu":
            return "cpu"

        if device.startswith("gpu:"):
            return device.replace("gpu:", "cuda:", 1)

        if device.isdigit():
            return f"cuda:{device}"

        return device
