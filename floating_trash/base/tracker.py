"""Abstract online tracker interface."""

from abc import ABC, abstractmethod

import numpy as np


class BaseTracker(ABC):
    """Convert frame detections into persistent identities."""

    @abstractmethod
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Update identities for one chronological frame."""
