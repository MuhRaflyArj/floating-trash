"""Per-frame YOLO detection for tracking-by-detection."""

from typing import List

import numpy as np
from ultralytics import YOLO

from floating_trash.schema import Detection, TrackerConfig


class Detector:
    """Convert YOLO predictions into typed BoxMOT detections."""

    def __init__(self, tracker_config: TrackerConfig) -> None:
        """Load the scene-specific detector checkpoint."""
        self.config = tracker_config
        self.model = YOLO(str(tracker_config.weights), task="detect")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect trash in one frame at the configured confidence."""
        result = self.model.predict(
            frame,
            conf=self.config.detector_confidence,
            device=self.config.device,
            verbose=False,
        )[0]

        if result.boxes is None:
            return []

        coordinates = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)

        return [
            Detection(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(confidence),
                class_id=int(class_id),
            )
            for box, confidence, class_id in zip(coordinates, confidences, classes)
        ]

    def as_array(self, detections: List[Detection]) -> np.ndarray:
        """Convert typed detections into an N by 6 BoxMOT array."""
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        return np.asarray([detection.as_row() for detection in detections], dtype=np.float32)
