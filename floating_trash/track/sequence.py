"""Chronological detector-to-tracker sequence processing."""

from floating_trash.base import BaseTracker
from floating_trash.schema import TrackingSequence, TrackObservation
from floating_trash.track import Detector
from floating_trash.utils import FrameBatch


class TrackingSequenceProcessor:
    """Transform ordered frames into identity-associated trajectories."""

    def __init__(self, detection_model: Detector, object_tracker: BaseTracker) -> None:
        """Store detection and association components."""
        self.detector = detection_model
        self.tracker = object_tracker

    def process(self, batch: FrameBatch) -> TrackingSequence:
        """Detect and associate objects in every frame."""
        observations = []

        for frame_index, frame in enumerate(batch.frames):
            detections = self.detector.as_array(self.detector.detect(frame))
            tracks = self.tracker.update(detections, frame)

            if tracks is None:
                continue

            for row in tracks:
                x1, y1, x2, y2, track_id, confidence = row[:6]
                class_id = int(row[6]) if len(row) > 6 else 0

                observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        track_id=int(track_id),
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                        confidence=float(confidence),
                        class_id=class_id,
                    )
                )

        return TrackingSequence(
            name=batch.name,
            frame_width=batch.width,
            frame_height=batch.height,
            frame_count=len(batch.frames),
            observations=observations,
        )
