"""Typed configuration, observation, and result schemas."""

from floating_trash.schema.config import CountConfig, EvaluateConfig, TrackConfig, TrackerConfig, TrainConfig
from floating_trash.schema.counting import CountEvent, CountSeries
from floating_trash.schema.detection import Detection
from floating_trash.schema.results import (
    CocoMetrics,
    CountingMetrics,
    TrackingMetrics,
    TrainingResult,
    UltralyticsMetrics,
)
from floating_trash.schema.tracking import TrackingSequence, TrackObservation

__all__ = [
    "CocoMetrics",
    "CountConfig",
    "CountEvent",
    "CountSeries",
    "CountingMetrics",
    "Detection",
    "EvaluateConfig",
    "TrackConfig",
    "TrackerConfig",
    "TrackingMetrics",
    "TrackingSequence",
    "TrackObservation",
    "TrainConfig",
    "TrainingResult",
    "UltralyticsMetrics",
]
