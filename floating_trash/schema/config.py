"""Configuration schemas for the four research stages."""

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Configure YOLO11-LCA training."""

    model: str
    data: Path
    epochs: int = Field(gt=0)
    image_size: int = Field(default=640, gt=0)
    batch_size: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=0.0005, gt=0)
    optimizer: str = "Adam"
    device: str = "0"
    run_name: str = "train"


class EvaluateConfig(BaseModel):
    """Configure Ultralytics or COCO detection evaluation."""

    weights: Path
    data: Path
    split: str = "test"
    coco_eval: bool = False
    image_size: int = Field(default=640, gt=0)
    device: str = "0"
    run_name: str = "evaluate"


class TrackerConfig(BaseModel):
    """Define shared detector and tracker settings."""

    weights: Path
    source: Path
    ground_truth: Path
    tracker: str = "botsort"
    detector_confidence: float = Field(default=0.40, ge=0, le=1)
    track_threshold: float = Field(default=0.20, ge=0, le=1)
    match_threshold: float = Field(default=0.95, ge=0, le=1)
    device: str = "0"
    fps: int = Field(default=30, gt=0)
    run_name: str = "track"


class TrackConfig(TrackerConfig):
    """Configure MOT evaluation."""


class CountConfig(TrackerConfig):
    """Configure trajectory-based multi-line counting."""

    line_positions: List[float] = Field(default_factory=lambda: [0.25, 0.75], min_length=1)
