"""Numeric result schemas printed by the entry points."""

from pydantic import BaseModel


class TrainingResult(BaseModel):
    """Summarize the completed training run."""

    final_map50: float
    epochs_completed: int


class UltralyticsMetrics(BaseModel):
    """Report the primary Ultralytics detection metrics."""

    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float


class CocoMetrics(BaseModel):
    """Report the selected COCO-style detection metrics."""

    ap50_95: float
    ap50: float
    ap75: float
    ap_small: float


class TrackingMetrics(BaseModel):
    """Report identity and CLEAR MOT diagnostics."""

    idf1: float
    mota: float
    idsw: int


class CountingMetrics(BaseModel):
    """Report final counts and cumulative time-series error."""

    gt_count: int
    predicted_count: int
    time_series_mae: float
