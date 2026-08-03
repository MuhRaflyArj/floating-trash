"""Cumulative counting metric calculation."""

from typing import List

from floating_trash.base import BaseMetric
from floating_trash.count import MultiLineCounter
from floating_trash.schema import CountSeries, CountingMetrics, TrackingSequence


class CountingMetricEvaluator(BaseMetric[TrackingSequence, CountingMetrics]):
    """Compare ground-truth and predicted cumulative count trajectories."""

    def __init__(self, line_positions: List[float], fps: int) -> None:
        """Store the common line configuration and sampling rate."""
        self.line_positions = line_positions
        self.fps = fps

    def evaluate(
        self,
        ground_truth: TrackingSequence,
        prediction: TrackingSequence,
    ) -> CountingMetrics:
        """Calculate final counts and one-second time-series MAE."""
        ground_truth_series = self._build_series(ground_truth)
        prediction_series = self._build_series(prediction)
        differences = [
            abs(predicted - actual)
            for actual, predicted in zip(ground_truth_series.values, prediction_series.values)
        ]
        mae = sum(differences) / len(differences) if differences else 0.0

        return CountingMetrics(
            gt_count=ground_truth_series.final_count,
            predicted_count=prediction_series.final_count,
            time_series_mae=mae,
        )

    def _build_series(self, sequence: TrackingSequence) -> CountSeries:
        """Run one identity sequence through the shared multi-line counter."""
        line_counter = MultiLineCounter(
            self.line_positions,
            sequence.frame_width,
            sequence.frame_height,
            self.fps,
        )

        for observation in sorted(sequence.observations, key=lambda item: (item.frame_index, item.track_id)):
            line_counter.update(observation)

        return line_counter.build_series(sequence.frame_count, self.fps)
