"""Ultralytics-native detector evaluation."""

from ultralytics import YOLO

from floating_trash.base import BaseEvaluator
from floating_trash.schema import EvaluateConfig, UltralyticsMetrics


class UltralyticsEvaluator(BaseEvaluator):
    """Calculate the primary detector metrics with YOLO.val."""

    def __init__(self, evaluate_config: EvaluateConfig) -> None:
        """Store the evaluation configuration."""
        self.config = evaluate_config
        self.model = YOLO(str(evaluate_config.weights))

    def evaluate(self) -> UltralyticsMetrics:
        """Run Ultralytics validation and extract research metrics."""
        result = self.model.val(
            data=str(self.config.data),
            split=self.config.split,
            imgsz=self.config.image_size,
            device=self.config.device,
            plots=False,
            save=False,
        )
        precision = float(result.box.mp)
        recall = float(result.box.mr)
        denominator = precision + recall
        f1 = 2.0 * precision * recall / denominator if denominator else 0.0

        return UltralyticsMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            map50=float(result.box.map50),
            map50_95=float(result.box.map),
        )
