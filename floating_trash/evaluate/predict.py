"""Detection prediction collection."""

from pathlib import Path
from typing import Any, Iterable, List

from ultralytics import YOLO

from floating_trash.schema import EvaluateConfig


class Predictor:
    """Stream YOLO predictions for an ordered image collection."""

    def __init__(self, evaluate_config: EvaluateConfig) -> None:
        """Load the configured evaluation checkpoint."""
        self.config = evaluate_config
        self.model = YOLO(str(evaluate_config.weights), task="detect")

    def predict(self, image_paths: List[Path]) -> Iterable[Any]:
        """Yield low-confidence predictions used by COCO evaluation."""
        return self.model.predict(
            source=[str(path) for path in image_paths],
            imgsz=self.config.image_size,
            device=self.config.device,
            conf=0.001,
            max_det=100,
            stream=True,
            verbose=False,
            save=False,
        )
