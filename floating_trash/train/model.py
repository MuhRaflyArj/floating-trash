"""YOLO11-LCA model construction."""

from typing import Any

from ultralytics import YOLO


class LoadModel:
    """Load the configured custom Ultralytics model."""

    def __init__(self, model_source: str) -> None:
        """Store the model definition or checkpoint."""
        self.model_source = model_source

        self.model = YOLO(self.model_source)

    def load(self) -> Any:
        """Return the loaded YOLO11-LCA model."""
        return self.model
