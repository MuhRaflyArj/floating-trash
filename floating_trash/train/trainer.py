"""Ultralytics training operation."""

from typing import Any

from floating_trash.schema import TrainConfig, TrainingResult
from floating_trash.train import LoadModel


class Trainer:
    """Apply the thesis fine-tuning configuration to YOLO11-LCA."""

    def __init__(self, train_config: TrainConfig, model_loader: LoadModel) -> None:
        """Store training dependencies."""
        self.config = train_config
        self.model_loader = model_loader

    def train(self) -> Any:
        """Execute model training and return Ultralytics results."""
        model = self.model_loader.load()

        return model.train(
            data=str(self.config.data),
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            lr0=self.config.learning_rate,
            optimizer=self.config.optimizer,
            device=self.config.device,
            project="runs/train",
            name=self.config.run_name,
        )

    def extract_metrics(self, result: Any) -> TrainingResult:
        """Extract final mAP50 and completed epochs."""
        values = getattr(result, "results_dict", {})
        map50 = values.get("metrics/mAP50(B)", values.get("metrics/mAP50", 0.0))

        return TrainingResult(final_map50=float(map50), epochs_completed=self.config.epochs)
