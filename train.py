"""Run YOLO11-LCA training."""

import argparse
from pathlib import Path

from floating_trash.base import BasePipeline
from floating_trash.schema import TrainConfig, TrainingResult
from floating_trash.train import LoadModel, Trainer
from floating_trash.utils import ConfigurationLoader, ResultWriter


class TrainingPipeline(BasePipeline[TrainConfig, TrainingResult]):
    """Coordinate model loading, training, and metric extraction."""

    def __init__(self, config_path: Path) -> None:
        """Create every training component."""
        self.config_loader = ConfigurationLoader(config_path)

        train_config = self.config_loader.load("train", TrainConfig)
        writer = ResultWriter("train", train_config.run_name)

        super().__init__(train_config, writer)

        self.model_loader = LoadModel(train_config.model)
        self.trainer = Trainer(train_config, self.model_loader)

    def execute(self) -> TrainingResult:
        """Train the model and return its presentation metrics."""
        training_result = self.trainer.train()

        return self.trainer.extract_metrics(training_result)


def create_parser() -> argparse.ArgumentParser:
    """Create the training command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/config.example.yaml"))

    return parser


def main() -> None:
    """Execute the training pipeline."""
    args = create_parser().parse_args()
    training_pipeline = TrainingPipeline(args.config)

    training_pipeline.run()


if __name__ == "__main__":
    main()
