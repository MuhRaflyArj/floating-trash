"""Evaluate a YOLO11-LCA detector."""

import argparse
from pathlib import Path

from floating_trash.base import BasePipeline
from floating_trash.evaluate import CocoEvaluator, Predictor, UltralyticsEvaluator
from floating_trash.schema import CocoMetrics, EvaluateConfig, UltralyticsMetrics
from floating_trash.utils import ConfigurationLoader, ResultWriter


class EvaluationPipeline(
    BasePipeline[EvaluateConfig, CocoMetrics | UltralyticsMetrics]
):
    """Select and execute one detector evaluation strategy."""

    def __init__(self, config_path: Path) -> None:
        """Create every evaluation component."""
        self.config_loader = ConfigurationLoader(config_path)

        evaluate_config = self.config_loader.load("evaluate", EvaluateConfig)
        writer = ResultWriter("evaluate", evaluate_config.run_name)

        super().__init__(evaluate_config, writer)

        if evaluate_config.coco_eval:
            self.predictor = Predictor(evaluate_config)
            self.evaluator = CocoEvaluator(evaluate_config, self.predictor)
        else:
            self.predictor = None
            self.evaluator = UltralyticsEvaluator(evaluate_config)

    def execute(self) -> CocoMetrics | UltralyticsMetrics:
        """Run the configured evaluator."""
        return self.evaluator.evaluate()


def create_parser() -> argparse.ArgumentParser:
    """Create the evaluation command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/config.example.yaml"))

    return parser


def main() -> None:
    """Execute the evaluation pipeline."""
    args = create_parser().parse_args()
    evaluation_pipeline = EvaluationPipeline(args.config)

    evaluation_pipeline.run()


if __name__ == "__main__":
    main()
