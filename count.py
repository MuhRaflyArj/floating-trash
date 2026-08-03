"""Count floating trash with normalized virtual lines."""

import argparse
from pathlib import Path

from floating_trash.base import BasePipeline
from floating_trash.count import CountingMetricEvaluator
from floating_trash.schema import CountConfig, CountingMetrics
from floating_trash.track import (
    BotSortTracker,
    ByteTrackTracker,
    Detector,
    TrackingLabelReader,
    TrackingSequenceProcessor,
)
from floating_trash.utils import ConfigurationLoader, FrameReader, ResultWriter


class CountingPipeline(BasePipeline[CountConfig, CountingMetrics]):
    """Coordinate tracking, ground truth, and multi-line count evaluation."""

    def __init__(self, config_path: Path) -> None:
        """Create every counting component."""
        self.config_loader = ConfigurationLoader(config_path)

        count_config = self.config_loader.load("count", CountConfig)
        writer = ResultWriter("count", count_config.run_name)

        super().__init__(count_config, writer)

        self.frame_reader = FrameReader(count_config.source)
        self.detector = Detector(count_config)

        if count_config.tracker == "botsort":
            self.tracker = BotSortTracker(count_config)
        elif count_config.tracker == "bytetrack":
            self.tracker = ByteTrackTracker(count_config)

        self.sequence_processor = TrackingSequenceProcessor(self.detector, self.tracker)
        self.label_reader = TrackingLabelReader(count_config.ground_truth)
        self.metric_evaluator = CountingMetricEvaluator(count_config.line_positions, count_config.fps)

    def execute(self) -> CountingMetrics:
        """Produce trajectories and return counting metrics."""
        batch = self.frame_reader.read()

        prediction = self.sequence_processor.process(batch)
        ground_truth = self.label_reader.read(batch)

        return self.metric_evaluator.evaluate(ground_truth, prediction)


def create_parser() -> argparse.ArgumentParser:
    """Create the counting command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/config.example.yaml"))

    return parser


def main() -> None:
    """Execute the counting pipeline."""
    args = create_parser().parse_args()
    counting_pipeline = CountingPipeline(args.config)

    counting_pipeline.run()


if __name__ == "__main__":
    main()
