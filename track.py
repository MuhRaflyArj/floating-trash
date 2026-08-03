"""Track floating trash and report MOT metrics."""

import argparse
from pathlib import Path

from floating_trash.base import BasePipeline
from floating_trash.schema import TrackConfig, TrackingMetrics
from floating_trash.track import (
    BotSortTracker,
    ByteTrackTracker,
    Detector,
    MotMetricEvaluator,
    TrackingLabelReader,
    TrackingSequenceProcessor,
)
from floating_trash.utils import ConfigurationLoader, FrameReader, ResultWriter


class TrackingPipeline(BasePipeline[TrackConfig, TrackingMetrics]):
    """Coordinate frame reading, detection, association, and MOT evaluation."""

    def __init__(self, config_path: Path) -> None:
        """Create every tracking component."""
        self.config_loader = ConfigurationLoader(config_path)

        track_config = self.config_loader.load("track", TrackConfig)
        writer = ResultWriter("track", track_config.run_name)

        super().__init__(track_config, writer)

        self.frame_reader = FrameReader(track_config.source)
        self.detector = Detector(track_config)

        if track_config.tracker == "botsort":
            self.tracker = BotSortTracker(track_config)
        else:
            self.tracker = ByteTrackTracker(track_config)

        self.sequence_processor = TrackingSequenceProcessor(self.detector, self.tracker)
        self.label_reader = TrackingLabelReader(track_config.ground_truth)
        self.metric_evaluator = MotMetricEvaluator(track_config.fps)

    def execute(self) -> TrackingMetrics:
        """Produce trajectories and return identity diagnostics."""
        batch = self.frame_reader.read()

        prediction = self.sequence_processor.process(batch)
        ground_truth = self.label_reader.read(batch)

        return self.metric_evaluator.evaluate(ground_truth, prediction)


def create_parser() -> argparse.ArgumentParser:
    """Create the tracking command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/config.example.yaml"))

    return parser


def main() -> None:
    """Execute the tracking pipeline."""
    args = create_parser().parse_args()
    tracking_pipeline = TrackingPipeline(args.config)

    tracking_pipeline.run()


if __name__ == "__main__":
    main()
