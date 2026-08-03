"""MOT conversion and TrackEval metric calculation."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import trackeval

from floating_trash.base import BaseMetric
from floating_trash.schema import TrackingMetrics, TrackingSequence


class MotMetricEvaluator(BaseMetric[TrackingSequence, TrackingMetrics]):
    """Prepare MOT records and calculate IDF1, MOTA, and IDSW."""

    def __init__(self, fps: int) -> None:
        """Store the sequence frame rate."""
        self.fps = fps

    def evaluate(
        self,
        ground_truth: TrackingSequence,
        prediction: TrackingSequence,
    ) -> TrackingMetrics:
        """Evaluate one predicted sequence with TrackEval."""
        with tempfile.TemporaryDirectory(prefix="floating_trash_mot_") as temporary:
            root = Path(temporary)

            self._write_layout(root, ground_truth, prediction)

            evaluator_config = trackeval.Evaluator.get_default_eval_config()
            evaluator_config.update(
                {
                    "DISPLAY_LESS_PROGRESS": True,
                    "PRINT_RESULTS": False,
                    "PRINT_CONFIG": False,
                    "TIME_PROGRESS": False,
                }
            )

            dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            dataset_config.update(
                {
                    "GT_FOLDER": str(root / "gt"),
                    "TRACKERS_FOLDER": str(root / "trackers"),
                    "TRACKER_SUB_FOLDER": "",
                    "OUTPUT_FOLDER": None,
                    "CLASSES_TO_EVAL": ["pedestrian"],
                    "SPLIT_TO_EVAL": "all",
                    "INPUT_AS_ZIP": False,
                    "PRINT_CONFIG": False,
                    "DO_PREPROC": False,
                    "TRACKERS_TO_EVAL": ["tracker"],
                    "BENCHMARK": "MOT17",
                    "SEQ_INFO": {ground_truth.name: ground_truth.frame_count},
                }
            )

            evaluator = trackeval.Evaluator(evaluator_config)
            datasets = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics = [trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]

            result, _ = evaluator.evaluate(datasets, metrics)

            return self._extract(result)

    def _write_layout(
        self,
        root: Path,
        ground_truth: TrackingSequence,
        prediction: TrackingSequence,
    ) -> None:
        """Create the temporary MOTChallenge directory layout."""
        benchmark = "MOT17-all"
        sequence_directory = root / "gt" / benchmark / ground_truth.name
        gt_directory = sequence_directory / "gt"
        tracker_directory = root / "trackers" / benchmark / "tracker"
        sequence_map = root / "gt" / "seqmaps"

        gt_directory.mkdir(parents=True)
        tracker_directory.mkdir(parents=True)
        sequence_map.mkdir(parents=True)

        (sequence_map / f"{benchmark}.txt").write_text(f"{ground_truth.name}\n", encoding="utf-8")

        (sequence_directory / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={ground_truth.name}\n"
            "imDir=img1\n"
            f"frameRate={self.fps}\n"
            f"seqLength={ground_truth.frame_count}\n"
            f"imWidth={ground_truth.frame_width}\n"
            f"imHeight={ground_truth.frame_height}\n"
            "imExt=.jpg\n",
            encoding="utf-8",
        )

        self._write_ground_truth(gt_directory / "gt.txt", ground_truth)
        self._write_predictions(tracker_directory / f"{ground_truth.name}.txt", prediction)

    def _write_ground_truth(self, path: Path, sequence: TrackingSequence) -> None:
        """Convert typed ground truth into MOTChallenge rows."""
        rows = [
            f"{item.frame_index + 1},{item.track_id},{item.x:.2f},{item.y:.2f},"
            f"{item.width:.2f},{item.height:.2f},1,1,1"
            for item in sequence.observations
        ]

        path.write_text("\n".join(rows), encoding="utf-8")

    def _write_predictions(self, path: Path, sequence: TrackingSequence) -> None:
        """Convert typed predictions into MOTChallenge rows."""
        rows = [
            f"{item.frame_index + 1},{item.track_id},{item.x:.2f},{item.y:.2f},"
            f"{item.width:.2f},{item.height:.2f},{item.confidence:.4f},-1,-1,-1"
            for item in sequence.observations
        ]

        path.write_text("\n".join(rows), encoding="utf-8")

    def _extract(self, result: Dict[str, Any]) -> TrackingMetrics:
        """Extract the three thesis tracking diagnostics."""
        bucket = result["MotChallenge2DBox"]["tracker"]["COMBINED_SEQ"]["pedestrian"]

        return TrackingMetrics(
            idf1=float(bucket["Identity"]["IDF1"]),
            mota=float(bucket["CLEAR"]["MOTA"]),
            idsw=int(bucket["CLEAR"]["IDSW"]),
        )
