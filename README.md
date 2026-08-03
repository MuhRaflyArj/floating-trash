# Floating Trash Research

This directory presents the codebase for YOLO11-LCA training, detector evaluation, online multi-object tracking, MOT metrics, and multi-line counting.

## Structure

The four root command files contain and run their corresponding pipeline classes:

```text
train.py     -> TrainingPipeline
evaluate.py  -> EvaluationPipeline
track.py     -> TrackingPipeline
count.py     -> CountingPipeline
```

The `base/` package defines the abstract pipeline, evaluator, tracker, metric, and counter contracts. The `schema/` package defines typed configuration, observations, sequences, events, and numeric results.

Each package exports its public classes and functions through `__init__.py`. Callers can therefore import and use a
component directly, such as `from floating_trash.track import Detector`, followed by `Detector(...)`.

## Configuration

All examples are stored in `configs/config.example.yaml`. Every command reads only its matching top-level section.

```bash
uv run python train.py --config configs/config.example.yaml
uv run python evaluate.py --config configs/config.example.yaml
uv run python track.py --config configs/config.example.yaml
uv run python count.py --config configs/config.example.yaml
```

Set `evaluate.coco_eval` to `false` for `YOLO.val()` or `true` for the official `pycocotools.COCOeval` path.

The tracking and counting sections expose the three tracker-grid variables from the thesis:

```text
detector_confidence
track_threshold
match_threshold
```

ByteTrack and BoT-SORT are selected with `tracker: bytetrack` or `tracker: botsort`.

## Training and Detection Evaluation

`train.py` shows model loading, training, and extraction of the final mAP50 result. Its dependencies are created in
`TrainingPipeline.__init__`, while the execution method calls the existing trainer object.

`floating_trash/evaluate/ultralytics_eval.py` reports precision, recall, F1, mAP50, and mAP50-95.

`floating_trash/evaluate/coco_eval.py` contains the complete explanatory COCO path: reading the YOLO test split, converting labels and predictions, running `COCOeval`, and returning AP50-95, AP50, AP75, and AP-small.

Detection labels use:

```text
class_id x_center y_center width height
```

## Tracking

`floating_trash/track/detector.py` creates frame detections. `bot_sort.py` and `byte_track.py` apply the same detector, track, and matching thresholds to separate tracker classes. `sequence.py` preserves chronological frame order.

`floating_trash/track/mot_metrics.py` converts ground truth and predictions into temporary MOTChallenge records and returns IDF1, MOTA, and IDSW through TrackEval.

Tracking labels use:

```text
track_id class_id x_center y_center width height
```

## Multi-Line Counting

`floating_trash/count/counting_line.py` converts normalized positions into horizontal lines and tests previous-to-current center segments for crossings.

`floating_trash/count/counter.py` keeps full-video `counted_ids` memory. One retained identity contributes at most one count across every active line and produces a cumulative one-second count series.

The example uses the retained double-line configuration at 25% and 75% of frame height. `floating_trash/count/metrics.py` reports the ground-truth count, predicted count, and time-series MAE.

## Output

Each command prints its numeric result and writes the same values to:

```text
runs/<task>/<run_name>/metrics.json
```

The source is designed for research explanation. It uses realistic APIs and computation paths while deliberately excluding MLflow, GUIs, visual exports, dataset construction, grid automation, and production orchestration.
