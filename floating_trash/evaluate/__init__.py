"""Detection evaluation components."""

from floating_trash.evaluate.predict import Predictor
from floating_trash.evaluate.ultralytics_eval import UltralyticsEvaluator
from floating_trash.evaluate.coco_eval import CocoEvaluator

__all__ = ["CocoEvaluator", "Predictor", "UltralyticsEvaluator"]
