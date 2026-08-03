"""Abstract interfaces shared by the research pipelines."""

from floating_trash.base.counter import BaseCounter
from floating_trash.base.evaluator import BaseEvaluator
from floating_trash.base.metric import BaseMetric
from floating_trash.base.pipeline import BasePipeline
from floating_trash.base.tracker import BaseTracker

__all__ = ["BaseCounter", "BaseEvaluator", "BaseMetric", "BasePipeline", "BaseTracker"]
