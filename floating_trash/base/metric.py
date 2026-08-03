"""Abstract metric-calculation interface."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


class BaseMetric(ABC, Generic[InputT, ResultT]):
    """Calculate a typed result from evaluation inputs."""

    @abstractmethod
    def evaluate(self, ground_truth: InputT, prediction: InputT) -> ResultT:
        """Compare ground truth with a prediction."""

