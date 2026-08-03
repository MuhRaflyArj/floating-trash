"""Abstract detection evaluator interface."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseEvaluator(ABC):
    """Define a detector evaluation strategy."""

    @abstractmethod
    def evaluate(self) -> BaseModel:
        """Calculate and return detection metrics."""
