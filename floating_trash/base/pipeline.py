"""Generic lifecycle for a research-stage pipeline."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from floating_trash.utils import ResultWriter

ConfigT = TypeVar("ConfigT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


class BasePipeline(ABC, Generic[ConfigT, ResultT]):
    """Execute one configured research stage and publish its numbers."""

    def __init__(self, config: ConfigT, writer: ResultWriter) -> None:
        """Store the configured stage and its result writer."""
        self.config = config
        self.result_writer = writer

    def run(self) -> ResultT:
        """Execute the stage, print its numbers, and save JSON."""
        result = self.execute()

        self.result_writer.write(result)

        return result

    @abstractmethod
    def execute(self) -> ResultT:
        """Execute the stage-specific research operation."""
