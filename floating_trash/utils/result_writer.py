"""Print and persist compact numeric results."""

import json
from pathlib import Path

from pydantic import BaseModel


class ResultWriter:
    """Write one result dictionary to stdout and a run directory."""

    def __init__(self, task_name: str, run_name: str) -> None:
        """Set the output location."""
        self.output_directory = Path("runs") / task_name / run_name

    def write(self, result: BaseModel) -> Path:
        """Print numeric fields and save them as JSON."""
        values = result.model_dump(mode="json")

        for name, value in values.items():
            print(f"{name}: {value}")

        self.output_directory.mkdir(parents=True, exist_ok=True)

        path = self.output_directory / "metrics.json"
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")

        return path
