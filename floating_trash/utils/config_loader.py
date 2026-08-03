"""Load one requested section from a YAML configuration file."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class ConfigurationLoader:
    """Validate only the section required by one entry point."""

    def __init__(self, path: Path) -> None:
        """Store the YAML path."""
        self.path = path

    def load(self, section: str, schema: type[ConfigT]) -> ConfigT:
        """Load and validate a single top-level YAML section."""
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}

        return schema.model_validate(raw[section])
