"""Read ordered images or a video into a frame sequence."""

from pathlib import Path
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict


class FrameBatch(BaseModel):
    """Store frames, source paths, and image dimensions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    frames: List[np.ndarray]
    paths: List[Path]
    width: int
    height: int


class FrameReader:
    """Load frames in chronological filename or video order."""

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, source: Path) -> None:
        """Store the frame source."""
        self.source = source

    def read(self) -> FrameBatch:
        """Read the supported source into an ordered batch."""
        if self.source.is_dir():
            return self._read_images()

        return self._read_video()

    def _read_images(self) -> FrameBatch:
        """Read a filename-sorted image directory."""
        paths = sorted(
            path for path in self.source.iterdir()
            if path.is_file() and path.suffix.lower() in self.image_suffixes
        )
        frames = [cv2.imread(str(path)) for path in paths]

        height, width = frames[0].shape[:2]

        return FrameBatch(name=self.source.parent.name, frames=frames, paths=paths, width=width, height=height)

    def _read_video(self) -> FrameBatch:
        """Read all frames from a video file."""
        capture = cv2.VideoCapture(str(self.source))
        frames = []

        while True:
            ok, frame = capture.read()

            if not ok:
                break

            frames.append(frame)

        capture.release()
        height, width = frames[0].shape[:2]

        return FrameBatch(name=self.source.stem, frames=frames, paths=[], width=width, height=height)
