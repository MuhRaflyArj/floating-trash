"""Read per-frame identity-aware YOLO labels."""

from pathlib import Path
from typing import List

from floating_trash.schema import TrackingSequence, TrackObservation
from floating_trash.utils import FrameBatch


class TrackingLabelReader:
    """Convert normalized tracking labels into pixel observations."""

    def __init__(self, label_directory: Path) -> None:
        """Store the tracking-label directory."""
        self.label_directory = label_directory

    def read(self, batch: FrameBatch) -> TrackingSequence:
        """Read labels that correspond to the ordered frame sequence."""
        label_paths = self._label_paths(batch)
        observations = []

        for frame_index, label_path in enumerate(label_paths):
            if not label_path.is_file():
                continue

            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue

                track_id, class_id, center_x, center_y, width, height = map(float, line.split())
                width_pixels = width * batch.width
                height_pixels = height * batch.height

                observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        track_id=int(track_id),
                        class_id=int(class_id),
                        x=(center_x * batch.width) - width_pixels / 2.0,
                        y=(center_y * batch.height) - height_pixels / 2.0,
                        width=width_pixels,
                        height=height_pixels,
                    )
                )

        return TrackingSequence(
            name=batch.name,
            frame_width=batch.width,
            frame_height=batch.height,
            frame_count=len(batch.frames),
            observations=observations,
        )

    def _label_paths(self, batch: FrameBatch) -> List[Path]:
        """Match label filenames to image stems or video order."""
        if batch.paths:
            return [self.label_directory / f"{path.stem}.txt" for path in batch.paths]
        return sorted(self.label_directory.glob("*.txt"))[: len(batch.frames)]
