"""YOLO-to-COCO conversion and official bbox evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from floating_trash.base import BaseEvaluator
from floating_trash.evaluate import Predictor
from floating_trash.schema import CocoMetrics, EvaluateConfig


class CocoEvaluator(BaseEvaluator):
    """Convert YOLO records and evaluate them with pycocotools."""

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, evaluate_config: EvaluateConfig, prediction_model: Predictor) -> None:
        """Store the evaluator dependencies."""
        self.config = evaluate_config
        self.predictor = prediction_model
        self.coco_ground_truth = COCO()

    def evaluate(self) -> CocoMetrics:
        """Run the complete COCO evaluation flow."""
        image_paths, class_names = self._read_dataset()
        dataset, image_ids = self._build_ground_truth(image_paths, class_names)
        predictions = self._build_predictions(image_paths, image_ids)

        self._prepare_ground_truth(dataset)
        evaluator = self._create_evaluator(predictions, image_ids)
        self._run_evaluator(evaluator)

        return self._extract_metrics(evaluator)

    def _read_dataset(self) -> Tuple[List[Path], Dict[int, str]]:
        """Read the image split and class names from a YOLO data file."""
        data = yaml.safe_load(self.config.data.read_text(encoding="utf-8")) or {}
        image_directory = self._resolve_image_directory(data)
        class_names = self._read_class_names(data)

        image_paths = sorted(
            path
            for path in image_directory.resolve().iterdir()
            if path.is_file() and path.suffix.lower() in self.image_suffixes
        )

        return image_paths, class_names

    def _resolve_image_directory(self, data: Dict[str, Any]) -> Path:
        """Resolve the configured test split directory."""
        split = data[self.config.split]
        split = split[0] if isinstance(split, list) else split

        configured_base = data.get("path")
        base = Path(configured_base) if configured_base else self.config.data.parent

        if configured_base and not base.is_absolute():
            base = self.config.data.parent / base

        image_directory = Path(split)
        return image_directory if image_directory.is_absolute() else base / image_directory

    def _read_class_names(self, data: Dict[str, Any]) -> Dict[int, str]:
        """Normalize list or dictionary class names."""
        names = data.get("names", {0: "trash"})

        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}

        return {int(index): str(name) for index, name in names.items()}

    def _build_ground_truth(
        self,
        image_paths: List[Path],
        class_names: Dict[int, str],
    ) -> Tuple[Dict[str, Any], Dict[Path, int]]:
        """Convert the complete YOLO test split into a COCO dataset."""
        images = []
        annotations = []
        image_ids = {}

        annotation_id = 1

        for image_id, image_path in enumerate(image_paths, start=1):
            image_record, image_annotations = self._build_image_ground_truth(image_path, image_id, annotation_id)

            images.append(image_record)
            annotations.extend(image_annotations)
            image_ids[image_path.resolve()] = image_id
            annotation_id += len(image_annotations)

        categories = self._build_categories(class_names)
        dataset = {"images": images, "annotations": annotations, "categories": categories}

        return dataset, image_ids

    def _build_image_ground_truth(
        self,
        image_path: Path,
        image_id: int,
        annotation_id: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Build the COCO image and annotation records for one image."""
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        image_record = {
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        }

        annotations = self._build_image_annotations(image_path, image_id, annotation_id, width, height)
        return image_record, annotations

    def _build_image_annotations(
        self,
        image_path: Path,
        image_id: int,
        annotation_id: int,
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        """Convert one image's YOLO labels into COCO annotations."""
        annotations = []

        for offset, fields in enumerate(self._read_detection_labels(image_path)):
            class_id, center_x, center_y, box_width, box_height = fields
            width_pixels = box_width * image_width
            height_pixels = box_height * image_height
            x = (center_x * image_width) - width_pixels / 2.0
            y = (center_y * image_height) - height_pixels / 2.0

            annotations.append(
                {
                    "id": annotation_id + offset,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [x, y, width_pixels, height_pixels],
                    "area": width_pixels * height_pixels,
                    "iscrowd": 0,
                }
            )

        return annotations

    def _read_detection_labels(self, image_path: Path) -> List[List[float]]:
        """Read the YOLO detection labels matching one image."""
        parts = list(image_path.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        label_path = Path(*parts).with_suffix(".txt")

        if not label_path.is_file():
            return []

        lines = label_path.read_text(encoding="utf-8").splitlines()
        return [list(map(float, line.split())) for line in lines if line]

    def _build_categories(self, class_names: Dict[int, str]) -> List[Dict[str, Any]]:
        """Convert YOLO class names into COCO categories."""
        return [
            {"id": index + 1, "name": name, "supercategory": "object"}
            for index, name in sorted(class_names.items())
        ]

    def _build_predictions(
        self,
        image_paths: List[Path],
        image_ids: Dict[Path, int],
    ) -> List[Dict[str, Any]]:
        """Convert all YOLO prediction results into COCO rows."""
        predictions = []

        for result in self.predictor.predict(image_paths):
            predictions.extend(self._convert_prediction(result, image_ids))

        return predictions

    def _convert_prediction(self, result: Any, image_ids: Dict[Path, int]) -> List[Dict[str, Any]]:
        """Convert one YOLO prediction result into COCO rows."""
        rows = []

        if result.boxes is None:
            return rows

        image_id = image_ids[Path(result.path).resolve()]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()

        for box, class_id, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(float, box)

            rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )

        return rows

    def _prepare_ground_truth(self, dataset: Dict[str, Any]) -> None:
        """Register the converted dataset with pycocotools."""
        self.coco_ground_truth.dataset = dataset
        self.coco_ground_truth.createIndex()

    def _create_evaluator(self, predictions: List[Dict[str, Any]], image_ids: Dict[Path, int]) -> COCOeval:
        """Create the COCO bbox evaluator for the converted predictions."""
        coco_predictions = self.coco_ground_truth.loadRes(predictions)
        evaluator = COCOeval(self.coco_ground_truth, coco_predictions, "bbox")
        evaluator.params.imgIds = sorted(image_ids.values())
        evaluator.params.maxDets = [1, 10, 100]

        return evaluator

    def _run_evaluator(self, evaluator: COCOeval) -> None:
        """Execute the COCO matching and accumulation stages."""
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    def _extract_metrics(self, evaluator: COCOeval) -> CocoMetrics:
        """Select the COCO metrics reported by the research refactor."""
        return CocoMetrics(
            ap50_95=float(evaluator.stats[0]),
            ap50=float(evaluator.stats[1]),
            ap75=float(evaluator.stats[2]),
            ap_small=float(evaluator.stats[3]),
        )
