from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-not-found]
from PIL import Image

from app.core.config import PRODUCT_TYPE_ALIASES, get_image_size_fallback, get_execution_providers
from app.utils.image import nms, preprocess

from app.models import ImageEvidence, InferResponse, ProductType, RawDetection


logger = logging.getLogger(__name__)

class YoloDetector:
    def __init__(self, model_path: Path, labels: tuple[str, ...], score_threshold: float) -> None:
        if not model_path.exists():
            raise RuntimeError(f"YOLO model not found at {model_path}")

        try:
            providers = get_execution_providers()
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            logger.info("Loaded YOLO model with providers: %s", self._session.get_providers())
            logger.debug("Model inputs: %s", [i.name for i in self._session.get_inputs()])
            logger.debug("Model outputs: %s", [o.name for o in self._session.get_outputs()])
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {exc}") from exc

        model_inputs = self._session.get_inputs()
        if len(model_inputs) != 1:
            raise RuntimeError("YOLO model must expose exactly one input tensor")

        input_tensor = model_inputs[0]
        self.input_name = input_tensor.name
        self.input_height, self.input_width = self._resolve_input_size(input_tensor.shape)
        self.labels = labels
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.active_providers = self._session.get_providers()

    @staticmethod
    def _normalize_label(label: str) -> str:
        return "_".join(label.strip().lower().replace("-", " ").split())

    def _label_to_product_type(self, label: str) -> ProductType | None:
        normalized = self._normalize_label(label)
        mapped = PRODUCT_TYPE_ALIASES.get(normalized)
        if mapped is None:
            return None
        try:
            return ProductType(mapped)
        except ValueError:
            return None

    def _resolve_detected_type(self, detections: list[RawDetection]) -> ProductType:
        scores_by_type: dict[ProductType, float] = {}
        for det in detections:
            class_id = int(det.class_id)
            if class_id < 0 or class_id >= len(self.labels):
                continue
            product_type = self._label_to_product_type(self.labels[class_id])
            if product_type is None:
                continue
            score = float(det.score)
            scores_by_type[product_type] = max(scores_by_type.get(product_type, 0.0), score)

        if scores_by_type:
            return max(scores_by_type, key=scores_by_type.get)
        return ProductType.CABINET

    def _resolve_input_size(self, shape: list[int | str | None]) -> tuple[int, int]:
        if len(shape) != 4:
            raise RuntimeError(f"Unexpected YOLO input tensor rank: {len(shape)}")

        height = shape[2]
        width = shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return height, width

        fallback_size = get_image_size_fallback()
        return fallback_size, fallback_size

    def analyze(self, image_data: list[tuple[Image.Image, str]]) -> InferResponse:
        from app.services.processor import assemble_project
        
        evidence_list: list[ImageEvidence] = []
        for image, url in image_data:
            orig_w, orig_h = image.size
            tensor = preprocess(image, self.input_width, self.input_height)
            outputs = self._session.run(None, {self.input_name: tensor})
            
            # Extract raw detections
            raw_detections = self._extract_detections(outputs[0], len(self.labels))
            
            # Post-process: scale xyxy boxes from model-space to image-space.
            scaled_detections: list[RawDetection] = []
            x_scale = orig_w / self.input_width
            y_scale = orig_h / self.input_height
            
            for d in raw_detections:
                x1, y1, x2, y2 = d.box
                x1 *= x_scale
                y1 *= y_scale
                x2 *= x_scale
                y2 *= y_scale
                
                scaled_detections.append(
                    RawDetection(
                        box=(
                            round(x1, 1),
                            round(y1, 1),
                            round(x2, 1),
                            round(y2, 1),
                        ),
                        score=d.score,
                        class_id=d.class_id,
                        label=self.labels[d.class_id] if d.class_id < len(self.labels) else "unknown",
                        image_width_px=orig_w,
                        image_height_px=orig_h,
                    )
                )

            # Resolved confidence and type for this image
            if scaled_detections:
                best_det = max(scaled_detections, key=lambda x: x.score)
                confidence = best_det.score
                detected_type = self._resolve_detected_type(scaled_detections)
            else:
                confidence = 0.0
                detected_type = ProductType.CABINET

            evidence_list.append(
                ImageEvidence(
                    image_url=url,
                    width_px=orig_w,
                    height_px=orig_h,
                    detected_type=detected_type,
                    confidence=confidence,
                    raw_detections=scaled_detections,
                )
            )

        return assemble_project(evidence_list, self.labels, self.score_threshold)

    def _extract_detections(self, output: np.ndarray, num_labels: int) -> list[RawDetection]:
        tensor = np.array(output)

        logger.debug("Raw tensor shape before processing: %s", tensor.shape)
        
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]

        # YOLO exports are often [1, 4+nc, 8400] or [1, 5+nc, 8400].
        # Normalize to [8400, channels].
        if tensor.ndim == 2 and tensor.shape[0] < tensor.shape[1] and tensor.shape[0] in (4 + num_labels, 5 + num_labels):
            logger.debug("Transposing tensor from %s to %s", tensor.shape, tensor.T.shape)
            tensor = tensor.T

        if tensor.ndim != 2:
            logger.warning("Unexpected tensor rank: %s", tensor.ndim)
            return []

        _, cols = tensor.shape
        detections: list[RawDetection] = []
        
        logger.debug("Final tensor shape: %s, labels expected: %s", tensor.shape, num_labels)

        # Case 1: Standard [x1, y1, x2, y2, score, class]
        if cols == 6:
            for row in tensor:
                score = float(row[4])
                if score >= self.score_threshold:
                    detections.append(
                        RawDetection(
                            box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                            score=score,
                            class_id=int(row[5]),
                        )
                    )
            return detections

        # Case 2: YOLO format [cx, cy, w, h, class_scores...] or
        # [cx, cy, w, h, objectness, class_scores...]
        max_seen = 0.0
        effective_threshold = self.score_threshold
        has_objectness = cols >= 5 + num_labels
        class_score_start = 5 if has_objectness else 4
        available_labels = max(0, cols - class_score_start)
        label_count = min(num_labels, available_labels)
        if label_count == 0:
            logger.warning("No class scores available in tensor with shape %s", tensor.shape)
            return []
        
        for row in tensor:
            if len(row) < class_score_start + label_count:
                continue

            class_scores = row[class_score_start : class_score_start + label_count]
            if class_scores.size == 0:
                continue
                
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            obj_conf = float(row[4]) if has_objectness else 1.0
            score = obj_conf * class_conf
            
            if score > max_seen:
                max_seen = score
            
            if score >= effective_threshold:
                cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                detections.append(
                    RawDetection(
                        box=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                        score=score,
                        class_id=class_id,
                    )
                )

        # Post-process with NMS
        if detections:
            boxes = np.array([d.box for d in detections])
            scores = np.array([d.score for d in detections])
            
            indices_to_keep = nms(boxes, scores, iou_threshold=0.5)
            
            logger.debug("NMS kept %s of %s boxes", len(indices_to_keep), len(detections))
            detections = [detections[i] for i in indices_to_keep]

        logger.debug("Max confidence score observed: %.4f", max_seen)
        if len(detections) == 0:
            logger.debug("No boxes found with threshold %.4f", effective_threshold)
        else:
            logger.debug("Found %s boxes with threshold %.4f", len(detections), effective_threshold)
        return detections
