from __future__ import annotations


import logging
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from app.models import SegmentationBackend

logger = logging.getLogger(__name__)


class Segmenter:
    def __init__(
        self,
        model_path: str | Path | None,
        *,
        backend: SegmentationBackend,
    ):
        self.model_path = Path(model_path) if model_path else None
        self._requested_backend = backend
        self._active_backend = SegmentationBackend.BOX_RASTERIZER
        self._sam2_predictor: Any | None = None
        self._sam2_init_attempted = False
        logger.info(
            "Segmenter initialized with requested_backend=%s model=%s",
            self._requested_backend,
            self.model_path,
        )

    @property
    def active_backend(self) -> SegmentationBackend:
        return self._active_backend

    @staticmethod
    def _clip_box(
        box: tuple[float, float, float, float], width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        x1_i = max(0, min(width - 1, int(round(x1))))
        y1_i = max(0, min(height - 1, int(round(y1))))
        x2_i = max(0, min(width, int(round(x2))))
        y2_i = max(0, min(height, int(round(y2))))
        if x2_i <= x1_i or y2_i <= y1_i:
            return None
        return x1_i, y1_i, x2_i, y2_i

    def _predict_box_rasterizer(
        self,
        image: Image.Image,
        bboxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        masks = np.zeros((len(bboxes), h, w), dtype=bool)

        for idx, box in enumerate(bboxes):
            clipped = self._clip_box(box, w, h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            masks[idx, y1:y2, x1:x2] = True

        return masks

    def _ensure_sam2_predictor(self) -> Any | None:
        if self._sam2_init_attempted:
            return self._sam2_predictor

        self._sam2_init_attempted = True
        if self.model_path is None:
            logger.warning("SAM2 backend requested but INFERENCE_SAM2_MODEL_PATH is not set")
            return None

        if not self.model_path.exists():
            logger.warning("SAM2 backend requested but model file does not exist at %s", self.model_path)
            return None

        try:
            # Imported lazily so service can run without SAM2 runtime dependencies.
            from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-not-found]

            predictor = SAM2ImagePredictor(build_sam2(str(self.model_path)))
            self._sam2_predictor = predictor
            logger.info("SAM2 predictor loaded from %s", self.model_path)
            return predictor
        except Exception as exc:  # pragma: no cover - runtime env dependent
            logger.warning("Failed to initialize SAM2 predictor: %s", exc)
            return None

    def _predict_sam2(
        self,
        image: Image.Image,
        bboxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        predictor = self._ensure_sam2_predictor()
        if predictor is None:
            self._active_backend = SegmentationBackend.BOX_RASTERIZER
            return self._predict_box_rasterizer(image, bboxes)

        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        masks = np.zeros((len(bboxes), h, w), dtype=bool)
        predictor.set_image(image_np)

        for idx, box in enumerate(bboxes):
            clipped = self._clip_box(box, w, h)
            if clipped is None:
                continue

            x1, y1, x2, y2 = clipped
            box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
            try:
                raw_masks, _, _ = predictor.predict(
                    box=box_np,
                    multimask_output=False,
                )
            except TypeError:
                # Some SAM2 builds require box as batch dimension.
                raw_masks, _, _ = predictor.predict(
                    box=box_np[None, :],
                    multimask_output=False,
                )

            if isinstance(raw_masks, np.ndarray) and raw_masks.size:
                candidate = raw_masks[0] if raw_masks.ndim == 3 else raw_masks
                masks[idx] = candidate.astype(bool)

        self._active_backend = SegmentationBackend.SAM2
        return masks

    def predict(self, image: Image.Image, bboxes: list[tuple[float, float, float, float]]) -> np.ndarray:
        """Generate instance masks from bounding boxes.

        Returns SAM2 masks when available, otherwise falls back to deterministic
        rectangular masks derived from input boxes.
        """
        if self._requested_backend == SegmentationBackend.SAM2:
            return self._predict_sam2(image, bboxes)

        self._active_backend = SegmentationBackend.BOX_RASTERIZER
        return self._predict_box_rasterizer(image, bboxes)

