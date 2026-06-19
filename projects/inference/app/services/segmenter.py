from __future__ import annotations


import logging

from PIL import Image
import numpy as np

from app.schemas import SegmentationBackend

logger = logging.getLogger(__name__)

class Segmenter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._backend = SegmentationBackend.BOX_RASTERIZER
        logger.info("Segmenter initialized with backend=%s model=%s", self._backend, model_path)

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

    def predict(self, image: Image.Image, bboxes: list[tuple[float, float, float, float]]) -> np.ndarray:
        """Generate instance masks from bounding boxes.

        When SAM2 backend is not configured, this method returns deterministic
        rectangular masks derived from the input boxes.
        """
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

