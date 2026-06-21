from __future__ import annotations


import json
import logging
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
import warnings

from PIL import Image, ImageDraw
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
        self._sam2_model: Any | None = None
        self._sam2_predictor: Any | None = None
        self._sam2_init_attempted = False
        self._debug_dir = self._resolve_debug_dir()
        logger.info(
            "Segmenter initialized with requested_backend=%s model=%s",
            self._requested_backend,
            self.model_path,
        )

    @staticmethod
    def _resolve_debug_dir() -> Path:
        configured = os.getenv("INFERENCE_SAM_DEBUG_DIR", "").strip()
        if configured:
            return Path(configured)
        return Path(__file__).resolve().parents[2] / "debug" / "sam2"

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

    @staticmethod
    def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    @staticmethod
    def _box_overlap_min_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = max(min(area_a, area_b), 1e-6)
        return inter / denom

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

    def _write_debug_artifacts(
        self,
        image: Image.Image,
        bboxes: list[tuple[float, float, float, float]],
        masks: np.ndarray,
    ) -> None:
        if self._requested_backend != SegmentationBackend.SAM2:
            return

        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            base = self._debug_dir / f"sam_debug_{stamp}"

            np.save(base.with_suffix(".npy"), masks.astype(np.uint8))

            mask_pixels = [int(mask.sum()) for mask in masks] if len(masks) else []
            metadata = {
                "requested_backend": self._requested_backend.value,
                "active_backend": self._active_backend.value,
                "model_path": str(self.model_path) if self.model_path else None,
                "image_width": int(image.width),
                "image_height": int(image.height),
                "boxes": [[float(v) for v in box] for box in bboxes],
                "mask_pixels": mask_pixels,
                "masks_file": base.with_suffix(".npy").name,
            }
            base.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            preview = image.convert("RGB").copy()
            draw = ImageDraw.Draw(preview)
            for box in bboxes:
                draw.rectangle(box, outline=(255, 80, 80), width=2)
            preview.save(base.with_suffix(".png"))

            if len(masks):
                union = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
                Image.fromarray(union, mode="L").save(base.with_name(base.name + "_union_mask").with_suffix(".png"))
        except Exception as exc:  # pragma: no cover - debug-only path
            logger.warning("Failed to write SAM debug artifacts: %s", exc)

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
            import torch

            checkpoint_name = self.model_path.name.lower()
            variant = "t"
            if any(token in checkpoint_name for token in ("b+", "bplus", "base_plus")):
                variant = "b+"
            elif "large" in checkpoint_name or "_l" in checkpoint_name:
                variant = "l"
            elif "small" in checkpoint_name or "_s" in checkpoint_name:
                variant = "s"

            config_candidates = [
                f"sam2_hiera_{variant}.yaml",
                f"configs/sam2/sam2_hiera_{variant}.yaml",
                f"configs/sam2.1/sam2.1_hiera_{variant}.yaml",
            ]
            device = "cuda" if torch.cuda.is_available() else "cpu"

            sam2_model: Any | None = None
            last_error: Exception | None = None
            for config_name in config_candidates:
                try:
                    sam2_model = build_sam2(
                        config_file=config_name,
                        ckpt_path=str(self.model_path),
                        device=device,
                    )
                    logger.info(
                        "SAM2 model loaded with config=%s checkpoint=%s device=%s",
                        config_name,
                        self.model_path,
                        device,
                    )
                    break
                except Exception as exc:
                    last_error = exc

            if sam2_model is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Unable to resolve a SAM2 configuration for checkpoint")

            predictor = SAM2ImagePredictor(sam2_model)
            self._sam2_model = sam2_model
            self._sam2_predictor = predictor
            logger.info("SAM2 predictor loaded from %s", self.model_path)
            return predictor
        except Exception as exc:  # pragma: no cover - runtime env dependent
            logger.warning("Failed to initialize SAM2 predictor: %s", exc)
            return None

    def propose_boxes(
        self,
        image: Image.Image,
        *,
        min_area_ratio: float = 0.01,
        max_boxes: int = 12,
    ) -> list[tuple[float, float, float, float]]:
        regions = self.propose_regions(
            image,
            min_area_ratio=min_area_ratio,
            max_regions=max_boxes,
        )
        return [region["box"] for region in regions]

    def propose_regions(
        self,
        image: Image.Image,
        *,
        min_area_ratio: float = 0.01,
        max_regions: int = 16,
    ) -> list[dict[str, float | tuple[float, float, float, float]]]:
        """Generate SAM2 region proposals with box/area/quality metadata."""
        if self._requested_backend != SegmentationBackend.SAM2:
            return []

        predictor = self._ensure_sam2_predictor()
        if predictor is None or self._sam2_model is None:
            return []

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore[import-not-found]

            image_np = np.array(image.convert("RGB"))
            image_area = max(1, image_np.shape[0] * image_np.shape[1])
            min_area_px = int(image_area * max(0.0, min_area_ratio))

            generator = SAM2AutomaticMaskGenerator(
                model=self._sam2_model,
                output_mode="binary_mask",
                points_per_side=24,
                points_per_batch=64,
                pred_iou_thresh=0.75,
                stability_score_thresh=0.92,
                box_nms_thresh=0.65,
                min_mask_region_area=max(min_area_px, 256),
            )

            entries = generator.generate(image_np)
            if not isinstance(entries, list) or not entries:
                return []

            filtered = [
                item for item in entries
                if isinstance(item, dict)
                and isinstance(item.get("bbox"), (list, tuple))
                and len(item.get("bbox")) == 4
                and float(item.get("area", 0.0)) >= float(min_area_px)
            ]
            candidates: list[dict[str, float | tuple[float, float, float, float]]] = []
            for item in filtered:
                x, y, w, h = item["bbox"]
                x1 = float(x)
                y1 = float(y)
                x2 = float(x) + float(w)
                y2 = float(y) + float(h)
                clipped = self._clip_box((x1, y1, x2, y2), image.width, image.height)
                if clipped is None:
                    continue
                area = float(item.get("area", 0.0))
                area_ratio = area / max(float(image_area), 1.0)
                if area_ratio > 0.92:
                    # Skip near full-frame masks because they usually represent background silhouette.
                    continue

                # Reject frame-hugging boxes that cover almost the whole image extent.
                touches_left = clipped[0] <= 1
                touches_top = clipped[1] <= 1
                touches_right = clipped[2] >= image.width - 1
                touches_bottom = clipped[3] >= image.height - 1
                touch_count = int(touches_left) + int(touches_top) + int(touches_right) + int(touches_bottom)
                bbox_cover_ratio = (
                    float(clipped[2] - clipped[0]) * float(clipped[3] - clipped[1])
                ) / max(float(image_area), 1.0)
                if touch_count >= 3 and bbox_cover_ratio > 0.80:
                    continue

                box_w = float(clipped[2] - clipped[0]) / max(float(image.width), 1.0)
                box_h = float(clipped[3] - clipped[1]) / max(float(image.height), 1.0)
                if box_w < 0.03 or box_h < 0.03:
                    continue

                candidates.append(
                    {
                        "box": (float(clipped[0]), float(clipped[1]), float(clipped[2]), float(clipped[3])),
                        "area": area,
                        "predicted_iou": float(item.get("predicted_iou", 0.0)),
                        "stability_score": float(item.get("stability_score", 0.0)),
                        "quality": float(item.get("predicted_iou", 0.0)) * 0.6
                        + float(item.get("stability_score", 0.0)) * 0.35
                        + area_ratio * 0.05,
                    }
                )

            candidates.sort(key=lambda item: float(item.get("quality", 0.0)), reverse=True)

            proposals: list[dict[str, float | tuple[float, float, float, float]]] = []
            for candidate in candidates:
                candidate_box = candidate["box"]
                if not isinstance(candidate_box, tuple):
                    continue
                if any(
                    (
                        self._box_iou(candidate_box, existing["box"]) > 0.72
                        or self._box_overlap_min_area(candidate_box, existing["box"]) > 0.86
                    )
                    for existing in proposals
                    if isinstance(existing.get("box"), tuple)
                ):
                    continue
                proposals.append(candidate)
                if len(proposals) >= max(1, max_regions):
                    break

            return proposals
        except Exception as exc:  # pragma: no cover - runtime env dependent
            logger.warning("SAM2 automatic proposal generation failed: %s", exc)
            return []

    def _predict_sam2(
        self,
        image: Image.Image,
        bboxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        predictor = self._ensure_sam2_predictor()
        if predictor is None:
            self._active_backend = SegmentationBackend.BOX_RASTERIZER
            masks = self._predict_box_rasterizer(image, bboxes)
            self._write_debug_artifacts(image, bboxes, masks)
            return masks

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
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"cannot import name '_C' from 'sam2'.*",
                        category=UserWarning,
                    )
                    raw_masks, _, _ = predictor.predict(
                        box=box_np,
                        multimask_output=False,
                    )
            except TypeError:
                # Some SAM2 builds require box as batch dimension.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"cannot import name '_C' from 'sam2'.*",
                        category=UserWarning,
                    )
                    raw_masks, _, _ = predictor.predict(
                        box=box_np[None, :],
                        multimask_output=False,
                    )

            if isinstance(raw_masks, np.ndarray) and raw_masks.size:
                candidate = raw_masks[0] if raw_masks.ndim == 3 else raw_masks
                masks[idx] = candidate.astype(bool)

        self._active_backend = SegmentationBackend.SAM2
        self._write_debug_artifacts(image, bboxes, masks)
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

