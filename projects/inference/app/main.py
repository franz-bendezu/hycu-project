from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
import math
import os
import random
from statistics import mean

from fastapi import FastAPI, HTTPException
import numpy as np
from PIL import Image

from app.core.config import (
    get_detector_model_path,
    get_detector_labels,
    get_confidence_threshold,
    get_sam2_model_path,
    get_segmentation_backend,
)
from app.models import (
    BenchmarkItemSummary,
    BenchmarkRequest,
    BenchmarkResponse,
    EscalationStrategy,
    InferRequest,
    InferResponse,
    RawDetection,
)
from app.services.detector import YoloDetector
from app.services.segmenter import Segmenter
from app.services.refiner import HeavyRefiner
from app.services.tracker import MultiViewTracker
from app.utils.image import load_image_bytes, open_image


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast during service boot when artifacts are missing or invalid.
    _detector()
    _segmenter()
    yield


app = FastAPI(title="Vision Inference Service", version="0.1.0", lifespan=lifespan)


@lru_cache(maxsize=1)
def _detector() -> YoloDetector:
    from app.services.processor import assemble_project

    return YoloDetector(
        model_path=get_detector_model_path(),
        labels=get_detector_labels(),
        score_threshold=get_confidence_threshold(),
        assemble_project=assemble_project,
    )


@lru_cache(maxsize=1)
def _segmenter() -> Segmenter:
    return Segmenter(
        model_path=get_sam2_model_path(),
        backend=get_segmentation_backend(),
    )


@lru_cache(maxsize=1)
def _refiner() -> HeavyRefiner:
    return HeavyRefiner()


def load_all_images(image_urls: list[str]) -> list[tuple[Image.Image, str]]:
    results = []
    for url in image_urls:
        try:
            image_bytes = load_image_bytes(url)
            image = open_image(image_bytes)
            results.append((image, url))
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not load image from {url}: {str(exc)}"
            )
    return results


def _first_label_index(labels: tuple[str, ...], candidates: list[str], default: int = 0) -> int:
    for name in candidates:
        try:
            return labels.index(name)
        except ValueError:
            continue
    return default


def _sam2_fallback_label_index(
    box: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    labels: tuple[str, ...],
    detected_type: str,
) -> int:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    nx1 = x1 / max(float(image_width), 1.0)
    ny1 = y1 / max(float(image_height), 1.0)
    nx2 = x2 / max(float(image_width), 1.0)
    ny2 = y2 / max(float(image_height), 1.0)
    nwidth = max(nx2 - nx1, 0.0)
    nheight = max(ny2 - ny1, 0.0)
    x_center = (nx1 + nx2) * 0.5
    aspect = width / max(height, 1.0)
    type_name = str(detected_type).strip().lower()

    # Broad product frame proposals.
    if nwidth > 0.72 and nheight > 0.60:
        return _first_label_index(labels, ["cabinet_body", "desk_frame", "shelf", "cabinet"]) 

    # Horizontal slabs are often top/bottom/shelf panels.
    if aspect >= 1.9 and nheight <= 0.26:
        if ny1 <= 0.18:
            return _first_label_index(labels, ["top_panel", "desk_frame", "cabinet_body"])
        if ny2 >= 0.84:
            return _first_label_index(labels, ["bottom_panel", "cabinet_body", "desk_frame"])
        return _first_label_index(labels, ["shelf_panel", "top_panel", "bottom_panel"])

    # Tall narrow side strips.
    if nwidth <= 0.24 and nheight >= 0.45 and (x_center <= 0.22 or x_center >= 0.78):
        return _first_label_index(labels, ["side_panel", "cabinet_body"])

    # Front-like components (doors/drawers) near cabinet center.
    if 0.18 <= nwidth <= 0.68 and nheight >= 0.30:
        if type_name in {"desk", "table"}:
            return _first_label_index(labels, ["front_apron", "drawer_front", "desk_frame"])
        if ny2 >= 0.62 and nheight <= 0.34:
            return _first_label_index(labels, ["drawer_front", "door_panel", "cabinet_body"])
        return _first_label_index(labels, ["door_panel", "drawer_front", "cabinet_body"])

    # Generic fallback.
    return _first_label_index(labels, ["cabinet_body", "desk_frame", "shelf", "cabinet"])


def _sam2_regions_to_detections(
    *,
    regions: list[dict[str, float | tuple[float, float, float, float]]],
    image: Image.Image,
    labels: tuple[str, ...],
    detected_type: str,
    view_index: int,
) -> list[RawDetection]:
    if not regions:
        return []

    def _to_box(region: dict[str, float | tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
        box_value = region.get("box")
        if not isinstance(box_value, tuple) or len(box_value) != 4:
            return None
        return (
            float(box_value[0]),
            float(box_value[1]),
            float(box_value[2]),
            float(box_value[3]),
        )

    def _augment_regions_with_drawer_splits(
        base_regions: list[dict[str, float | tuple[float, float, float, float]]],
    ) -> list[dict[str, float | tuple[float, float, float, float]]]:
        image_np = np.array(image.convert("L"), dtype=np.float32)
        h_img, w_img = image_np.shape[:2]
        augmented = list(base_regions)

        for region in base_regions:
            box = _to_box(region)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            nw = bw / max(float(w_img), 1.0)
            nh = bh / max(float(h_img), 1.0)
            aspect = bw / max(bh, 1.0)

            # Target drawer-stack like front faces: tall and moderately narrow regions.
            if not (0.18 <= nw <= 0.38 and nh >= 0.45 and aspect <= 0.7):
                continue

            ix1 = max(0, min(w_img - 2, int(round(x1))))
            ix2 = max(ix1 + 2, min(w_img, int(round(x2))))
            iy1 = max(0, min(h_img - 2, int(round(y1))))
            iy2 = max(iy1 + 2, min(h_img, int(round(y2))))
            crop = image_np[iy1:iy2, ix1:ix2]
            if crop.shape[0] < 30 or crop.shape[1] < 20:
                continue

            # Horizontal front seams create strong row-wise gradients.
            grad_y = np.abs(np.diff(crop, axis=0))
            row_energy = grad_y.mean(axis=1)
            if row_energy.size < 8:
                continue

            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            kernel = kernel / kernel.sum()
            smooth = np.convolve(row_energy, kernel, mode="same")
            threshold = float(smooth.mean() + 0.75 * smooth.std())

            min_gap = max(16, int(0.08 * crop.shape[0]))
            peaks: list[int] = []
            for idx in range(2, len(smooth) - 2):
                val = smooth[idx]
                if val < threshold:
                    continue
                if not (val >= smooth[idx - 1] and val >= smooth[idx + 1]):
                    continue
                if idx < int(0.1 * crop.shape[0]) or idx > int(0.9 * crop.shape[0]):
                    continue
                if peaks and abs(idx - peaks[-1]) < min_gap:
                    if val > smooth[peaks[-1]]:
                        peaks[-1] = idx
                    continue
                peaks.append(idx)

            if not peaks:
                continue

            cuts = [0, *sorted(peaks), crop.shape[0] - 1]
            source_quality = float(region.get("quality", 0.6))
            source_iou = float(region.get("predicted_iou", 0.6))
            source_stability = float(region.get("stability_score", 0.7))
            source_area = float(region.get("area", bw * bh))

            for cidx in range(len(cuts) - 1):
                sy1 = cuts[cidx]
                sy2 = cuts[cidx + 1]
                seg_h = sy2 - sy1
                if seg_h < max(18, int(0.08 * crop.shape[0])):
                    continue

                new_y1 = float(iy1 + sy1)
                new_y2 = float(iy1 + sy2)
                if new_y2 <= new_y1 + 4:
                    continue

                seg_area = (new_y2 - new_y1) * bw
                augmented.append(
                    {
                        "box": (x1, new_y1, x2, new_y2),
                        "area": max(1.0, min(source_area, seg_area)),
                        "predicted_iou": min(1.0, source_iou * 0.95),
                        "stability_score": min(1.0, source_stability * 0.95),
                        "quality": max(0.2, source_quality * 0.9),
                        "label_hint": "drawer_front",
                    }
                )

        return augmented

    def _augment_regions_with_top_panel(
        base_regions: list[dict[str, float | tuple[float, float, float, float]]],
    ) -> list[dict[str, float | tuple[float, float, float, float]]]:
        type_name = str(detected_type).strip().lower()
        if type_name not in {"desk", "table"}:
            return base_regions

        boxes = [b for b in (_to_box(region) for region in base_regions) if b is not None]
        if len(boxes) < 2:
            return base_regions

        h_img = max(float(image.height), 1.0)
        w_img = max(float(image.width), 1.0)

        tall_side_like = []
        for box in boxes:
            x1, y1, x2, y2 = box
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            if (bh / h_img) >= 0.45 and (bw / w_img) <= 0.42:
                tall_side_like.append(box)

        if len(tall_side_like) < 2:
            return base_regions

        left = min(tall_side_like, key=lambda box: box[0])
        right = max(tall_side_like, key=lambda box: box[2])
        y_top = min(left[1], right[1])
        span_w = max(right[2] - left[0], 1.0)
        panel_h = max(24.0, 0.09 * h_img)

        panel = (
            max(0.0, left[0] - 0.02 * span_w),
            max(0.0, y_top - 0.05 * h_img),
            min(w_img - 1.0, right[2] + 0.02 * span_w),
            min(h_img - 1.0, y_top + panel_h),
        )

        if panel[2] <= panel[0] + 6 or panel[3] <= panel[1] + 6:
            return base_regions

        augmented = list(base_regions)
        augmented.append(
            {
                "box": panel,
                "area": max(1.0, (panel[2] - panel[0]) * (panel[3] - panel[1])),
                "predicted_iou": 0.72,
                "stability_score": 0.72,
                "quality": 0.7,
                "label_hint": "top_panel",
            }
        )
        return augmented

    regions = _augment_regions_with_drawer_splits(regions)
    regions = _augment_regions_with_top_panel(regions)

    boxes = [
        region.get("box")
        for region in regions
        if isinstance(region.get("box"), tuple) and len(region.get("box")) == 4
    ]
    parsed_boxes = [box for box in boxes if isinstance(box, tuple)]
    if not parsed_boxes:
        return []

    extent_x1 = min(float(box[0]) for box in parsed_boxes)
    extent_y1 = min(float(box[1]) for box in parsed_boxes)
    extent_x2 = max(float(box[2]) for box in parsed_boxes)
    extent_y2 = max(float(box[3]) for box in parsed_boxes)
    extent_w = max(extent_x2 - extent_x1, 1.0)
    extent_h = max(extent_y2 - extent_y1, 1.0)

    image_area = max(float(image.width * image.height), 1.0)

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

    def _overlap_min_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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
        return inter / max(min(area_a, area_b), 1e-6)

    def _label_candidates(
        box: tuple[float, float, float, float],
        label_hint: str | None,
    ) -> list[tuple[str, float]]:
        if label_hint == "drawer_front":
            return [("drawer_front", 1.0), ("door_panel", 0.45)]
        if label_hint == "top_panel":
            return [("top_panel", 1.0), ("shelf_panel", 0.4)]

        x1, y1, x2, y2 = box
        rel_x1 = (x1 - extent_x1) / extent_w
        rel_x2 = (x2 - extent_x1) / extent_w
        rel_y1 = (y1 - extent_y1) / extent_h
        rel_y2 = (y2 - extent_y1) / extent_h
        rel_w = max(rel_x2 - rel_x1, 0.0)
        rel_h = max(rel_y2 - rel_y1, 0.0)
        x_center = (rel_x1 + rel_x2) * 0.5

        if rel_h > 0.48 and rel_w < 0.22 and (x_center < 0.2 or x_center > 0.8):
            return [("side_panel", 1.0), ("cabinet_body", 0.45)]
        if rel_w > 0.70 and rel_h < 0.24 and rel_y1 < 0.2:
            return [("top_panel", 1.0), ("cabinet_body", 0.4)]
        if rel_w > 0.70 and rel_h < 0.24 and rel_y2 > 0.8:
            return [("bottom_panel", 1.0), ("cabinet_body", 0.4)]
        if rel_w > 0.55 and rel_h < 0.20:
            return [("shelf_panel", 0.95), ("top_panel", 0.45), ("bottom_panel", 0.45)]
        if rel_w > 0.28 and rel_h > 0.30 and rel_y1 > 0.16 and rel_y2 < 0.95:
            if str(detected_type).strip().lower() in {"desk", "table"}:
                return [("front_apron", 0.85), ("drawer_front", 0.7), ("desk_frame", 0.4)]
            return [("door_panel", 0.9), ("drawer_front", 0.7), ("cabinet_body", 0.4)]
        return [("cabinet_body", 0.65), ("shelf_panel", 0.35)]

    # Build hypothesis pool for beam search.
    hypotheses_by_region: list[list[dict[str, object]]] = []
    for ridx, region in enumerate(regions):
        box_value = region.get("box")
        if not isinstance(box_value, tuple) or len(box_value) != 4:
            continue
        box = (
            float(box_value[0]),
            float(box_value[1]),
            float(box_value[2]),
            float(box_value[3]),
        )

        area_hint = float(region.get("area", 0.0)) / image_area
        iou_hint = float(region.get("predicted_iou", 0.0))
        stability_hint = float(region.get("stability_score", 0.0))
        quality_hint = float(region.get("quality", 0.0))
        if quality_hint <= 0.0:
            quality_hint = (0.6 * iou_hint) + (0.35 * stability_hint) + (0.05 * area_hint)

        depth_score = float(region.get("depth_score", 0.0))
        label_hint_value = region.get("label_hint")
        label_hint = str(label_hint_value) if isinstance(label_hint_value, str) else None
        label_candidates = _label_candidates(box, label_hint)
        region_hypotheses: list[dict[str, object]] = []
        for label_name, label_prior in label_candidates:
            class_id = _first_label_index(labels, [label_name])
            resolved = labels[class_id] if 0 <= class_id < len(labels) else label_name
            confidence = max(0.2, min(0.92, (0.25 * area_hint) + (0.4 * iou_hint) + (0.35 * stability_hint)))
            region_hypotheses.append(
                {
                    "region_idx": ridx,
                    "box": box,
                    "label": resolved,
                    "class_id": class_id,
                    "label_prior": float(label_prior),
                    "quality": float(quality_hint),
                    "depth_score": depth_score,
                    "confidence": float(confidence),
                }
            )
        hypotheses_by_region.append(region_hypotheses)

    if not hypotheses_by_region:
        return []

    depth_weight = float(os.getenv("INFERENCE_SAM_DEPTH_WEIGHT", "0.0"))
    beam_width = max(4, int(os.getenv("INFERENCE_SAM_BEAM_WIDTH", "12")))

    def _score_state(selected: list[dict[str, object]]) -> float:
        if not selected:
            return -0.2

        score = 0.0
        label_counts: dict[str, int] = {}
        for item in selected:
            label = str(item["label"])
            label_counts[label] = label_counts.get(label, 0) + 1
            score += 1.2 * float(item["quality"])
            score += 0.8 * float(item["label_prior"])
            score += 0.6 * float(item["confidence"])
            score += depth_weight * float(item.get("depth_score", 0.0))

            x1, y1, x2, y2 = item["box"]  # type: ignore[index]
            cx = ((float(x1) - extent_x1) / extent_w + (float(x2) - extent_x1) / extent_w) * 0.5
            cy = ((float(y1) - extent_y1) / extent_h + (float(y2) - extent_y1) / extent_h) * 0.5
            if label == "side_panel":
                score += 0.35 if (cx < 0.24 or cx > 0.76) else -0.65
            elif label == "top_panel":
                score += 0.35 if cy < 0.26 else -0.6
            elif label == "bottom_panel":
                score += 0.35 if cy > 0.72 else -0.6
            elif label in {"door_panel", "drawer_front"}:
                score += 0.25 if 0.18 <= cx <= 0.82 else -0.35

        for idx in range(len(selected)):
            for jdx in range(idx + 1, len(selected)):
                box_a = selected[idx]["box"]  # type: ignore[index]
                box_b = selected[jdx]["box"]  # type: ignore[index]
                iou = _box_iou(box_a, box_b)
                contain = _overlap_min_area(box_a, box_b)
                score -= 1.15 * iou
                score -= 1.35 * max(0.0, contain - 0.58)

        # Grammar penalties / rewards (panel furniture prior).
        side_count = label_counts.get("side_panel", 0)
        top_count = label_counts.get("top_panel", 0)
        bottom_count = label_counts.get("bottom_panel", 0)
        shelf_count = label_counts.get("shelf_panel", 0)
        front_count = label_counts.get("door_panel", 0) + label_counts.get("drawer_front", 0)
        body_count = label_counts.get("cabinet_body", 0)

        if side_count >= 2:
            score += 0.9
        elif side_count == 1:
            score -= 0.45
        if top_count >= 1:
            score += 0.5
        if bottom_count >= 1:
            score += 0.5
        if front_count >= 1:
            score += 0.6
        if shelf_count >= 1:
            score += 0.25

        if side_count > 2:
            score -= 0.9 * (side_count - 2)
        if top_count > 1:
            score -= 0.95 * (top_count - 1)
        if bottom_count > 1:
            score -= 0.95 * (bottom_count - 1)
        if front_count > 4:
            score -= 0.35 * (front_count - 4)
        if body_count > 1:
            score -= 1.2 * (body_count - 1)
        if body_count > 0 and len(selected) > 2:
            score -= 0.8

        return score

    # Phase 1: deterministic beam search.
    beam: list[list[dict[str, object]]] = [[]]
    for options in hypotheses_by_region[:18]:
        candidates: list[list[dict[str, object]]] = []
        for state in beam:
            candidates.append(state)  # skip region
            for hyp in options:
                region_idx = int(hyp["region_idx"])
                if any(int(item["region_idx"]) == region_idx for item in state):
                    continue
                hyp_box = hyp["box"]  # type: ignore[index]
                if any(_box_iou(hyp_box, item["box"]) > 0.78 for item in state):
                    continue
                candidates.append([*state, hyp])

        candidates.sort(key=_score_state, reverse=True)
        beam = candidates[:beam_width]

    best = beam[0] if beam else []

    # Phase 2 (optional): stochastic rjMCMC-style refinement.
    search_mode = os.getenv("INFERENCE_SAM_SEARCH_MODE", "beam").strip().lower()
    if search_mode in {"rjmcmc", "stochastic"} and best:
        rng = random.Random(7)
        current = list(best)
        current_score = _score_state(current)
        best_state = list(current)
        best_score = current_score

        flat = [hyp for group in hypotheses_by_region for hyp in group]
        for step in range(80):
            trial = list(current)
            move = rng.choice(["add", "remove", "swap"])
            temperature = max(0.08, 1.0 - (step / 90.0))

            if move == "remove" and trial:
                trial.pop(rng.randrange(len(trial)))
            elif move == "add":
                pool = [
                    hyp
                    for hyp in flat
                    if all(int(item["region_idx"]) != int(hyp["region_idx"]) for item in trial)
                ]
                if pool:
                    hyp = rng.choice(pool)
                    hyp_box = hyp["box"]  # type: ignore[index]
                    if not any(_box_iou(hyp_box, item["box"]) > 0.82 for item in trial):
                        trial.append(hyp)
            else:  # swap
                if trial:
                    trial.pop(rng.randrange(len(trial)))
                pool = [
                    hyp
                    for hyp in flat
                    if all(int(item["region_idx"]) != int(hyp["region_idx"]) for item in trial)
                ]
                if pool:
                    hyp = rng.choice(pool)
                    hyp_box = hyp["box"]  # type: ignore[index]
                    if not any(_box_iou(hyp_box, item["box"]) > 0.82 for item in trial):
                        trial.append(hyp)

            trial_score = _score_state(trial)
            delta = trial_score - current_score
            accept = delta >= 0 or rng.random() < math.exp(max(-20.0, delta / temperature))
            if accept:
                current = trial
                current_score = trial_score
                if current_score > best_score:
                    best_state = list(current)
                    best_score = current_score

        best = best_state

    # Convert best selected hypotheses to RawDetection list.
    best.sort(key=lambda item: float(item.get("quality", 0.0)), reverse=True)
    detections: list[RawDetection] = []
    for item in best:
        box = item["box"]  # type: ignore[index]
        detections.append(
            RawDetection(
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=round(float(item.get("confidence", 0.35)), 4),
                class_id=int(item.get("class_id", 0)),
                label=str(item.get("label", "sam2_auto")),
                image_width_px=image.width,
                image_height_px=image.height,
                view_index=view_index,
            )
        )
    return detections


@app.get("/health")
def health() -> dict:
    try:
        detector = _detector()
        segmenter = _segmenter()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model_path": str(detector.model_path),
        "labels": list(detector.labels),
        "confidence_threshold": detector.score_threshold,
        "active_providers": detector.active_providers,
        "segmentation_backend": segmenter.active_backend,
    }


def _infer_once(image_urls: list[str]) -> InferResponse:
    from app.services.processor import assemble_project

    images_with_urls = load_all_images(image_urls)

    # Stage 1: Object Detection
    detector = _detector()
    detection_result = detector.analyze(images_with_urls)

    # Stage 2: Multi-view tracking + segmentation
    segmenter = _segmenter()
    tracker = MultiViewTracker(iou_threshold=0.35, max_age=2)
    for idx, evidence in enumerate(detection_result.evidence):
        image, _ = images_with_urls[idx]

        normalized_detections: list[RawDetection] = []
        for detection in evidence.raw_detections:
            box = detection.box
            if isinstance(box, tuple) and len(box) == 4:
                enriched = detection.model_copy(update={"view_index": idx})
                normalized_detections.append(enriched)

        if not normalized_detections:
            sam_regions = segmenter.propose_regions(image)
            normalized_detections = _sam2_regions_to_detections(
                regions=sam_regions,
                image=image,
                labels=detector.labels,
                detected_type=evidence.detected_type,
                view_index=idx,
            )

            if not normalized_detections:
                sam_boxes = segmenter.propose_boxes(image)
                if sam_boxes:
                    for box in sam_boxes:
                        fallback_class_id = _sam2_fallback_label_index(
                            box,
                            image_width=image.width,
                            image_height=image.height,
                            labels=detector.labels,
                            detected_type=evidence.detected_type,
                        )
                        label_name = (
                            detector.labels[fallback_class_id]
                            if 0 <= fallback_class_id < len(detector.labels)
                            else "sam2_auto"
                        )
                        normalized_detections.append(
                            RawDetection(
                                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                                score=0.35,
                                class_id=fallback_class_id,
                                label=label_name,
                                image_width_px=image.width,
                                image_height_px=image.height,
                                view_index=idx,
                            )
                        )

        tracked = tracker.update(normalized_detections)
        boxes = [item.box for item in tracked]
        masks = segmenter.predict(image, boxes)

        enriched_tracked: list[RawDetection] = []
        for det_idx, item in enumerate(tracked):
            update_data: dict[str, int | float] = {}
            if item.image_width_px is None:
                update_data["image_width_px"] = image.width
            if item.image_height_px is None:
                update_data["image_height_px"] = image.height
            if det_idx < len(masks):
                mask_area = int(masks[det_idx].sum())
                update_data["mask_area_px"] = mask_area
                update_data["mask_fill_ratio"] = round(mask_area / max(image.width * image.height, 1), 5)
            enriched_tracked.append(item.model_copy(update=update_data) if update_data else item)

        evidence.raw_detections = enriched_tracked

    response = assemble_project(detection_result.evidence, detector.labels, detector.score_threshold)
    response = _refiner().maybe_refine(response, detection_result.evidence)
    return response


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    return _infer_once(payload.image_urls)


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(payload: BenchmarkRequest) -> BenchmarkResponse:
    item_results: list[BenchmarkItemSummary] = []
    strategy_counts: Counter[str] = Counter()

    for idx, item in enumerate(payload.items, start=1):
        response = _infer_once(item.image_urls)

        component_coverage = float(response.validation_metrics.get("component_coverage", 0.0))
        physical_validity = float(response.validation_metrics.get("physical_validity_score", 0.0))
        raw_strategy = str(response.escalation.get("strategy", EscalationStrategy.FAST_2D_FUSION.value))
        try:
            escalation_strategy = EscalationStrategy(raw_strategy)
        except ValueError:
            escalation_strategy = EscalationStrategy.FAST_2D_FUSION
        human_review = bool(response.escalation.get("human_review_required", False))

        item_results.append(
            BenchmarkItemSummary(
                item_id=item.item_id or f"item_{idx}",
                detected_type=response.detected_type,
                confidence=response.confidence,
                component_coverage=component_coverage,
                physical_validity_score=physical_validity,
                escalation_strategy=escalation_strategy,
                human_review_required=human_review,
            )
        )
        strategy_counts[escalation_strategy.value] += 1

    total_items = len(item_results)
    human_review_count = sum(1 for item in item_results if item.human_review_required)
    return BenchmarkResponse(
        items_analyzed=total_items,
        avg_confidence=round(mean(item.confidence for item in item_results), 4),
        avg_component_coverage=round(mean(item.component_coverage for item in item_results), 4),
        avg_physical_validity=round(mean(item.physical_validity_score for item in item_results), 4),
        human_review_rate=round(human_review_count / max(total_items, 1), 4),
        escalation_strategy_counts=dict(strategy_counts),
        item_results=item_results,
    )
