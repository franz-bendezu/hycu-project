from __future__ import annotations

from dataclasses import dataclass, field

from app.models import RawDetection


@dataclass
class TrackState:
    track_id: int
    class_id: int
    box: tuple[float, float, float, float]
    score: float
    age: int = 0
    class_scores: dict[int, float] = field(default_factory=dict)


def _dominant_class_id(class_scores: dict[int, float], fallback: int) -> int:
    if not class_scores:
        return fallback
    return max(class_scores, key=class_scores.get)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class MultiViewTracker:
    """
    ByteTrack-style lightweight IoU tracker for multi-image identity consistency.

    This implementation is dependency-free and production-safe for static photo batches.
    """

    def __init__(self, iou_threshold: float = 0.35, max_age: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_track_id = 1
        self._tracks: list[TrackState] = []

    def update(self, detections: list[RawDetection]) -> list[RawDetection]:
        # Age all existing tracks before matching.
        for track in self._tracks:
            track.age += 1

        assigned_track_ids: set[int] = set()
        output: list[RawDetection] = []

        for det in detections:
            box = det.box

            class_id = int(det.class_id)
            best_idx = -1
            best_score = -1.0
            best_iou = 0.0

            for idx, track in enumerate(self._tracks):
                if track.track_id in assigned_track_ids:
                    continue
                iou_score = _iou(track.box, (float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                if iou_score <= 0:
                    continue

                # Prefer same-class matches, but allow class-jitter recovery when IoU is strong.
                class_bonus = 0.10 if track.class_id == class_id else -0.12
                candidate_score = iou_score + class_bonus
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_iou = iou_score
                    best_idx = idx

            cross_class_iou_floor = max(self.iou_threshold + 0.18, 0.55)
            if best_idx >= 0 and (
                best_iou >= self.iou_threshold
                or (self._tracks[best_idx].class_id != class_id and best_iou >= cross_class_iou_floor)
            ):
                track = self._tracks[best_idx]
                track.box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                track.score = float(det.score)
                track.class_scores[class_id] = track.class_scores.get(class_id, 0.0) + float(det.score)
                track.class_id = _dominant_class_id(track.class_scores, class_id)
                track.age = 0
                assigned_track_ids.add(track.track_id)

                updated = det.model_copy(
                    update={
                        "track_id": track.track_id,
                        "track_iou": round(best_iou, 4),
                        "class_id": track.class_id,
                    }
                )
                output.append(updated)
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                initial_score = float(det.score)
                self._tracks.append(
                    TrackState(
                        track_id=track_id,
                        class_id=class_id,
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        score=initial_score,
                        age=0,
                        class_scores={class_id: initial_score},
                    )
                )
                assigned_track_ids.add(track_id)

                updated = det.model_copy(update={"track_id": track_id, "track_iou": 1.0})
                output.append(updated)

        # Drop stale tracks.
        self._tracks = [track for track in self._tracks if track.age <= self.max_age]
        return output
