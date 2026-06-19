from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackState:
    track_id: int
    class_id: int
    box: tuple[float, float, float, float]
    score: float
    age: int = 0


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

    def update(self, detections: list[dict]) -> list[dict]:
        # Age all existing tracks before matching.
        for track in self._tracks:
            track.age += 1

        assigned_track_ids: set[int] = set()
        output: list[dict] = []

        for det in detections:
            box = det.get("box")
            if not isinstance(box, (tuple, list)) or len(box) != 4:
                output.append(det)
                continue

            class_id = int(det.get("class_id", -1))
            best_idx = -1
            best_iou = 0.0

            for idx, track in enumerate(self._tracks):
                if track.track_id in assigned_track_ids:
                    continue
                if track.class_id != class_id:
                    continue
                iou_score = _iou(track.box, (float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_idx = idx

            if best_idx >= 0 and best_iou >= self.iou_threshold:
                track = self._tracks[best_idx]
                track.box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                track.score = float(det.get("score", track.score))
                track.age = 0
                assigned_track_ids.add(track.track_id)

                updated = dict(det)
                updated["track_id"] = track.track_id
                updated["track_iou"] = round(best_iou, 4)
                output.append(updated)
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks.append(
                    TrackState(
                        track_id=track_id,
                        class_id=class_id,
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        score=float(det.get("score", 0.0)),
                        age=0,
                    )
                )
                assigned_track_ids.add(track_id)

                updated = dict(det)
                updated["track_id"] = track_id
                updated["track_iou"] = 1.0
                output.append(updated)

        # Drop stale tracks.
        self._tracks = [track for track in self._tracks if track.age <= self.max_age]
        return output
