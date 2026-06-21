from __future__ import annotations

from dataclasses import dataclass, field

from app.models.inference import ComponentKind, RawDetection


@dataclass(frozen=True)
class DetectionCandidate:
    base_name: str
    is_product_label: bool
    detection: RawDetection
    normalized_box: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class StableDetection:
    name: str
    kind: ComponentKind
    box: tuple[float, float, float, float] | None
    score: float
    track_id: int | None
    view_index: int | None


@dataclass
class ComponentAggregate:
    kind: ComponentKind
    box: tuple[float, float, float, float] | None
    best_score: float
    score_sum: float
    score_count: int
    track_ids: set[int] = field(default_factory=set)
    view_ids: set[int] = field(default_factory=set)
    quantity: int = 0
