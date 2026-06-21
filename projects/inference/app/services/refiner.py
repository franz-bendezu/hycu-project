from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

from app.core.config import get_enable_heavy_refinement
from app.models import Component, EscalationStrategy, ImageEvidence, InferResponse, RawDetection


@dataclass(frozen=True)
class RectangleCandidate:
    component_id: str
    component_name: str
    box: tuple[float, float, float, float]
    score: float
    depth_score: float = 0.0


def _area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _normalize_box(
    box: tuple[float, float, float, float] | list[float] | None,
    image_w: int | float | None,
    image_h: int | float | None,
) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
        return (x1, y1, x2, y2)
    if not isinstance(image_w, (int, float)) or not isinstance(image_h, (int, float)):
        return None
    if image_w <= 0 or image_h <= 0:
        return None
    return (x1 / float(image_w), y1 / float(image_h), x2 / float(image_w), y2 / float(image_h))


def _split_candidate(candidate: RectangleCandidate) -> list[RectangleCandidate]:
    x1, y1, x2, y2 = candidate.box
    w = x2 - x1
    h = y2 - y1
    if w >= h and w > 0.16:
        mid = (x1 + x2) * 0.5
        return [
            RectangleCandidate(
                candidate.component_id,
                candidate.component_name,
                (x1, y1, mid, y2),
                max(0.0, candidate.score - 0.05),
                candidate.depth_score,
            ),
            RectangleCandidate(
                candidate.component_id,
                candidate.component_name,
                (mid, y1, x2, y2),
                max(0.0, candidate.score - 0.05),
                candidate.depth_score,
            ),
        ]
    if h > 0.16:
        mid = (y1 + y2) * 0.5
        return [
            RectangleCandidate(
                candidate.component_id,
                candidate.component_name,
                (x1, y1, x2, mid),
                max(0.0, candidate.score - 0.05),
                candidate.depth_score,
            ),
            RectangleCandidate(
                candidate.component_id,
                candidate.component_name,
                (x1, mid, x2, y2),
                max(0.0, candidate.score - 0.05),
                candidate.depth_score,
            ),
        ]
    return [candidate]


def _merge_candidates(a: RectangleCandidate, b: RectangleCandidate) -> RectangleCandidate:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    merged_box = (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))
    merged_score = min(1.0, (a.score + b.score) * 0.5)
    merged_depth = (a.depth_score + b.depth_score) * 0.5
    return RectangleCandidate(a.component_id, a.component_name, merged_box, merged_score, merged_depth)


def _depth_value(det: RawDetection) -> float | None:
    for key in ("depth_mm", "depth_mean_mm", "mean_depth_mm"):
        raw = getattr(det, key, None)
        if isinstance(raw, (int, float)) and float(raw) > 0:
            return float(raw)
    return None


def _depth_preference(component_name: str, depth_norm: float) -> float:
    name = component_name.lower()
    # Lower depth is closer to camera.
    if any(token in name for token in ("door", "drawer", "front", "handle")):
        return 1.0 - depth_norm
    if "back" in name:
        return depth_norm
    if any(token in name for token in ("shelf", "side", "divider", "top", "bottom")):
        return 1.0 - abs(depth_norm - 0.5) * 2.0
    return 0.5


def _component_depth_scores_from_evidence(
    components: list[Component],
    evidence: list[ImageEvidence],
) -> tuple[dict[str, float], int]:
    scores: dict[str, float] = {component.id: 0.0 for component in components}
    components_with_boxes = [component for component in components if component.box_corners is not None]
    if not components_with_boxes or not evidence:
        return scores, 0

    samples: dict[str, list[float]] = {component.id: [] for component in components_with_boxes}

    for image in evidence:
        for det in image.raw_detections:
            depth = _depth_value(det)
            if depth is None:
                continue
            det_box = _normalize_box(det.box, det.image_width_px, det.image_height_px)
            if det_box is None:
                continue

            best_component: Component | None = None
            best_iou = 0.0
            for component in components_with_boxes:
                comp_box = component.box_corners
                if comp_box is None:
                    continue
                overlap = _iou(det_box, comp_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_component = component

            if best_component is not None and best_iou >= 0.35:
                samples[best_component.id].append(depth)

    all_depths = [value for values in samples.values() for value in values]
    if not all_depths:
        return scores, 0

    min_depth = min(all_depths)
    max_depth = max(all_depths)
    spread = max(1e-6, max_depth - min_depth)

    for component in components_with_boxes:
        values = samples.get(component.id, [])
        if not values:
            continue
        avg_depth = sum(values) / len(values)
        depth_norm = min(1.0, max(0.0, (avg_depth - min_depth) / spread))
        scores[component.id] = round(min(1.0, max(0.0, _depth_preference(component.name, depth_norm))), 5)

    return scores, len(all_depths)


def _generate_candidates(components: list[Component], depth_scores: dict[str, float]) -> list[RectangleCandidate]:
    seeds: list[RectangleCandidate] = []
    for component in components:
        if component.box_corners is None:
            continue
        seeds.append(
            RectangleCandidate(
                component.id,
                component.name,
                component.box_corners,
                float(component.confidence),
                float(depth_scores.get(component.id, 0.0)),
            )
        )

    if not seeds:
        return []

    augmented: list[RectangleCandidate] = list(seeds)
    for seed in seeds:
        augmented.extend(_split_candidate(seed))

    merged: list[RectangleCandidate] = []
    for idx in range(len(seeds)):
        for jdx in range(idx + 1, len(seeds)):
            if _iou(seeds[idx].box, seeds[jdx].box) >= 0.55:
                merged.append(_merge_candidates(seeds[idx], seeds[jdx]))
    augmented.extend(merged)

    unique: dict[tuple[str, float, float, float, float], RectangleCandidate] = {}
    for candidate in augmented:
        x1, y1, x2, y2 = candidate.box
        key = (
            candidate.component_id,
            round(x1, 3),
            round(y1, 3),
            round(x2, 3),
            round(y2, 3),
        )
        current = unique.get(key)
        if current is None or candidate.score > current.score:
            unique[key] = candidate

    return list(unique.values())


def _subset_energy(candidates: list[RectangleCandidate]) -> float:
    if not candidates:
        return 0.0
    cover = sum(_area(candidate.box) for candidate in candidates)
    overlap_penalty = 0.0
    for idx in range(len(candidates)):
        for jdx in range(idx + 1, len(candidates)):
            overlap_penalty += _iou(candidates[idx].box, candidates[jdx].box)
    weight = sum(candidate.score for candidate in candidates)
    depth_signal = sum(candidate.depth_score for candidate in candidates)
    size_penalty = 0.025 * len(candidates)
    return (1.6 * cover) + (0.9 * weight) + (0.35 * depth_signal) - (0.8 * overlap_penalty) - size_penalty


def _canonical_key(candidate: RectangleCandidate) -> str:
    x1, y1, x2, y2 = candidate.box
    return f"{candidate.component_id}:{x1:.5f}:{y1:.5f}:{x2:.5f}:{y2:.5f}:{candidate.score:.5f}:{candidate.depth_score:.5f}"


def _seed_from_candidates(candidates: list[RectangleCandidate]) -> int:
    payload = "|".join(sorted(_canonical_key(candidate) for candidate in candidates))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _accept_move(delta: float, temperature: float, rng: random.Random) -> bool:
    if delta >= 0.0:
        return True
    safe_temp = max(1e-4, temperature)
    prob = math.exp(max(-60.0, min(0.0, delta / safe_temp)))
    return rng.random() < prob


def _select_subset(
    candidates: list[RectangleCandidate],
) -> tuple[list[RectangleCandidate], float, dict[str, int | float]]:
    if not candidates:
        return [], 0.0, {
            "iterations": 0,
            "accepted_moves": 0,
            "birth_moves": 0,
            "death_moves": 0,
            "exchange_moves": 0,
        }

    seed = _seed_from_candidates(candidates)
    rng = random.Random(seed)
    ordered = sorted(candidates, key=_canonical_key)

    current: list[RectangleCandidate] = []
    current_energy = _subset_energy(current)
    best = list(current)
    best_energy = current_energy

    iterations = min(160, max(60, len(ordered) * 8))
    accepted = 0
    birth_moves = 0
    death_moves = 0
    exchange_moves = 0

    for step in range(iterations):
        temperature = max(0.05, 1.0 - (0.92 * (step / max(iterations - 1, 1))))
        in_set = {_canonical_key(candidate) for candidate in current}
        out_pool = [candidate for candidate in ordered if _canonical_key(candidate) not in in_set]

        proposal = list(current)
        move_roll = rng.random()

        if out_pool and (not current or move_roll < 0.42):
            birth_moves += 1
            candidate = out_pool[rng.randrange(len(out_pool))]
            proposal.append(candidate)
        elif current and move_roll < 0.74:
            death_moves += 1
            remove_idx = rng.randrange(len(current))
            proposal.pop(remove_idx)
        elif current and out_pool:
            exchange_moves += 1
            remove_idx = rng.randrange(len(current))
            proposal.pop(remove_idx)
            add_candidate = out_pool[rng.randrange(len(out_pool))]
            proposal.append(add_candidate)
        else:
            continue

        dedup: dict[str, RectangleCandidate] = {}
        for item in proposal:
            dedup[_canonical_key(item)] = item
        proposal = list(dedup.values())

        proposal_energy = _subset_energy(proposal)
        if _accept_move(proposal_energy - current_energy, temperature, rng):
            current = proposal
            current_energy = proposal_energy
            accepted += 1
            if current_energy > best_energy:
                best = list(current)
                best_energy = current_energy

    stats: dict[str, int | float] = {
        "iterations": iterations,
        "accepted_moves": accepted,
        "birth_moves": birth_moves,
        "death_moves": death_moves,
        "exchange_moves": exchange_moves,
    }
    return best, best_energy, stats


def _apply_geometry_refinement(response: InferResponse) -> InferResponse:
    depth_scores, depth_samples = _component_depth_scores_from_evidence(response.components, response.evidence)

    # Strict mode: geometry refinement requires real depth evidence.
    if depth_samples <= 0:
        response.constraints_report["proposal_depth_signal"] = 0.0
        response.constraints_report["proposal_depth_samples"] = 0
        response.escalation["strategy"] = EscalationStrategy.GEOMETRY_REFINEMENT_REQUIRES_DEPTH.value
        if "geometry_refinement_missing_depth" not in response.review_flags:
            response.review_flags.append("geometry_refinement_missing_depth")
        return response

    candidates = _generate_candidates(response.components, depth_scores)
    selected, energy, stats = _select_subset(candidates)

    selected_ids = {candidate.component_id for candidate in selected}
    for component in response.components:
        if component.id in selected_ids and component.box_corners is not None:
            component.confidence = round(min(1.0, component.confidence + 0.06), 4)
            component.uncertainty = round(max(0.0, component.uncertainty - 0.06), 4)

    response.constraints_report["proposal_candidates"] = len(candidates)
    response.constraints_report["proposal_selected"] = len(selected)
    response.constraints_report["proposal_energy"] = round(energy, 5)
    response.constraints_report["proposal_depth_signal"] = round(sum(candidate.depth_score for candidate in candidates), 5)
    response.constraints_report["proposal_depth_samples"] = int(depth_samples)
    response.constraints_report["proposal_sampler_iterations"] = int(stats["iterations"])
    response.constraints_report["proposal_sampler_accepted"] = int(stats["accepted_moves"])
    response.constraints_report["proposal_sampler_birth"] = int(stats["birth_moves"])
    response.constraints_report["proposal_sampler_death"] = int(stats["death_moves"])
    response.constraints_report["proposal_sampler_exchange"] = int(stats["exchange_moves"])
    response.escalation["proposal_refiner"] = "deterministic_rjmcmc_v1"
    response.escalation["strategy"] = EscalationStrategy.GEOMETRY_REFINEMENT_APPLIED.value
    if "geometry_refinement_applied" not in response.review_flags:
        response.review_flags.append("geometry_refinement_applied")
    return response


class HeavyRefiner:
    """
    Optional heavy reconstruction hook used only for high-uncertainty requests.

    The current implementation is deterministic and side-effect free. It records
    whether heavy refinement was attempted so downstream systems can branch
    between online and offline refinement paths.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = get_enable_heavy_refinement() if enabled is None else bool(enabled)

    def maybe_refine(self, response: InferResponse, evidence: list[ImageEvidence]) -> InferResponse:
        strategy = str(response.escalation.get("strategy", ""))
        if strategy == EscalationStrategy.ESCALATE_GEOMETRY_OPTIMIZATION.value:
            return _apply_geometry_refinement(response)
        if strategy != EscalationStrategy.ESCALATE_MVS_REFINEMENT.value:
            return response

        if not self.enabled:
            if "heavy_refinement_disabled" not in response.review_flags:
                response.review_flags.append("heavy_refinement_disabled")
            response.constraints_report["heavy_refinement_attempted"] = False
            return response

        # Hook point for MVS/NeRF/3DGS refinement. Keep deterministic metadata now.
        response.constraints_report["heavy_refinement_attempted"] = True
        response.escalation["strategy"] = EscalationStrategy.MVS_REFINEMENT_APPLIED.value
        response.escalation["refiner"] = "heavy_refiner_v1"
        response.escalation["views_used"] = len(evidence)
        if "heavy_refinement_applied" not in response.review_flags:
            response.review_flags.append("heavy_refinement_applied")

        return response
