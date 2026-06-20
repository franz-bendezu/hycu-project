from __future__ import annotations

import uuid
from collections import Counter
from statistics import mean

from PIL import Image

from app.core.config import (
    COMPONENT_ALIAS_EXACT,
    COMPONENT_ALIAS_CONTAINS,
    HARDWARE_PARTS,
    KNOWN_INTERIOR_PARTS,
    PRODUCT_TYPE_ALIASES,
    PRODUCT_TYPE_COMPONENT_HINTS,
    PRODUCT_TYPE_COMPONENT_MINIMUMS,
    PRODUCT_TYPE_CONFIDENCE_MULTIPLIERS,
)
from app.schemas import (
    Component,
    ComponentKind,
    DoorType,
    InteriorVisibility,
    JointType,
    JointEvidence,
    HardwareCode,
    HardwareRecommendation,
    InteriorAssessment,
    DoorAssessment,
    UncertaintyAssessment,
    ImageEvidence,
    InferResponse,
    ProductType,
)


def estimate_dimensions(category: ProductType, image: Image.Image) -> tuple[float, float, float]:
    width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)

    if category in {ProductType.DESK, ProductType.TABLE}:
        suggested_width = 1200 + max(0, (aspect - 1.2) * 220)
        return round(suggested_width, 1), 750.0, 600.0

    if category in {ProductType.SHELF, ProductType.BOOKCASE}:
        suggested_height = 1650 + max(0, (1.0 / max(aspect, 0.35) - 1.0) * 180)
        return 900.0, round(suggested_height, 1), 300.0

    if category in {ProductType.NIGHTSTAND, ProductType.DRESSER}:
        suggested_height = 900 + max(0, (1.0 / max(aspect, 0.45) - 1.0) * 90)
        return 700.0, round(suggested_height, 1), 450.0

    suggested_height = 1200 + max(0, (1.0 / max(aspect, 0.4) - 1.0) * 120)
    return 800.0, round(suggested_height, 1), 450.0


def product_type_from_label(label: str) -> ProductType | None:
    canonical = PRODUCT_TYPE_ALIASES.get(normalize_component_name(label))
    if canonical is None:
        return None
    try:
        return ProductType(canonical)
    except ValueError:
        return None


def infer_type_from_components(component_counts: Counter[str]) -> ProductType | None:
    if not component_counts:
        return None

    best_name: str | None = None
    best_score = -1.0

    for candidate, hints in PRODUCT_TYPE_COMPONENT_HINTS.items():
        if not hints:
            continue
        present_hints = [hint for hint in hints if component_counts.get(hint, 0) > 0]
        coverage = len(present_hints) / len(hints)
        # Cap each hint contribution to reduce over-bias from repeated shelves/drawers.
        density = sum(min(2, component_counts.get(hint, 0)) for hint in hints) / (2 * len(hints))
        score = (0.7 * coverage) + (0.3 * density)
        if score > best_score:
            best_name = candidate
            best_score = score

    if best_name is None or best_score <= 0:
        return None

    # Disambiguate drawer-centric classes that share similar signatures.
    if best_name in {"nightstand", "dresser"}:
        if component_counts.get("drawer_front", 0) >= 3 or component_counts.get("drawer_box", 0) >= 3:
            best_name = "dresser"

    # Disambiguate open-front storage: prefer shelf/bookcase when cabinet-like hardware is absent.
    door_hardware_absent = (
        component_counts.get("door_panel", 0) == 0
        and component_counts.get("hinge", 0) == 0
        and component_counts.get("sliding_door_track", 0) == 0
    )
    if best_name in {"cabinet", "wardrobe", "sideboard", "tv_stand"} and door_hardware_absent:
        if component_counts.get("shelf_panel", 0) >= 2 and component_counts.get("side_panel", 0) >= 2:
            best_name = "bookcase" if component_counts.get("back_panel", 0) > 0 else "shelf"

    try:
        return ProductType(best_name)
    except ValueError:
        return None


def minimum_components_for_type(detected_type: ProductType) -> list[Component]:
    minimums = PRODUCT_TYPE_COMPONENT_MINIMUMS.get(detected_type.value, {})
    return [
        Component(
            id=component_id(name),
            name=name,
            kind=component_kind(name),
            quantity=max(1, int(quantity)),
        )
        for name, quantity in sorted(minimums.items())
    ]


def normalize_component_name(label: str) -> str:
    normalized = "_".join(label.strip().lower().replace("-", " ").split())
    return normalized or "component"


COMPONENT_ID_NAMESPACE = uuid.UUID("5e7488f8-944f-4f20-816b-70a6f8eb8cb1")

def component_id(name: str) -> str:
    normalized = normalize_component_name(name)
    return str(uuid.uuid5(COMPONENT_ID_NAMESPACE, normalized))


def joint_id(parent_component_id: str, child_component_id: str, joint_type: JointType) -> str:
    return f"joint_{parent_component_id}_{child_component_id}_{joint_type.value}"


def canonical_component_name(label: str) -> str:
    normalized = normalize_component_name(label)
    if normalized in COMPONENT_ALIAS_EXACT:
        return COMPONENT_ALIAS_EXACT[normalized]
    for token, mapped in COMPONENT_ALIAS_CONTAINS:
        if token in normalized:
            return mapped
    return normalized


def component_kind(label: str) -> ComponentKind:
    normalized = canonical_component_name(label)
    if any(token in normalized for token in ("slide", "hinge", "handle_pull", "bracket", "track")):
        return ComponentKind.HARDWARE
    if any(token in normalized for token in ("panel", "door", "shelf", "top", "bottom", "back", "side", "drawer")):
        return ComponentKind.PANEL
    if any(token in normalized for token in ("leg", "support", "brace", "apron", "rail", "rod")):
        return ComponentKind.SUPPORT
    return ComponentKind.ASSEMBLY


def geometry_relabel_component_name(name: str, det: dict) -> str:
    box = det.get("box")
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return name

    image_w = det.get("image_width_px")
    image_h = det.get("image_height_px")
    if not isinstance(image_w, (int, float)) or not isinstance(image_h, (int, float)):
        return name
    if image_w <= 1 or image_h <= 1:
        return name

    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    width_ratio = bw / float(image_w)
    height_ratio = bh / float(image_h)
    x_center_ratio = ((x1 + x2) / 2.0) / float(image_w)
    y_center_ratio = ((y1 + y2) / 2.0) / float(image_h)
    vertical_aspect = bh / bw
    area_ratio = float(det.get("mask_fill_ratio", width_ratio * height_ratio))
    score = float(det.get("score", 0.0))

    panel_family = {
        "side_panel",
        "shelf_panel",
        "back_panel",
        "top_panel",
        "bottom_panel",
        "door_panel",
        "cabinet_body",
        "bookcase_body",
    }
    if name not in panel_family:
        return name

    if name == "door_panel" and score >= 0.35:
        return name

    # Preserve high-confidence explicit panel classes unless geometry is very obvious.
    trust_model = name in {"side_panel", "shelf_panel", "back_panel", "top_panel", "bottom_panel", "door_panel"} and score >= 0.75

    is_left_or_right_edge = x_center_ratio <= 0.22 or x_center_ratio >= 0.78
    strong_side = vertical_aspect >= 1.4 and width_ratio <= 0.24 and is_left_or_right_edge
    if strong_side and (not trust_model or name != "side_panel"):
        return "side_panel"

    strong_horizontal = vertical_aspect <= 0.45 and width_ratio >= 0.40
    if strong_horizontal and (not trust_model or name not in {"top_panel", "bottom_panel", "shelf_panel"}):
        if y_center_ratio <= 0.12:
            return "top_panel"
        if y_center_ratio >= 0.88:
            return "bottom_panel"
        return "shelf_panel"

    strong_back = (
        width_ratio >= 0.55
        and height_ratio >= 0.55
        and area_ratio >= 0.25
        and 0.20 <= x_center_ratio <= 0.80
        and 0.20 <= y_center_ratio <= 0.80
    )
    if strong_back and (not trust_model or name != "back_panel"):
        return "back_panel"

    return name


def merge_component_sets(primary: list[Component], fallback: list[Component]) -> list[Component]:
    merged: dict[str, Component] = {c.name: c for c in primary}
    for component in fallback:
        current = merged.get(component.name)
        if current is None:
            merged[component.name] = component
        elif current.quantity < component.quantity:
            merged[component.name] = Component(
                id=current.id,
                name=current.name,
                kind=current.kind,
                quantity=component.quantity,
            )
    return [merged[key] for key in sorted(merged)]


def is_only_product_class_components(components: list[Component]) -> bool:
    if not components:
        return True
    names = {component.name for component in components}
    return names.issubset(
        {
            "cabinet_body",
            "desk_frame",
            "shelf_panel",
            "table_top",
            "dresser_body",
            "nightstand_body",
            "bookcase_body",
            "tv_stand_body",
            "sideboard_body",
        }
    )


def components_from_detections(
    detections: list[dict],
    labels: tuple[str, ...],
    threshold: float,
    detected_type: ProductType,
) -> list[Component]:
    # 1. Filter and normalize detections
    valid_detections = []
    for det in detections:
        class_id = det["class_id"]
        score = det["score"]
        if class_id < 0 or class_id >= len(labels) or score < threshold:
            continue
        
        label = labels[class_id]
        name = canonical_component_name(label)
        name = geometry_relabel_component_name(name, det)
        kind = component_kind(name)
        box = det["box"] # [x1, y1, x2, y2]
        track_id = det.get("track_id")
        
        valid_detections.append({
            "name": name,
            "kind": kind,
            "box": box,
            "score": score,
            "track_id": track_id,
        })

    if not valid_detections:
        return minimum_components_for_type(detected_type)

    # 2. Stabilize per-track class jitter by selecting one dominant label per track.
    non_tracked: list[dict] = []
    track_scores: dict[int, dict[str, float]] = {}
    track_best_by_name: dict[int, dict[str, dict]] = {}

    for det in valid_detections:
        track_id = det.get("track_id")
        if not isinstance(track_id, int):
            non_tracked.append(det)
            continue

        name = str(det["name"])
        score = float(det["score"])
        score_bucket = track_scores.setdefault(track_id, {})
        score_bucket[name] = score_bucket.get(name, 0.0) + score

        best_by_name = track_best_by_name.setdefault(track_id, {})
        current_best = best_by_name.get(name)
        if current_best is None or score > float(current_best.get("score", 0.0)):
            best_by_name[name] = det

    stabilized: list[dict] = list(non_tracked)
    for track_id, scores in track_scores.items():
        if not scores:
            continue
        winner_name = max(scores, key=scores.get)
        winner = track_best_by_name.get(track_id, {}).get(winner_name)
        if winner is not None:
            stabilized.append(winner)

    # 3. Aggregate by canonical component name and count quantity.
    by_name: dict[str, dict] = {}
    for det in stabilized:
        name = det["name"]
        current = by_name.get(name)
        if current is None:
            by_name[name] = {
                "kind": det["kind"],
                "quantity": 1,
                "box": det["box"],
                "best_score": det["score"],
            }
        else:
            current["quantity"] += 1

            if det["score"] > current["best_score"]:
                current["best_score"] = det["score"]
                current["box"] = det["box"]

    components = [
        Component(
            id=component_id(name),
            name=name,
            kind=entry["kind"],
            quantity=entry["quantity"],
            box_corners=entry["box"],
        )
        for name, entry in sorted(by_name.items())
    ]

    # 4. Taxonomy prior merge only when detections are too sparse.
    if is_only_product_class_components(components):
        components = merge_component_sets(components, minimum_components_for_type(detected_type))

    return components


def is_hardware_component(component: Component) -> bool:
    return component.name in HARDWARE_PARTS or component.kind == ComponentKind.HARDWARE


def split_components(components: list[Component]) -> tuple[list[Component], list[Component]]:
    structural = [component for component in components if not is_hardware_component(component)]
    hardware = [component for component in components if is_hardware_component(component)]
    return structural, hardware


def interior_visibility(structural_components: list[Component], hardware_components: list[Component]) -> tuple[
    InteriorVisibility,
    float,
    bool,
]:
    observed = {
        component.name
        for component in [*structural_components, *hardware_components]
        if component.name in KNOWN_INTERIOR_PARTS
    }
    ratio = len(observed) / max(len(KNOWN_INTERIOR_PARTS), 1)
    if not observed:
        return InteriorVisibility.INTERIOR_NOT_VISIBLE, 0.0, True
    if ratio >= 0.6:
        return InteriorVisibility.INTERIOR_FULLY_VISIBLE, round(ratio, 3), False
    return InteriorVisibility.INTERIOR_PARTIALLY_VISIBLE, round(ratio, 3), True


def door_metadata(
    structural_components: list[Component],
    hardware_components: list[Component],
    visibility: InteriorVisibility,
) -> tuple[DoorType, bool]:
    door_qty = next((component.quantity for component in structural_components if component.name == "door_panel"), 0)
    hinge_qty = next((component.quantity for component in hardware_components if component.name == "hinge"), 0)
    has_track = any(component.name == "sliding_door_track" for component in hardware_components)
    if has_track:
        return DoorType.SLIDING, visibility != InteriorVisibility.INTERIOR_FULLY_VISIBLE
    if door_qty > 0 or hinge_qty > 0:
        return DoorType.HINGED, visibility != InteriorVisibility.INTERIOR_FULLY_VISIBLE
    return DoorType.UNKNOWN, False


def has_uncertain_hardware(
    detections: list[dict], labels: tuple[str, ...], threshold: float
) -> bool:
    soft_threshold = min(0.95, threshold + 0.15)
    for det in detections:
        class_id = det["class_id"]
        score = det["score"]
        if class_id < 0 or class_id >= len(labels):
            continue
        name = normalize_component_name(labels[class_id])
        if name in HARDWARE_PARTS and score < soft_threshold:
            return True
    return False


def component_quantity(components: list[Component], name: str) -> int:
    return sum(component.quantity for component in components if component.name == name)


def component_index(components: list[Component]) -> dict[str, Component]:
    return {component.id: component for component in components}


def build_joints(
    structural_components: list[Component],
    hardware_components: list[Component],
    door_type: DoorType,
) -> list[JointEvidence]:
    joints: list[JointEvidence] = []
    by_name = {component.name: component for component in [*structural_components, *hardware_components]}

    def comp_id(name: str) -> str:
        component = by_name.get(name)
        return component.id if component else component_id(name)

    def add_joint(parent_name: str, child_name: str, joint_type: JointType, count: int) -> None:
        parent_id = comp_id(parent_name)
        child_id = comp_id(child_name)
        joints.append(
            JointEvidence(
                id=joint_id(parent_id, child_id, joint_type),
                parent_component_id=parent_id,
                child_component_id=child_id,
                joint_type=joint_type,
                count=max(1, count),
            )
        )

    side_qty = component_quantity(structural_components, "side_panel")
    shelf_qty = component_quantity(structural_components, "shelf_panel")
    door_qty = component_quantity(structural_components, "door_panel")
    hinge_qty = component_quantity(hardware_components, "hinge")
    leg_qty = component_quantity(structural_components, "leg")
    drawer_qty = component_quantity(structural_components, "drawer_box")

    if side_qty >= 2 and component_quantity(structural_components, "top_panel") >= 1:
        add_joint("side_panel", "top_panel", JointType.CAM_LOCK, 2)
    if side_qty >= 2 and component_quantity(structural_components, "bottom_panel") >= 1:
        add_joint("side_panel", "bottom_panel", JointType.CAM_LOCK, 2)
    if side_qty >= 2 and shelf_qty > 0:
        add_joint("side_panel", "shelf_panel", JointType.SHELF_PIN, max(2, shelf_qty * 2))
    if component_quantity(structural_components, "back_panel") > 0:
        add_joint("cabinet_body", "back_panel", JointType.SCREW, 12)
    if door_qty > 0:
        if door_type == DoorType.SLIDING or any(c.name == "sliding_door_track" for c in hardware_components):
            add_joint("cabinet_body", "door_panel", JointType.SLIDING_TRACK, max(2, door_qty))
        else:
            add_joint("side_panel", "door_panel", JointType.HINGE, max(2, door_qty * 2))
    elif hinge_qty > 0:
        add_joint("side_panel", "door_panel", JointType.HINGE, max(2, hinge_qty * 2))
    if drawer_qty > 0:
        add_joint("cabinet_body", "drawer_box", JointType.TELESCOPIC_SLIDE, max(2, drawer_qty * 2))
    if leg_qty > 0 and component_quantity(structural_components, "top_panel") > 0:
        add_joint("top_panel", "leg", JointType.BRACKET, max(4, leg_qty))

    return joints


def build_hardware(joints: list[JointEvidence], uncertain_hardware: bool) -> list[HardwareRecommendation]:
    hardware_qty: dict[HardwareCode, int] = {}

    for joint in joints:
        if joint.joint_type == JointType.CAM_LOCK:
            hardware_qty[HardwareCode.CAM_LOCK_15MM] = hardware_qty.get(HardwareCode.CAM_LOCK_15MM, 0) + joint.count
        elif joint.joint_type == JointType.SHELF_PIN:
            hardware_qty[HardwareCode.SHELF_PIN_5MM] = hardware_qty.get(HardwareCode.SHELF_PIN_5MM, 0) + joint.count
        elif joint.joint_type == JointType.HINGE:
            hardware_qty[HardwareCode.HINGE_SOFT_CLOSE_110] = (
                hardware_qty.get(HardwareCode.HINGE_SOFT_CLOSE_110, 0) + joint.count
            )
            hardware_qty[HardwareCode.WOOD_SCREW_4X16] = hardware_qty.get(HardwareCode.WOOD_SCREW_4X16, 0) + (
                joint.count * 2
            )
        elif joint.joint_type == JointType.SLIDING_TRACK:
            hardware_qty[HardwareCode.SLIDING_DOOR_TRACK_SET] = (
                hardware_qty.get(HardwareCode.SLIDING_DOOR_TRACK_SET, 0) + max(1, joint.count // 2)
            )
        elif joint.joint_type == JointType.TELESCOPIC_SLIDE:
            hardware_qty[HardwareCode.TELESCOPIC_SLIDE_400] = (
                hardware_qty.get(HardwareCode.TELESCOPIC_SLIDE_400, 0) + max(1, joint.count // 2)
            )
        elif joint.joint_type in {JointType.SCREW, JointType.BRACKET}:
            hardware_qty[HardwareCode.WOOD_SCREW_4X40] = hardware_qty.get(HardwareCode.WOOD_SCREW_4X40, 0) + joint.count
            if joint.joint_type == JointType.BRACKET:
                hardware_qty[HardwareCode.CORNER_BRACKET_40] = (
                    hardware_qty.get(HardwareCode.CORNER_BRACKET_40, 0) + joint.count
                )

    recommendations = [
        HardwareRecommendation(
            code=code,
            qty=qty,
            reason="Estimated from detected joints",
        )
        for code, qty in sorted(hardware_qty.items())
        if qty > 0
    ]

    if uncertain_hardware and recommendations:
        recommendations.append(
            HardwareRecommendation(
                code=HardwareCode.HARDWARE_REVIEW_REQUIRED,
                qty=1,
                reason="Low-confidence hardware evidence detected",
            )
        )
    return recommendations


def estimate_dimensions_multi(evidence: list[ImageEvidence]) -> tuple[float, float, float]:
    widths: list[float] = []
    heights: list[float] = []
    depths: list[float] = []

    for item in evidence:
        aspect = item.width_px / max(item.height_px, 1)
        if item.detected_type in {ProductType.DESK, ProductType.TABLE}:
            w = 1200 + max(0, (aspect - 1.2) * 220)
            widths.append(w)
            heights.append(750.0)
            depths.append(600.0)
        elif item.detected_type in {ProductType.SHELF, ProductType.BOOKCASE}:
            h = 1650 + max(0, (1.0 / max(aspect, 0.35) - 1.0) * 180)
            widths.append(900.0)
            heights.append(h)
            depths.append(300.0)
        elif item.detected_type in {ProductType.NIGHTSTAND, ProductType.DRESSER}:
            h = 900 + max(0, (1.0 / max(aspect, 0.45) - 1.0) * 90)
            widths.append(700.0)
            heights.append(h)
            depths.append(450.0)
        else:
            h = 1200 + max(0, (1.0 / max(aspect, 0.4) - 1.0) * 120)
            widths.append(800.0)
            heights.append(h)
            depths.append(450.0)

    return (
        round(mean(widths or [800]), 1),
        round(mean(heights or [1200]), 1),
        round(mean(depths or [450]), 1)
    )


def assemble_project(
    evidence: list[ImageEvidence], labels: tuple[str, ...], threshold: float
) -> InferResponse:
    from fastapi import HTTPException
    
    if not evidence:
        raise HTTPException(status_code=422, detail="No visual evidence provided")

    all_detections: list[dict] = []
    for item in evidence:
        all_detections.extend(item.raw_detections)

    # Determine project-wide product type using multiple evidence sources.
    scores_by_type: dict[ProductType, float] = {}
    for item in evidence:
        scores_by_type[item.detected_type] = scores_by_type.get(item.detected_type, 0.0) + item.confidence

    component_counts: Counter[str] = Counter()
    for det in all_detections:
        class_id = det.get("class_id", -1)
        score = float(det.get("score", 0.0))
        if not isinstance(class_id, int) or class_id < 0 or class_id >= len(labels):
            continue

        label = labels[class_id]
        label_type = product_type_from_label(label)
        if label_type is not None:
            # Keep class-head logits useful but secondary to image-level votes.
            scores_by_type[label_type] = scores_by_type.get(label_type, 0.0) + (score * 0.5)
            continue

        component_counts[canonical_component_name(label)] += 1

    inferred_component_type = infer_type_from_components(component_counts)
    if inferred_component_type is not None:
        support = float(sum(component_counts.values()))
        boost = min(1.2, 0.4 + (0.1 * support))
        scores_by_type[inferred_component_type] = scores_by_type.get(inferred_component_type, 0.0) + boost

    detected_type = ProductType(max(scores_by_type, key=scores_by_type.get))
    confidence = min(1.0, scores_by_type[detected_type] / len(evidence))
    effective_threshold = threshold * PRODUCT_TYPE_CONFIDENCE_MULTIPLIERS.get(detected_type.value, 1.0)
    if confidence < effective_threshold:
        avg_aspect = mean(item.width_px / max(item.height_px, 1) for item in evidence)
        if avg_aspect >= 1.45:
            detected_type = ProductType.TABLE
        elif avg_aspect >= 1.15:
            detected_type = ProductType.DESK
        elif avg_aspect <= 0.45:
            detected_type = ProductType.BOOKCASE
        elif avg_aspect <= 0.78:
            detected_type = ProductType.SHELF
        else:
            detected_type = ProductType.CABINET
    
    # Assembly Logic (The "Model")
    components = components_from_detections(all_detections, labels, threshold, detected_type)
    structural_components, hardware_components = split_components(components)
    
    vis, cov, unknown_int = interior_visibility(structural_components, hardware_components)
    door_type, _ = door_metadata(structural_components, hardware_components, vis)
    uncertain_hw = has_uncertain_hardware(all_detections, labels, threshold)
    
    joints = build_joints(structural_components, hardware_components, door_type)
    hardware = build_hardware(joints, uncertain_hw)
    
    width, height, depth = estimate_dimensions_multi(evidence)
    index = component_index(components)

    return InferResponse(
        detected_type=detected_type,
        confidence=round(float(confidence), 3),
        suggested_width=width,
        suggested_height=height,
        suggested_depth=depth,
        components=components,
        component_index=index,
        interior=InteriorAssessment(
            visibility=vis,
            coverage_ratio=cov,
            unknown_interior=unknown_int,
        ),
        door=DoorAssessment(type=door_type, count_uncertain=False),
        uncertainty=UncertaintyAssessment(hardware_uncertain=uncertain_hw),
        joints=joints,
        hardware=hardware,
        image_url=evidence[0].image_url,
        images_analyzed=len(evidence),
        image_results=evidence,
        evidence=evidence,
    )
