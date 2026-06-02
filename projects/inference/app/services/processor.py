from __future__ import annotations

from statistics import mean
from typing import Literal

from PIL import Image

from app.core.config import (
    COMPONENT_ALIAS_EXACT,
    COMPONENT_ALIAS_CONTAINS,
    HARDWARE_PARTS,
    KNOWN_INTERIOR_PARTS,
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
    InferImageResult,
    InferResponse,
)


def estimate_dimensions(category: str, image: Image.Image) -> tuple[float, float, float]:
    width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)

    if category == "desk":
        suggested_width = 1200 + max(0, (aspect - 1.2) * 220)
        return round(suggested_width, 1), 750.0, 600.0

    if category == "shelf":
        suggested_height = 1650 + max(0, (1.0 / max(aspect, 0.35) - 1.0) * 180)
        return 900.0, round(suggested_height, 1), 300.0

    suggested_height = 1200 + max(0, (1.0 / max(aspect, 0.4) - 1.0) * 120)
    return 800.0, round(suggested_height, 1), 450.0


def fallback_type_from_aspect(image: Image.Image) -> Literal["cabinet", "desk", "shelf"]:
    width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)
    if aspect >= 1.22:
        return "desk"
    if aspect <= 0.72:
        return "shelf"
    return "cabinet"


def normalize_component_name(label: str) -> str:
    normalized = "_".join(label.strip().lower().replace("-", " ").split())
    return normalized or "component"


def component_id(name: str) -> str:
    return f"cmp_{name}"


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
    if any(token in normalized for token in ("slide", "hinge", "handle", "bracket", "track")):
        return ComponentKind.HARDWARE
    if any(token in normalized for token in ("panel", "door", "shelf", "top", "bottom", "back", "side")):
        return ComponentKind.PANEL
    if any(token in normalized for token in ("leg", "support", "brace", "apron", "rail")):
        return ComponentKind.SUPPORT
    return ComponentKind.ASSEMBLY


def template_components_for_type(detected_type: str) -> list[Component]:
    if detected_type == "desk":
        return [
            Component(id=component_id("top_panel"), name="top_panel", kind=ComponentKind.PANEL, quantity=1),
            Component(id=component_id("leg"), name="leg", kind=ComponentKind.SUPPORT, quantity=4),
            Component(id=component_id("front_apron"), name="front_apron", kind=ComponentKind.SUPPORT, quantity=1),
        ]
    if detected_type == "shelf":
        return [
            Component(id=component_id("side_panel"), name="side_panel", kind=ComponentKind.PANEL, quantity=2),
            Component(id=component_id("top_panel"), name="top_panel", kind=ComponentKind.PANEL, quantity=1),
            Component(id=component_id("bottom_panel"), name="bottom_panel", kind=ComponentKind.PANEL, quantity=1),
            Component(id=component_id("shelf_panel"), name="shelf_panel", kind=ComponentKind.PANEL, quantity=4),
            Component(id=component_id("back_panel"), name="back_panel", kind=ComponentKind.PANEL, quantity=1),
        ]
    return [
        Component(id=component_id("side_panel"), name="side_panel", kind=ComponentKind.PANEL, quantity=2),
        Component(id=component_id("top_panel"), name="top_panel", kind=ComponentKind.PANEL, quantity=1),
        Component(id=component_id("bottom_panel"), name="bottom_panel", kind=ComponentKind.PANEL, quantity=1),
        Component(id=component_id("back_panel"), name="back_panel", kind=ComponentKind.PANEL, quantity=1),
        Component(id=component_id("door_panel"), name="door_panel", kind=ComponentKind.PANEL, quantity=2),
        Component(id=component_id("shelf_panel"), name="shelf_panel", kind=ComponentKind.PANEL, quantity=2),
    ]


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
    return names.issubset({"cabinet_body", "desk_frame", "shelf_panel"})


def components_from_detections(
    detections: list[tuple[int, float]],
    labels: tuple[str, ...],
    threshold: float,
    detected_type: str,
) -> list[Component]:
    counts: dict[str, tuple[ComponentKind, int]] = {}
    for class_id, score in detections:
        if class_id < 0 or class_id >= len(labels) or score < threshold:
            continue
        label = labels[class_id]
        name = canonical_component_name(label)
        kind = component_kind(label)
        current = counts.get(name)
        if current is None:
            counts[name] = (kind, 1)
        else:
            counts[name] = (current[0], current[1] + 1)

    detected_components = [
        Component(id=component_id(name), name=name, kind=kind, quantity=quantity)
        for name, (kind, quantity) in sorted(counts.items())
    ]
    fallback_components = template_components_for_type(detected_type)

    if not detected_components or is_only_product_class_components(detected_components):
        return fallback_components

    return merge_component_sets(detected_components, fallback_components)


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
    has_track = any(component.name == "sliding_door_track" for component in hardware_components)
    if has_track:
        return DoorType.SLIDING, visibility != InteriorVisibility.INTERIOR_FULLY_VISIBLE
    if door_qty > 0:
        return DoorType.HINGED, visibility != InteriorVisibility.INTERIOR_FULLY_VISIBLE
    return DoorType.UNKNOWN, False


def has_uncertain_hardware(
    detections: list[tuple[int, float]], labels: tuple[str, ...], threshold: float
) -> bool:
    soft_threshold = min(0.95, threshold + 0.15)
    for class_id, score in detections:
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


def aggregate_joints(results: list[InferImageResult]) -> list[JointEvidence]:
    merged: dict[tuple[str, str, JointType], JointEvidence] = {}
    for result in results:
        for joint in result.joints:
            key = (joint.parent_component_id, joint.child_component_id, joint.joint_type)
            current = merged.get(key)
            if current is None or joint.count > current.count:
                merged[key] = JointEvidence(
                    id=joint.id,
                    parent_component_id=joint.parent_component_id,
                    child_component_id=joint.child_component_id,
                    joint_type=joint.joint_type,
                    count=joint.count,
                )
    return [merged[key] for key in sorted(merged)]


def aggregate_hardware(results: list[InferImageResult]) -> list[HardwareRecommendation]:
    merged: dict[HardwareCode, HardwareRecommendation] = {}
    for result in results:
        for item in result.hardware:
            current = merged.get(item.code)
            if current is None or item.qty > current.qty:
                merged[item.code] = HardwareRecommendation(
                    code=item.code,
                    qty=item.qty,
                    reason=item.reason,
                )
    return [merged[key] for key in sorted(merged)]


def aggregate_components(results: list[InferImageResult]) -> list[Component]:
    merged: dict[str, Component] = {}
    for result in results:
        for component in result.components:
            current = merged.get(component.name)
            if current is None or component.quantity > current.quantity:
                merged[component.name] = Component(
                    id=component.id,
                    name=component.name,
                    kind=component.kind,
                    quantity=component.quantity,
                )
    return [merged[key] for key in sorted(merged)]


def aggregate_results(results: list[InferImageResult]) -> InferResponse:
    from fastapi import HTTPException
    
    if not results:
        raise HTTPException(status_code=422, detail="No images were analyzed")

    scores_by_type: dict[str, float] = {}
    counts_by_type: dict[str, int] = {}
    for result in results:
        scores_by_type[result.detected_type] = scores_by_type.get(result.detected_type, 0.0) + result.confidence
        counts_by_type[result.detected_type] = counts_by_type.get(result.detected_type, 0) + 1

    if any(score > 0 for score in scores_by_type.values()):
        detected_type = max(scores_by_type, key=scores_by_type.get)
    else:
        detected_type = max(counts_by_type, key=counts_by_type.get)
    
    matching = [result for result in results if result.detected_type == detected_type]
    aggregated_components = aggregate_components(results)
    structural_components, hardware_components = split_components(aggregated_components)
    
    visibility_rank = {
        InteriorVisibility.INTERIOR_NOT_VISIBLE: 0,
        InteriorVisibility.INTERIOR_PARTIALLY_VISIBLE: 1,
        InteriorVisibility.INTERIOR_FULLY_VISIBLE: 2,
    }
    interior_vis = max(results, key=lambda item: visibility_rank[item.interior.visibility]).interior.visibility
    interior_cov = round(mean(item.interior.coverage_ratio for item in results), 3)
    unknown_int = interior_vis != InteriorVisibility.INTERIOR_FULLY_VISIBLE
    
    if any(component.name == "sliding_door_track" for component in hardware_components):
        door_type = DoorType.SLIDING
    elif any(component.name == "door_panel" for component in structural_components):
        door_type = DoorType.HINGED
    else:
        door_type = DoorType.UNKNOWN
        
    door_count_uncertain = any(item.door.count_uncertain for item in results)
    uncertain_hw = any(item.uncertainty.hardware_uncertain for item in results)
    joints = aggregate_joints(results)
    hardware = aggregate_hardware(results)
    index = component_index(aggregated_components)

    return InferResponse(
        detected_type=detected_type,
        confidence=round(mean(item.confidence for item in matching), 3),
        suggested_width=round(mean(item.suggested_width for item in results), 1),
        suggested_height=round(mean(item.suggested_height for item in results), 1),
        suggested_depth=round(mean(item.suggested_depth for item in results), 1),
        components=aggregated_components,
        component_index=index,
        interior=InteriorAssessment(
            visibility=interior_vis,
            coverage_ratio=interior_cov,
            unknown_interior=unknown_int,
        ),
        door=DoorAssessment(type=door_type, count_uncertain=door_count_uncertain),
        uncertainty=UncertaintyAssessment(hardware_uncertain=uncertain_hw),
        joints=joints,
        hardware=hardware,
        image_url=results[0].image_url,
        images_analyzed=len(results),
        image_results=results,
    )
