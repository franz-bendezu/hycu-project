from __future__ import annotations

import uuid
from statistics import mean

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
    ImageEvidence,
    InferResponse,
    ProductType,
)


def estimate_dimensions(category: ProductType, image: Image.Image) -> tuple[float, float, float]:
    width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)

    if category == ProductType.DESK:
        suggested_width = 1200 + max(0, (aspect - 1.2) * 220)
        return round(suggested_width, 1), 750.0, 600.0

    if category == ProductType.SHELF:
        suggested_height = 1650 + max(0, (1.0 / max(aspect, 0.35) - 1.0) * 180)
        return 900.0, round(suggested_height, 1), 300.0

    suggested_height = 1200 + max(0, (1.0 / max(aspect, 0.4) - 1.0) * 120)
    return 800.0, round(suggested_height, 1), 450.0


def fallback_type_from_aspect(image: Image.Image) -> ProductType:
    width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)
    if aspect >= 1.22:
        return ProductType.DESK
    if aspect <= 0.72:
        return ProductType.SHELF
    return ProductType.CABINET


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
    if any(token in normalized for token in ("panel", "door", "shelf", "top", "bottom", "back", "side")):
        return ComponentKind.PANEL
    if any(token in normalized for token in ("leg", "support", "brace", "apron", "rail")):
        return ComponentKind.SUPPORT
    return ComponentKind.ASSEMBLY


def template_components_for_type(detected_type: ProductType) -> list[Component]:
    if detected_type == ProductType.DESK:
        return [
            Component(id=component_id("top_panel"), name="top_panel", kind=ComponentKind.PANEL, quantity=1),
            Component(id=component_id("left_leg"), name="left_leg", kind=ComponentKind.SUPPORT, quantity=1, position_label="left"),
            Component(id=component_id("right_leg"), name="right_leg", kind=ComponentKind.SUPPORT, quantity=1, position_label="right"),
            Component(id=component_id("front_apron"), name="front_apron", kind=ComponentKind.SUPPORT, quantity=1),
        ]
    if detected_type == ProductType.SHELF:
        return [
            Component(id=component_id("left_side_panel"), name="left_side_panel", kind=ComponentKind.PANEL, quantity=1, position_label="left"),
            Component(id=component_id("right_side_panel"), name="right_side_panel", kind=ComponentKind.PANEL, quantity=1, position_label="right"),
            Component(id=component_id("top_panel"), name="top_panel", kind=ComponentKind.PANEL, quantity=1, position_label="top"),
            Component(id=component_id("bottom_panel"), name="bottom_panel", kind=ComponentKind.PANEL, quantity=1, position_label="bottom"),
            Component(id=component_id("shelf_panel"), name="shelf_panel", kind=ComponentKind.PANEL, quantity=3),
            Component(id=component_id("back_panel"), name="back_panel", kind=ComponentKind.PANEL, quantity=1, position_label="back"),
        ]
    return [
        Component(id=component_id("left_side_panel"), name="left_side_panel", kind=ComponentKind.PANEL, quantity=1, position_label="left"),
        Component(id=component_id("right_side_panel"), name="right_side_panel", kind=ComponentKind.PANEL, quantity=1, position_label="right"),
        Component(id=component_id("top_panel"), name="top_panel", kind=ComponentKind.PANEL, quantity=1, position_label="top"),
        Component(id=component_id("bottom_panel"), name="bottom_panel", kind=ComponentKind.PANEL, quantity=1, position_label="bottom"),
        Component(id=component_id("back_panel"), name="back_panel", kind=ComponentKind.PANEL, quantity=1, position_label="back"),
        Component(id=component_id("left_door_panel"), name="left_door_panel", kind=ComponentKind.PANEL, quantity=1, position_label="left"),
        Component(id=component_id("right_door_panel"), name="right_door_panel", kind=ComponentKind.PANEL, quantity=1, position_label="right"),
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
        kind = component_kind(label)
        box = det["box"] # [xc, yc, w, h] or [x1, y1, x2, y2] depends on export
        
        valid_detections.append({
            "name": name,
            "kind": kind,
            "box": box,
            "score": score
        })

    if not valid_detections:
        return template_components_for_type(detected_type)

    # 2. Aggregate by canonical component name and count quantity.
    by_name: dict[str, dict] = {}
    for det in valid_detections:
        name = det["name"]
        current = by_name.get(name)
        if current is None:
            by_name[name] = {
                "kind": det["kind"],
                "quantity": 1,
                "box": det["box"],
            }
        else:
            current["quantity"] += 1

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

    # 3. Template merge for critical structural completeness.
    template = template_components_for_type(detected_type)
    existing_names = set(by_name)
    for t_comp in template:
        if t_comp.name not in existing_names:
            components.append(t_comp)

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
    from statistics import mean
    
    widths: list[float] = []
    heights: list[float] = []
    depths: list[float] = []
    
    for item in evidence:
        aspect = item.width_px / max(item.height_px, 1)
        if item.detected_type == "desk":
            w = 1200 + max(0, (aspect - 1.2) * 220)
            widths.append(w)
            heights.append(750.0)
            depths.append(600.0)
        elif item.detected_type == "shelf":
            h = 1650 + max(0, (1.0 / max(aspect, 0.35) - 1.0) * 180)
            widths.append(900.0)
            heights.append(h)
            depths.append(300.0)
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

    # Determine project-wide product type
    scores_by_type: dict[str, float] = {}
    for item in evidence:
        scores_by_type[item.detected_type] = scores_by_type.get(item.detected_type, 0.0) + item.confidence

    detected_type = ProductType(max(scores_by_type, key=scores_by_type.get))
    confidence = min(1.0, scores_by_type[detected_type] / len(evidence))
    if confidence < threshold:
        avg_aspect = mean(item.width_px / max(item.height_px, 1) for item in evidence)
        if avg_aspect >= 1.22:
            detected_type = ProductType.DESK
        elif avg_aspect <= 0.72:
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
