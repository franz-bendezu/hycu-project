from __future__ import annotations

import math
import uuid
from collections.abc import Iterable

from app.presentation.schemas.project_design import (
    Component,
    ComponentCategory,
    ComponentKind,
    FeatureSpec,
    HardwareAnchor,
    HardwareItem,
    HardwareMountFace,
    HardwareMountTarget,
    JointSpec,
    MaterialSpec,
    ProductSpec,
    ProjectModel,
)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _normalize_type(raw: object, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    if text in {
        "cabinet",
        "wardrobe",
        "bookcase",
        "desk",
        "table",
        "shelf",
        "nightstand",
        "dresser",
        "sideboard",
        "tv_stand",
    }:
        return text
    return fallback


def _infer_type_from_components(components: list[dict]) -> str | None:
    names = {
        str(component.get("name", "")).strip().lower()
        for component in components
        if isinstance(component, dict)
    }
    if any(name in {"desktop", "front_apron", "table_top", "desk_frame"} for name in names):
        return "desk"
    if any("door" in name for name in names) or any("drawer" in name for name in names):
        return "cabinet"
    if "shelf_panel" in names and not any("door" in name for name in names):
        return "shelf"
    return None


def _extract_image_size(result: dict) -> tuple[float, float]:
    image_results = result.get("image_results")
    if isinstance(image_results, list) and image_results:
        first = image_results[0]
        if isinstance(first, dict):
            w = _safe_float(first.get("width_px"), 1.0)
            h = _safe_float(first.get("height_px"), 1.0)
            return max(w, 1.0), max(h, 1.0)
    return 1.0, 1.0


def _component_size_mm(
    component: dict,
    *,
    kind: ComponentKind,
    image_w_px: float,
    image_h_px: float,
    target_w_mm: float,
    target_h_mm: float,
    target_d_mm: float,
    material_thickness: float,
) -> tuple[float, float, float]:
    box = component.get("box_corners")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1 = _safe_float(box[0], 0.0)
        y1 = _safe_float(box[1], 0.0)
        x2 = _safe_float(box[2], 0.0)
        y2 = _safe_float(box[3], 0.0)
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)
        if bw > 0 and bh > 0:
            width_mm = max((bw / image_w_px) * target_w_mm, material_thickness)
            height_mm = max((bh / image_h_px) * target_h_mm, material_thickness)
            depth_mm = max(target_d_mm * 0.15, material_thickness)
            return round(width_mm, 2), round(height_mm, 2), round(depth_mm, 2)

    # Kind-aware priors are used when no bounding box evidence is available.
    if kind == ComponentKind.TOP_PANEL:
        return round(target_w_mm, 2), round(material_thickness, 2), round(target_d_mm, 2)
    if kind == ComponentKind.BOTTOM_PANEL:
        return round(target_w_mm, 2), round(material_thickness, 2), round(target_d_mm, 2)
    if kind == ComponentKind.BACK_PANEL:
        return round(target_w_mm * 0.95, 2), round(target_h_mm * 0.92, 2), round(material_thickness, 2)
    if kind in {ComponentKind.LEFT_SIDE, ComponentKind.RIGHT_SIDE}:
        return round(material_thickness, 2), round(target_h_mm * 0.92, 2), round(target_d_mm * 0.92, 2)
    if kind == ComponentKind.DIVIDER_PANEL:
        return round(material_thickness, 2), round(target_h_mm * 0.78, 2), round(target_d_mm * 0.9, 2)
    if kind in {ComponentKind.FRONT_PANEL, ComponentKind.DOOR_PANEL, ComponentKind.DRAWER_FRONT}:
        return round(target_w_mm * 0.22, 2), round(target_h_mm * 0.14, 2), round(material_thickness, 2)
    if kind == ComponentKind.SHELF:
        return round(target_w_mm * 0.46, 2), round(material_thickness, 2), round(target_d_mm * 0.85, 2)
    if kind in {
        ComponentKind.LEFT_LEG_FRONT,
        ComponentKind.RIGHT_LEG_FRONT,
        ComponentKind.LEFT_LEG_BACK,
        ComponentKind.RIGHT_LEG_BACK,
    }:
        return round(material_thickness * 2, 2), round(target_h_mm * 0.92, 2), round(material_thickness * 2, 2)

    return (
        round(max(target_w_mm * 0.25, material_thickness * 2), 2),
        round(max(target_h_mm * 0.25, material_thickness * 2), 2),
        round(max(target_d_mm * 0.15, material_thickness), 2),
    )


def _as_iterable(value: object) -> Iterable[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _extend_validation_warnings(
    *,
    warnings: list[str],
    components: list[Component],
    joints: list[JointSpec],
    product: ProductSpec,
) -> None:
    kinds = [component.kind for component in components]
    drawer_components = sum(1 for kind in kinds if kind == ComponentKind.DRAWER_FRONT)
    door_components = sum(1 for kind in kinds if kind == ComponentKind.DOOR_PANEL)
    side_components = sum(1 for kind in kinds if kind in {ComponentKind.LEFT_SIDE, ComponentKind.RIGHT_SIDE})

    if product.drawer_count > 0 and drawer_components == 0:
        warnings.append("Validation: drawer_count is non-zero but no drawer components were generated")
    if product.door_count > 0 and door_components == 0:
        warnings.append("Validation: door_count is non-zero but no door components were generated")

    if drawer_components > 0 and product.drawer_count != drawer_components:
        warnings.append(
            "Validation: drawer_count does not match generated drawer components "
            f"({product.drawer_count} vs {drawer_components})"
        )
    if door_components > 0 and product.door_count != door_components:
        warnings.append(
            "Validation: door_count does not match generated door components "
            f"({product.door_count} vs {door_components})"
        )

    if product.inferred_type in {"desk", "cabinet"} and side_components < 2:
        warnings.append("Validation: expected at least two side support panels for cabinet/desk layout")

    if not joints:
        warnings.append("Validation: no joints were generated; assembly placement may be unreliable")
        return

    child_ids_with_joints = {joint.child_id for joint in joints}
    critical_kinds = {
        ComponentKind.LEFT_SIDE,
        ComponentKind.RIGHT_SIDE,
        ComponentKind.TOP_PANEL,
        ComponentKind.BOTTOM_PANEL,
        ComponentKind.BACK_PANEL,
        ComponentKind.DIVIDER_PANEL,
    }
    critical_components = [component for component in components if component.kind in critical_kinds]
    unanchored_critical = [
        component for component in critical_components if component.id not in child_ids_with_joints
    ]
    if unanchored_critical:
        warnings.append(
            "Validation: some structural components have no joint placement ("
            f"{len(unanchored_critical)} missing)"
        )

    drawer_ids = {component.id for component in components if component.kind == ComponentKind.DRAWER_FRONT}
    if drawer_ids:
        anchored_drawers = len([component_id for component_id in drawer_ids if component_id in child_ids_with_joints])
        if anchored_drawers < max(1, len(drawer_ids) // 2):
            warnings.append(
                "Validation: most drawer components are not anchored by joints "
                f"({anchored_drawers}/{len(drawer_ids)})"
            )


def _joint_pose_magnitude(joint: JointSpec) -> float:
    return (
        abs(joint.pos_x)
        + abs(joint.pos_y)
        + abs(joint.pos_z)
        + abs(joint.rot_x)
        + abs(joint.rot_y)
        + abs(joint.rot_z)
    )


def _normalize_joints(joints: list[JointSpec], valid_component_ids: set[str]) -> list[JointSpec]:
    # Drop broken links first.
    filtered: list[JointSpec] = []
    for joint in joints:
        if joint.parent_id not in valid_component_ids or joint.child_id not in valid_component_ids:
            continue
        if joint.parent_id == joint.child_id:
            continue
        filtered.append(joint)

    # Remove duplicate constraints between the same pair by keeping the stronger
    # (non-zero pose) joint.
    by_pair: dict[tuple[str, str], JointSpec] = {}
    for joint in filtered:
        pair_key = tuple(sorted((joint.parent_id, joint.child_id)))
        existing = by_pair.get(pair_key)
        if existing is None or _joint_pose_magnitude(joint) > _joint_pose_magnitude(existing):
            by_pair[pair_key] = joint

    return list(by_pair.values())


def _category_for_kind(kind: ComponentKind) -> ComponentCategory:
    if kind in {
        ComponentKind.LEFT_SIDE,
        ComponentKind.RIGHT_SIDE,
        ComponentKind.TOP_PANEL,
        ComponentKind.BOTTOM_PANEL,
        ComponentKind.BACK_PANEL,
        ComponentKind.DIVIDER_PANEL,
    }:
        return ComponentCategory.STRUCTURAL
    if kind in {ComponentKind.DOOR_PANEL, ComponentKind.DRAWER_FRONT, ComponentKind.FRONT_PANEL}:
        return ComponentCategory.FRONT
    if kind == ComponentKind.SHELF:
        return ComponentCategory.INTERNAL
    return ComponentCategory.SUPPORT


def build_project_model_from_inference(
    result: dict,
    *,
    project_name: str,
    fallback_type: str = "cabinet",
    material_thickness: float = 18.0,
) -> ProjectModel:
    source_components = list(_as_iterable(result.get("components")))
    component_type = _infer_type_from_components(source_components)
    if component_type is not None:
        inferred_type = component_type
    else:
        inferred_type = _normalize_type(result.get("detected_type"), fallback=fallback_type)
    target_width = max(_safe_float(result.get("suggested_width"), 800.0), 1.0)
    target_height = max(_safe_float(result.get("suggested_height"), 1200.0), 1.0)
    target_depth = max(_safe_float(result.get("suggested_depth"), 450.0), 1.0)

    image_w_px, image_h_px = _extract_image_size(result)

    components: list[Component] = []
    component_id_map: dict[str, list[str]] = {}
    component_name_map: dict[str, list[str]] = {}

    side_index = 0
    leg_index = 0
    skipped_names: list[str] = []
    product_container_names = {
        "cabinet_body",
        "bookcase_body",
        "desk_frame",
        "nightstand_body",
        "dresser_body",
        "sideboard_body",
        "tv_stand_body",
        "drawer_box",
    }

    def map_kind(name: str) -> ComponentKind | None:
        nonlocal side_index, leg_index
        if name in {"top_panel", "desktop", "table_top"}:
            return ComponentKind.TOP_PANEL
        if name == "bottom_panel":
            return ComponentKind.BOTTOM_PANEL
        if name == "back_panel":
            return ComponentKind.BACK_PANEL
        if name in {"front_panel", "front_apron"}:
            return ComponentKind.FRONT_PANEL
        if name in {"shelf", "shelf_panel"}:
            return ComponentKind.SHELF
        if name == "divider_panel":
            return ComponentKind.DIVIDER_PANEL
        if name == "door_panel":
            return ComponentKind.DOOR_PANEL
        if name == "drawer_front":
            return ComponentKind.DRAWER_FRONT
        if name == "side_panel":
            kind = ComponentKind.LEFT_SIDE if side_index % 2 == 0 else ComponentKind.RIGHT_SIDE
            side_index += 1
            return kind
        if name == "leg":
            cycle = (
                ComponentKind.LEFT_LEG_FRONT,
                ComponentKind.RIGHT_LEG_FRONT,
                ComponentKind.LEFT_LEG_BACK,
                ComponentKind.RIGHT_LEG_BACK,
            )
            kind = cycle[leg_index % len(cycle)]
            leg_index += 1
            return kind
        return None

    for idx, comp in enumerate(source_components):
        name = str(comp.get("name", "")).strip().lower()
        if not name:
            continue

        quantity = max(_safe_int(comp.get("quantity"), 1), 1)

        base_source_id = str(comp.get("id", f"cv_{name}_{idx + 1}"))
        mapped_any = False
        for q in range(quantity):
            kind = map_kind(name)
            if kind is None:
                break

            width_mm, height_mm, depth_mm = _component_size_mm(
                comp,
                kind=kind,
                image_w_px=image_w_px,
                image_h_px=image_h_px,
                target_w_mm=target_width,
                target_h_mm=target_height,
                target_d_mm=target_depth,
                material_thickness=material_thickness,
            )

            comp_id = str(uuid.uuid4())
            component_id_map.setdefault(base_source_id, []).append(comp_id)
            component_name_map.setdefault(name, []).append(comp_id)
            mapped_any = True
            components.append(
                Component(
                    id=comp_id,
                    kind=kind,
                    category=_category_for_kind(kind),
                    material_id="material_board_default",
                    width=width_mm,
                    height=height_mm,
                    depth=depth_mm,
                )
            )

        if not mapped_any and name not in product_container_names:
            skipped_names.append(name)

    door_count = sum(max(_safe_int(c.get("quantity"), 1), 1) for c in source_components if str(c.get("name", "")).strip().lower() == "door_panel")
    drawer_front_count = sum(
        max(_safe_int(c.get("quantity"), 1), 1)
        for c in source_components
        if str(c.get("name", "")).strip().lower() == "drawer_front"
    )
    drawer_box_count = sum(
        max(_safe_int(c.get("quantity"), 1), 1)
        for c in source_components
        if str(c.get("name", "")).strip().lower() == "drawer_box"
    )
    drawer_count = drawer_front_count if drawer_front_count > 0 else drawer_box_count
    if inferred_type == "desk":
        drawer_count = min(drawer_count, 4)
    shelf_count = sum(max(_safe_int(c.get("quantity"), 1), 1) for c in source_components if str(c.get("name", "")).strip().lower() in {"shelf", "shelf_panel"})
    divider_count = sum(max(_safe_int(c.get("quantity"), 1), 1) for c in source_components if str(c.get("name", "")).strip().lower() == "divider_panel")

    product = ProductSpec(
        id=str(uuid.uuid4()),
        sku=f"SKU-{uuid.uuid4().hex[:8]}",
        name=project_name,
        inferred_type=inferred_type,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
        material_thickness=material_thickness,
        shelf_count=shelf_count,
        divider_count=divider_count,
        door_count=door_count,
        drawer_count=drawer_count,
    )

    materials = [
        MaterialSpec(
            id="material_board_default",
            thickness_mm=material_thickness,
            texture_map_url=None,
        )
    ]

    hardware: list[HardwareItem] = []
    for entry in _as_iterable(result.get("hardware")):
        code = str(entry.get("code", "")).strip()
        if not code:
            continue
        qty = max(_safe_int(entry.get("qty"), 1), 1)
        anchor = None
        try:
            anchor = HardwareAnchor(code)
        except ValueError:
            anchor = None

        mount_targets: list[HardwareMountTarget] = []
        face_map: dict[str, HardwareMountFace] = {
            "+x": HardwareMountFace.POS_X,
            "-x": HardwareMountFace.NEG_X,
            "+y": HardwareMountFace.POS_Y,
            "-y": HardwareMountFace.NEG_Y,
            "+z": HardwareMountFace.POS_Z,
            "-z": HardwareMountFace.NEG_Z,
        }
        for target in _as_iterable(entry.get("mount_targets")):
            if not isinstance(target, dict):
                continue
            component_id = str(target.get("component_id", "")).strip()
            if not component_id:
                continue
            raw_face = str(target.get("face", "")).strip().lower()
            face = face_map.get(raw_face)
            if face is None:
                continue
            mount_targets.append(
                HardwareMountTarget(
                    component_id=component_id,
                    face=face,
                    local_x=_safe_float(target.get("local_x"), 0.0),
                    local_y=_safe_float(target.get("local_y"), 0.0),
                    local_z=_safe_float(target.get("local_z"), 0.0),
                    normal_offset_mm=max(_safe_float(target.get("normal_offset_mm"), 2.0), 0.0),
                )
            )

        lower = code.lower()
        hardware.append(
            HardwareItem(
                code=code,
                qty=qty,
                id=str(entry.get("id", f"hardware_{lower}")),
                anchor=anchor,
                mesh_path=entry.get("mesh_path"),
                svg_path=entry.get("svg_path"),
                mount_targets=mount_targets,
            )
        )

    inference_joints: list[JointSpec] = []
    seen_joint_pairs: set[tuple[str, str]] = set()
    for entry in _as_iterable(result.get("joints")):
        parent_source = str(entry.get("parent_component_id", "")).strip()
        child_source = str(entry.get("child_component_id", "")).strip()
        if not parent_source or not child_source:
            continue

        parent_ids = component_id_map.get(parent_source)
        child_ids = component_id_map.get(child_source)
        if parent_ids is None:
            parent_ids = component_name_map.get(parent_source.lower())
        if child_ids is None:
            child_ids = component_name_map.get(child_source.lower())

        if not parent_ids or not child_ids:
            continue

        inferred_count = max(_safe_int(entry.get("count"), 1), 1)
        if len(child_ids) > 1:
            joint_target = max(len(child_ids), min(inferred_count, len(child_ids)))
        else:
            joint_target = max(1, min(inferred_count, len(parent_ids)))

        for idx in range(joint_target):
            parent_id = parent_ids[idx % len(parent_ids)]
            child_id = child_ids[idx % len(child_ids)]
            pair = (parent_id, child_id)
            if pair in seen_joint_pairs:
                continue
            seen_joint_pairs.add(pair)
            inference_joints.append(
                JointSpec(
                    parent_id=parent_id,
                    child_id=child_id,
                    joint_rule=None,
                    pos_x=0.0,
                    pos_y=0.0,
                    pos_z=0.0,
                    rot_x=0.0,
                    rot_y=0.0,
                    rot_z=0.0,
                )
            )

    joints: list[JointSpec] = list(inference_joints)

    # If CV did not return spatial joints for desks, place common supports and
    # drawer fronts in a stable two-pedestal layout to avoid origin-stacked parts.
    weak_inference_joints = (
        len(inference_joints) <= 2
        and all(_joint_pose_magnitude(joint) == 0.0 for joint in inference_joints)
    )
    if inferred_type == "desk" and (len(joints) <= 1 or weak_inference_joints):
        # Ignore weak zero-pose joints so they don't conflict with desk fallback.
        joints = []
        top = next((component for component in components if component.kind == ComponentKind.TOP_PANEL), None)
        left_side = next((component for component in components if component.kind == ComponentKind.LEFT_SIDE), None)
        right_side = next((component for component in components if component.kind == ComponentKind.RIGHT_SIDE), None)
        divider = next((component for component in components if component.kind == ComponentKind.DIVIDER_PANEL), None)
        drawer_fronts = [component for component in components if component.kind == ComponentKind.DRAWER_FRONT]

        # Pedestal mode is enabled when desk has side supports and drawers.
        pedestal_mode = (
            top is not None
            and left_side is not None
            and right_side is not None
            and len(drawer_fronts) >= 4
        )

        if pedestal_mode and top is not None and left_side is not None and right_side is not None:
            top_thickness = max(top.height, material_thickness)
            support_height = _clamp(
                target_height - top_thickness - material_thickness,
                target_height * 0.60,
                target_height * 0.94,
            )
            pedestal_depth = _clamp(target_depth * 0.86, target_depth * 0.68, target_depth - material_thickness)

            max_drawer_face_w = max((drawer.width for drawer in drawer_fronts), default=target_width * 0.18)
            pedestal_width = _clamp(
                max(max_drawer_face_w * 1.18, target_width * 0.20),
                target_width * 0.20,
                target_width * 0.30,
            )

            half_width = target_width / 2.0
            lateral_clearance = max(material_thickness * 2.0, target_width * 0.04)
            max_center_offset = half_width - (pedestal_width / 2.0) - lateral_clearance
            center_offset = _clamp(target_width * 0.34, material_thickness * 3.0, max_center_offset)

            # Align side supports with pedestal geometry.
            left_side.width = round(material_thickness, 2)
            left_side.height = round(support_height, 2)
            left_side.depth = round(pedestal_depth, 2)
            right_side.width = round(material_thickness, 2)
            right_side.height = round(support_height, 2)
            right_side.depth = round(pedestal_depth, 2)

            # Ensure two explicit pedestal carcass volume blocks.
            pedestal_block_height = round(_clamp(support_height * 0.94, support_height * 0.72, support_height), 2)
            pedestal_block_width = round(pedestal_width, 2)
            pedestal_block_depth = round(_clamp(pedestal_depth * 0.94, pedestal_depth * 0.72, pedestal_depth), 2)

            existing_dividers = [component for component in components if component.kind == ComponentKind.DIVIDER_PANEL]
            while len(existing_dividers) < 2:
                block = Component(
                    id=str(uuid.uuid4()),
                    kind=ComponentKind.DIVIDER_PANEL,
                    material_id="material_board_default",
                    width=pedestal_block_width,
                    height=pedestal_block_height,
                    depth=pedestal_block_depth,
                )
                components.append(block)
                existing_dividers.append(block)

            left_pedestal_block = existing_dividers[0]
            right_pedestal_block = existing_dividers[1]
            for block in (left_pedestal_block, right_pedestal_block):
                block.width = pedestal_block_width
                block.height = pedestal_block_height
                block.depth = pedestal_block_depth

            support_center_y = round(-((target_height - top_thickness) / 2.0), 2)
            block_center_y = round(support_center_y + (support_height - pedestal_block_height) / 2.0, 2)

            joints.append(
                JointSpec(
                    parent_id=top.id,
                    child_id=left_side.id,
                    joint_rule=None,
                    pos_x=round(-center_offset, 2),
                    pos_y=support_center_y,
                    pos_z=0.0,
                    rot_x=0.0,
                    rot_y=0.0,
                    rot_z=0.0,
                )
            )
            joints.append(
                JointSpec(
                    parent_id=top.id,
                    child_id=right_side.id,
                    joint_rule=None,
                    pos_x=round(center_offset, 2),
                    pos_y=support_center_y,
                    pos_z=0.0,
                    rot_x=0.0,
                    rot_y=0.0,
                    rot_z=0.0,
                )
            )
            joints.append(
                JointSpec(
                    parent_id=top.id,
                    child_id=left_pedestal_block.id,
                    joint_rule=None,
                    pos_x=round(-center_offset, 2),
                    pos_y=block_center_y,
                    pos_z=0.0,
                    rot_x=0.0,
                    rot_y=0.0,
                    rot_z=0.0,
                )
            )
            joints.append(
                JointSpec(
                    parent_id=top.id,
                    child_id=right_pedestal_block.id,
                    joint_rule=None,
                    pos_x=round(center_offset, 2),
                    pos_y=block_center_y,
                    pos_z=0.0,
                    rot_x=0.0,
                    rot_y=0.0,
                    rot_z=0.0,
                )
            )

            rows = max(1, math.ceil(len(drawer_fronts) / 2))
            usable_height = max(support_height - (material_thickness * 3.0), material_thickness * 6.0)
            drawer_slot_height = usable_height / rows
            drawer_face_height = _clamp(drawer_slot_height * 0.82, material_thickness * 2.0, target_height * 0.22)
            drawer_face_width = _clamp(pedestal_width * 0.82, material_thickness * 4.0, pedestal_width - material_thickness * 2.0)
            drawer_depth = _clamp(pedestal_depth * 0.82, material_thickness * 2.0, pedestal_depth - material_thickness)

            drawer_top_limit = support_center_y + (support_height / 2.0) - material_thickness
            drawer_bottom_limit = support_center_y - (support_height / 2.0) + material_thickness
            z_limit = (pedestal_depth / 2.0) - (drawer_depth / 2.0) - (material_thickness * 0.25)

            for idx, drawer in enumerate(drawer_fronts):
                col = idx % 2
                row = idx // 2

                drawer.width = round(drawer_face_width, 2)
                drawer.height = round(drawer_face_height, 2)
                drawer.depth = round(drawer_depth, 2)

                x_pos = round((-center_offset) if col == 0 else center_offset, 2)
                y_raw = drawer_top_limit - ((row + 0.5) * drawer_slot_height)
                y_pos = round(_clamp(y_raw, drawer_bottom_limit, drawer_top_limit), 2)
                z_pos = round(max(material_thickness, z_limit), 2)

                joints.append(
                    JointSpec(
                        parent_id=top.id,
                        child_id=drawer.id,
                        joint_rule=None,
                        pos_x=x_pos,
                        pos_y=y_pos,
                        pos_z=z_pos,
                        rot_x=0.0,
                        rot_y=0.0,
                        rot_z=0.0,
                    )
                )

            # Optional center divider can stay, but keep it low-profile so it does not
            # interpenetrate the drawer bay envelope.
            if divider is not None:
                divider.width = round(material_thickness, 2)
                divider.height = round(_clamp(support_height * 0.82, support_height * 0.50, support_height), 2)
                divider.depth = round(_clamp(pedestal_depth * 0.82, pedestal_depth * 0.55, pedestal_depth), 2)
                joints.append(
                    JointSpec(
                        parent_id=top.id,
                        child_id=divider.id,
                        joint_rule=None,
                        pos_x=0.0,
                        pos_y=round(support_center_y + material_thickness, 2),
                        pos_z=0.0,
                        rot_x=0.0,
                        rot_y=0.0,
                        rot_z=0.0,
                    )
                )

        if not pedestal_mode and top is not None:
            if left_side is not None:
                joints.append(
                    JointSpec(
                        parent_id=top.id,
                        child_id=left_side.id,
                        joint_rule=None,
                        pos_x=round(-target_width * 0.42, 2),
                        pos_y=round(-target_height * 0.36, 2),
                        pos_z=0.0,
                        rot_x=0.0,
                        rot_y=0.0,
                        rot_z=0.0,
                    )
                )
            if right_side is not None:
                joints.append(
                    JointSpec(
                        parent_id=top.id,
                        child_id=right_side.id,
                        joint_rule=None,
                        pos_x=round(target_width * 0.42, 2),
                        pos_y=round(-target_height * 0.36, 2),
                        pos_z=0.0,
                        rot_x=0.0,
                        rot_y=0.0,
                        rot_z=0.0,
                    )
                )
            if divider is not None:
                joints.append(
                    JointSpec(
                        parent_id=top.id,
                        child_id=divider.id,
                        joint_rule=None,
                        pos_x=0.0,
                        pos_y=round(-target_height * 0.33, 2),
                        pos_z=0.0,
                        rot_x=0.0,
                        rot_y=0.0,
                        rot_z=0.0,
                    )
                )

        if not pedestal_mode and drawer_fronts:
            rows = max(1, math.ceil(len(drawer_fronts) / 2))
            for idx, drawer in enumerate(drawer_fronts):
                col = idx % 2
                row = idx // 2
                x_pos = round((-target_width * 0.42) if col == 0 else (target_width * 0.42), 2)
                y_start = target_height * 0.22
                y_step = max(target_height * 0.16, 40.0)
                y_pos = round(y_start - min(row, rows - 1) * y_step, 2)

                parent_id = top.id if top is not None else drawer.id
                joints.append(
                    JointSpec(
                        parent_id=parent_id,
                        child_id=drawer.id,
                        joint_rule=None,
                        pos_x=x_pos,
                        pos_y=y_pos,
                        pos_z=round(target_depth * 0.46, 2),
                        rot_x=0.0,
                        rot_y=0.0,
                        rot_z=0.0,
                    )
                )

    joints = _normalize_joints(joints, {component.id for component in components})

    warnings: list[str] = []
    if skipped_names:
        skipped_set = sorted(set(skipped_names))
        warnings.append(
            "Vision-first mode skipped unsupported component kinds: " + ", ".join(skipped_set)
        )

    _extend_validation_warnings(
        warnings=warnings,
        components=components,
        joints=joints,
        product=product,
    )

    return ProjectModel(
        product=product,
        materials=materials,
        components=components,
        hardware=hardware,
        joints=joints,
        features=list(_as_iterable(result.get("features"))) if isinstance(result.get("features"), list) else [],
        warnings=warnings,
    )
