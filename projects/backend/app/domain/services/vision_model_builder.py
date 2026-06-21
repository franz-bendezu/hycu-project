from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from importlib import import_module

from app.domain.services.product_types import (
    CABINET_PROFILE_ALIASES,
    normalize_known_product_type,
)
from app.presentation.schemas.project_design import (
    Component,
    ComponentCategory,
    ComponentKind,
    FaceSpec,
    FeatureSpec,
    HardwareAnchor,
    HardwareItem,
    HardwareMountFace,
    HardwareMountTarget,
    JointRule,
    JointSpec,
    MaterialSpec,
    ProductSpec,
    ProjectModel,
)
from app.presentation.schemas.inference import InferenceOutput


def _assign_component_transforms(product: ProductSpec, components: list[Component], joints: list[JointSpec]) -> None:
    module = import_module("app.domain.services.component_transformer")
    module.assign_component_transforms(product=product, components=components, joints=joints)


FACE_ORDER: tuple[HardwareMountFace, ...] = (
    HardwareMountFace.POS_X,
    HardwareMountFace.NEG_X,
    HardwareMountFace.POS_Y,
    HardwareMountFace.NEG_Y,
    HardwareMountFace.POS_Z,
    HardwareMountFace.NEG_Z,
)


def _face_id(component_id: str, face: HardwareMountFace) -> str:
    return f"{component_id}:{face.value}:{uuid.uuid4()}"


def _component_id_from_face_id(face_id: str) -> str:
    return face_id.split(":", 1)[0]


def _face_from_id(face_id: str) -> HardwareMountFace | None:
    try:
        return HardwareMountFace(face_id.split(":")[1])
    except (ValueError, IndexError):
        return None


def _set_component_faces(components: list[Component]) -> None:
    for component in components:
        component.faces = [
            FaceSpec(id=_face_id(component.id, face), component_id=component.id, normal=face)
            for face in FACE_ORDER
        ]


def _ensure_full_component_faces(components: list[Component]) -> None:
    for component in components:
        existing_normals = {face.normal for face in component.faces}
        for normal in FACE_ORDER:
            if normal in existing_normals:
                continue
            component.faces.append(FaceSpec(id=_face_id(component.id, normal), component_id=component.id, normal=normal))


def _face_index(components: list[Component]) -> dict[tuple[str, HardwareMountFace], str]:
    return {
        (component.id, face.normal): face.id
        for component in components
        for face in component.faces
    }


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _normalize_type(raw: object, fallback: str) -> str:
    text = normalize_known_product_type(raw)
    if text is not None:
        return text
    return fallback


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

    linked_component_ids = {
        _component_id_from_face_id(joint.parent_face_id) for joint in joints
    } | {
        _component_id_from_face_id(joint.child_face_id) for joint in joints
    }
    critical_kinds = {
        ComponentKind.LEFT_SIDE,
        ComponentKind.RIGHT_SIDE,
        ComponentKind.TOP_PANEL,
        ComponentKind.BOTTOM_PANEL,
        ComponentKind.BACK_PANEL,
        ComponentKind.DIVIDER_PANEL,
    }
    critical_components = [component for component in components if component.kind in critical_kinds]
    unanchored_critical = [component for component in critical_components if component.id not in linked_component_ids]
    if unanchored_critical:
        warnings.append(
            "Validation: some structural components have no joint placement ("
            f"{len(unanchored_critical)} missing)"
        )

    drawer_ids = {component.id for component in components if component.kind == ComponentKind.DRAWER_FRONT}
    if drawer_ids:
        anchored_drawers = len([component_id for component_id in drawer_ids if component_id in linked_component_ids])
        if anchored_drawers < max(1, len(drawer_ids) // 2):
            warnings.append(
                "Validation: most drawer components are not anchored by joints "
                f"({anchored_drawers}/{len(drawer_ids)})"
            )


def _joint_pose_magnitude(joint: JointSpec) -> float:
    return (
        abs(joint.offset_u)
        + abs(joint.offset_v)
        + abs(joint.clearance)
    )


def _face_normal(face: HardwareMountFace) -> tuple[int, int, int]:
    if face == HardwareMountFace.POS_X:
        return (1, 0, 0)
    if face == HardwareMountFace.NEG_X:
        return (-1, 0, 0)
    if face == HardwareMountFace.POS_Y:
        return (0, 1, 0)
    if face == HardwareMountFace.NEG_Y:
        return (0, -1, 0)
    if face == HardwareMountFace.POS_Z:
        return (0, 0, 1)
    return (0, 0, -1)


def _faces_are_opposed(parent_face: HardwareMountFace, child_face: HardwareMountFace) -> bool:
    px, py, pz = _face_normal(parent_face)
    cx, cy, cz = _face_normal(child_face)
    return (px + cx, py + cy, pz + cz) == (0, 0, 0)


def _faces_match_axis(face_a: HardwareMountFace, face_b: HardwareMountFace, axis: str) -> bool:
    if axis == "x":
        return {face_a, face_b} == {HardwareMountFace.POS_X, HardwareMountFace.NEG_X}
    if axis == "y":
        return {face_a, face_b} == {HardwareMountFace.POS_Y, HardwareMountFace.NEG_Y}
    return {face_a, face_b} == {HardwareMountFace.POS_Z, HardwareMountFace.NEG_Z}


def _joint_faces_compatible(
    parent_kind: ComponentKind,
    child_kind: ComponentKind,
    parent_face: HardwareMountFace,
    child_face: HardwareMountFace,
) -> bool:
    pair = {parent_kind, child_kind}

    side_like = {ComponentKind.LEFT_SIDE, ComponentKind.RIGHT_SIDE, ComponentKind.DIVIDER_PANEL}
    top_bottom = {ComponentKind.TOP_PANEL, ComponentKind.BOTTOM_PANEL}
    front_like = {ComponentKind.FRONT_PANEL, ComponentKind.DOOR_PANEL, ComponentKind.DRAWER_FRONT}

    if pair & side_like and pair & top_bottom:
        return _faces_match_axis(parent_face, child_face, "y")

    if ComponentKind.BACK_PANEL in pair and (pair & side_like or pair & top_bottom):
        return _faces_match_axis(parent_face, child_face, "z")

    if pair & front_like and (pair & side_like or pair & {ComponentKind.DIVIDER_PANEL, ComponentKind.BACK_PANEL}):
        return _faces_match_axis(parent_face, child_face, "z")

    return True


def _normalize_joints(joints: list[JointSpec], component_by_id: dict[str, Component]) -> list[JointSpec]:
    # Drop broken links first.
    filtered: list[JointSpec] = []
    valid_component_ids = set(component_by_id)
    for joint in joints:
        parent_id = _component_id_from_face_id(joint.parent_face_id)
        child_id = _component_id_from_face_id(joint.child_face_id)
        if parent_id not in valid_component_ids or child_id not in valid_component_ids:
            continue
        if parent_id == child_id:
            continue
        parent_face = _face_from_id(joint.parent_face_id)
        child_face = _face_from_id(joint.child_face_id)
        if parent_face is None or child_face is None:
            continue
        if not _faces_are_opposed(parent_face, child_face):
            continue
        parent_component = component_by_id.get(parent_id)
        child_component = component_by_id.get(child_id)
        if parent_component is None or child_component is None:
            continue
        if not _joint_faces_compatible(parent_component.kind, child_component.kind, parent_face, child_face):
            continue
        filtered.append(joint)

    # Remove duplicate constraints between the same pair of faces.
    by_pair: dict[tuple[str, str], JointSpec] = {}
    for joint in filtered:
        pair_key = tuple(sorted((joint.parent_face_id, joint.child_face_id)))
        existing = by_pair.get(pair_key)
        if existing is None or _joint_pose_magnitude(joint) > _joint_pose_magnitude(existing):
            by_pair[pair_key] = joint

    return list(by_pair.values())


def _ensure_minimum_connectivity_joints(components: list[Component], joints: list[JointSpec]) -> list[JointSpec]:
    face_index = _face_index(components)
    by_id = {component.id: component for component in components}

    side_supports = [
        component
        for component in components
        if component.kind in {ComponentKind.LEFT_SIDE, ComponentKind.RIGHT_SIDE, ComponentKind.DIVIDER_PANEL}
    ]
    tops = [component for component in components if component.kind == ComponentKind.TOP_PANEL]
    bottoms = [component for component in components if component.kind == ComponentKind.BOTTOM_PANEL]
    backs = [component for component in components if component.kind == ComponentKind.BACK_PANEL]
    drawers = [component for component in components if component.kind == ComponentKind.DRAWER_FRONT]
    front_like = [
        component
        for component in components
        if component.kind in {ComponentKind.FRONT_PANEL, ComponentKind.DOOR_PANEL}
    ]

    existing_pairs = {
        tuple(sorted((joint.parent_face_id, joint.child_face_id)))
        for joint in joints
    }

    def add_joint(
        parent: Component,
        child: Component,
        parent_face: HardwareMountFace,
        child_face: HardwareMountFace,
        rule: JointRule,
    ) -> None:
        parent_face_id = face_index.get((parent.id, parent_face))
        child_face_id = face_index.get((child.id, child_face))
        if parent_face_id is None or child_face_id is None:
            return

        pair_key = tuple(sorted((parent_face_id, child_face_id)))
        if pair_key in existing_pairs:
            return

        joints.append(
            JointSpec(
                id=str(uuid.uuid4()),
                parent_face_id=parent_face_id,
                child_face_id=child_face_id,
                joint_rule=rule,
                offset_u=0.0,
                offset_v=0.0,
                clearance=0.0,
            )
        )
        existing_pairs.add(pair_key)

    # Structural frame: supports to top/bottom using vertical contact faces.
    for support in side_supports:
        for top in tops:
            add_joint(support, top, HardwareMountFace.POS_Y, HardwareMountFace.NEG_Y, JointRule.OVERLAP)
        for bottom in bottoms:
            add_joint(support, bottom, HardwareMountFace.NEG_Y, HardwareMountFace.POS_Y, JointRule.OVERLAP)

    # Back panel should be anchored to carcass supports.
    back_supports = side_supports or tops or bottoms
    if back_supports:
        for back in backs:
            for support in back_supports[:4]:
                add_joint(support, back, HardwareMountFace.POS_Z, HardwareMountFace.NEG_Z, JointRule.FLUSH_BACK)

    # Front parts should be anchored to any available support.
    supports_for_front = side_supports or tops or bottoms
    if supports_for_front:
        base_support = supports_for_front[0]
        for drawer in drawers:
            add_joint(base_support, drawer, HardwareMountFace.POS_Z, HardwareMountFace.NEG_Z, JointRule.BETWEEN)
        for front in front_like:
            add_joint(base_support, front, HardwareMountFace.POS_Z, HardwareMountFace.NEG_Z, JointRule.MOUNT)

    # Keep only joints that still reference existing components/faces.
    return _normalize_joints(joints, by_id)


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


def _is_front_like_kind(kind: ComponentKind) -> bool:
    return kind in {ComponentKind.DOOR_PANEL, ComponentKind.DRAWER_FRONT, ComponentKind.FRONT_PANEL}


def _preferred_hardware_targets(code: str, components: list[Component]) -> list[Component]:
    upper = code.upper()
    doors = [component for component in components if component.kind == ComponentKind.DOOR_PANEL]
    drawers = [component for component in components if component.kind == ComponentKind.DRAWER_FRONT]
    fronts = [component for component in components if component.kind == ComponentKind.FRONT_PANEL]
    front_like = [component for component in components if _is_front_like_kind(component.kind)]

    if "HINGE" in upper and doors:
        return doors
    if ("SLIDE" in upper or "TRACK" in upper or "RAIL" in upper) and drawers:
        return drawers
    if "CAM_LOCK" in upper and (drawers or doors):
        return drawers or doors
    if front_like:
        return front_like
    return []


def _synthesize_mount_targets(code: str, qty: int, components: list[Component]) -> list[HardwareMountTarget]:
    target_components = _preferred_hardware_targets(code, components)
    if not target_components:
        return []

    upper = code.upper()

    def offset_for(
        component: Component,
        *,
        component_idx: int,
        local_idx: int,
    ) -> tuple[float, float, float]:
        width = max(component.width, 1.0)
        height = max(component.height, 1.0)
        depth = max(component.depth, 1.0)

        x_inset = _clamp(width * 0.38, 10.0, 85.0)
        y_inset = _clamp(height * 0.35, 16.0, 220.0)
        z_local = round(depth * 0.5, 3)

        if "HINGE" in upper:
            side = -1.0 if component_idx % 2 == 0 else 1.0
            y_slots = [-y_inset, y_inset, 0.0]
            return (round(side * x_inset, 3), round(y_slots[local_idx % len(y_slots)], 3), z_local)

        if "CAM_LOCK" in upper:
            corners = [
                (-x_inset, -y_inset),
                (x_inset, -y_inset),
                (-x_inset, y_inset),
                (x_inset, y_inset),
            ]
            x, y = corners[local_idx % len(corners)]
            return (round(x, 3), round(y, 3), z_local)

        if "SCREW" in upper or "BOLT" in upper:
            perimeter = [
                (-x_inset, -y_inset),
                (0.0, -y_inset),
                (x_inset, -y_inset),
                (-x_inset, 0.0),
                (x_inset, 0.0),
                (-x_inset, y_inset),
                (0.0, y_inset),
                (x_inset, y_inset),
            ]
            x, y = perimeter[local_idx % len(perimeter)]
            return (round(x, 3), round(y, 3), z_local)

        # Generic fallback keeps broad spacing.
        broad = [
            (-x_inset, -y_inset),
            (x_inset, -y_inset),
            (-x_inset, y_inset),
            (x_inset, y_inset),
            (0.0, 0.0),
        ]
        x, y = broad[local_idx % len(broad)]
        return (round(x, 3), round(y, 3), z_local)

    targets: list[HardwareMountTarget] = []
    for idx in range(max(qty, 0)):
        component_idx = idx % len(target_components)
        component = target_components[component_idx]
        local_x, local_y, local_z = offset_for(component, component_idx=component_idx, local_idx=idx // len(target_components))

        targets.append(
            HardwareMountTarget(
                component_id=component.id,
                face=HardwareMountFace.POS_Z,
                local_x=local_x,
                local_y=local_y,
                local_z=local_z,
                normal_offset_mm=2.0,
            )
        )

    return targets


def _normalize_inferred_type_from_mapped_components(
    inferred_type: str,
    components: list[Component],
) -> str:
    if inferred_type != "desk":
        return inferred_type

    kinds = {component.kind for component in components}
    has_leg = any(
        kind in {
            ComponentKind.LEFT_LEG_FRONT,
            ComponentKind.RIGHT_LEG_FRONT,
            ComponentKind.LEFT_LEG_BACK,
            ComponentKind.RIGHT_LEG_BACK,
        }
        for kind in kinds
    )
    has_storage_carcass = (
        ComponentKind.LEFT_SIDE in kinds
        and ComponentKind.RIGHT_SIDE in kinds
        and (ComponentKind.TOP_PANEL in kinds or ComponentKind.BOTTOM_PANEL in kinds)
        and (ComponentKind.BACK_PANEL in kinds or ComponentKind.DIVIDER_PANEL in kinds)
    )
    has_storage_front = any(kind in {ComponentKind.DOOR_PANEL, ComponentKind.DRAWER_FRONT, ComponentKind.FRONT_PANEL} for kind in kinds)

    if not has_leg and (has_storage_carcass or has_storage_front):
        return "cabinet"
    return inferred_type


def _ensure_side_support_pair(
    components: list[Component],
    *,
    inferred_type: str,
    target_height: float,
    target_depth: float,
    material_thickness: float,
) -> None:
    if inferred_type not in {"cabinet", "desk", *CABINET_PROFILE_ALIASES}:
        return

    if len(components) < 4:
        return

    has_front_or_internal_signal = any(
        component.kind in {
            ComponentKind.DIVIDER_PANEL,
            ComponentKind.DRAWER_FRONT,
            ComponentKind.DOOR_PANEL,
            ComponentKind.FRONT_PANEL,
            ComponentKind.SHELF,
            ComponentKind.BACK_PANEL,
            ComponentKind.TOP_PANEL,
            ComponentKind.BOTTOM_PANEL,
        }
        for component in components
    )
    if not has_front_or_internal_signal:
        return

    left = [component for component in components if component.kind == ComponentKind.LEFT_SIDE]
    right = [component for component in components if component.kind == ComponentKind.RIGHT_SIDE]
    if left and right:
        return

    template = (left or right or [
        component for component in components if component.kind == ComponentKind.DIVIDER_PANEL
    ])
    sample = template[0] if template else None

    width = max(sample.width, material_thickness) if sample is not None else material_thickness
    height = max(sample.height, target_height * 0.92) if sample is not None else target_height * 0.92
    depth = max(sample.depth, target_depth * 0.92) if sample is not None else target_depth * 0.92

    def create_side(kind: ComponentKind) -> Component:
        side = Component(
            id=str(uuid.uuid4()),
            kind=kind,
            category=ComponentCategory.STRUCTURAL,
            material_id="material_board_default",
            width=round(width, 2),
            height=round(height, 2),
            depth=round(depth, 2),
        )
        _set_component_faces([side])
        return side

    if not left:
        components.append(create_side(ComponentKind.LEFT_SIDE))
    if not right:
        components.append(create_side(ComponentKind.RIGHT_SIDE))


def _build_project_model_from_inference(
    inference: InferenceOutput,
    *,
    project_name: str,
    fallback_type: str = "cabinet",
    material_thickness: float = 18.0,
    assign_component_transforms: Callable[[ProductSpec, list[Component], list[JointSpec]], None] = _assign_component_transforms,
) -> ProjectModel:
    source_components = [component.model_dump(mode="json") for component in inference.components]
    source_faces = [face.model_dump(mode="json") for face in inference.faces]
    source_joints = [joint.model_dump(mode="json") for joint in inference.joints]
    if not source_components:
        raise ValueError("Inference payload must include non-empty 'components'")
    if not source_faces:
        raise ValueError("Inference payload must include non-empty 'faces'")
    if not source_joints:
        raise ValueError("Inference payload must include non-empty 'joints'")

    inferred_type = _normalize_type(inference.detected_type, fallback=fallback_type)
    target_width = max(float(inference.suggested_width), 1.0)
    target_height = max(float(inference.suggested_height), 1.0)
    target_depth = max(float(inference.suggested_depth), 1.0)

    components: list[Component] = []
    component_by_id: dict[str, Component] = {}
    source_component_id_to_uuid: dict[str, str] = {}
    for entry in source_components:
        source_component_id = str(entry.get("id", "")).strip()
        if not source_component_id:
            raise ValueError("Inference component is missing 'id'")
        try:
            kind = ComponentKind(str(entry.get("kind", "")).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Inference component '{source_component_id}' has invalid kind") from exc

        width = _safe_float(entry.get("width"), -1.0)
        height = _safe_float(entry.get("height"), -1.0)
        depth = _safe_float(entry.get("depth"), -1.0)
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError(f"Inference component '{source_component_id}' must include positive width/height/depth")

        component_id = str(uuid.uuid4())
        source_component_id_to_uuid[source_component_id] = component_id

        component = Component(
            id=component_id,
            kind=kind,
            category=_category_for_kind(kind),
            material_id=str(entry.get("material_id") or "material_board_default"),
            width=width,
            height=height,
            depth=depth,
        )
        components.append(component)
        component_by_id[component.id] = component

    source_face_to_uuid: dict[str, str] = {}
    for entry in source_faces:
        source_face_id = str(entry.get("id", "")).strip()
        source_component_id = str(entry.get("component_id", "")).strip()
        normal_raw = str(entry.get("normal", "")).strip().lower()
        if not source_face_id or not source_component_id or not normal_raw:
            raise ValueError("Inference face must include id, component_id and normal")
        component_id = source_component_id_to_uuid.get(source_component_id)
        if component_id is None:
            raise ValueError(
                f"Inference face '{source_face_id}' references unknown component '{source_component_id}'"
            )
        component = component_by_id.get(component_id)
        if component is None:
            raise ValueError(f"Inference face '{source_face_id}' references unknown component '{source_component_id}'")
        try:
            normal = HardwareMountFace(normal_raw)
        except ValueError as exc:
            raise ValueError(f"Inference face '{source_face_id}' has invalid normal '{normal_raw}'") from exc

        face_uuid = _face_id(component_id, normal)
        component.faces.append(FaceSpec(id=face_uuid, component_id=component_id, normal=normal))
        source_face_to_uuid[source_face_id] = face_uuid

    if any(not component.faces for component in components):
        raise ValueError("Each inferred component must include at least one face")

    _ensure_full_component_faces(components)

    _ensure_side_support_pair(
        components,
        inferred_type=inferred_type,
        target_height=target_height,
        target_depth=target_depth,
        material_thickness=material_thickness,
    )

    face_index = _face_index(components)
    face_ids = {face.id for component in components for face in component.faces}

    door_count = sum(1 for component in components if component.kind == ComponentKind.DOOR_PANEL)
    drawer_count = sum(1 for component in components if component.kind == ComponentKind.DRAWER_FRONT)
    if inferred_type == "desk":
        drawer_count = min(drawer_count, 4)
    shelf_count = sum(1 for component in components if component.kind == ComponentKind.SHELF)
    divider_count = sum(1 for component in components if component.kind == ComponentKind.DIVIDER_PANEL)

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
    for hardware_entry in inference.hardware or []:
        code = str(hardware_entry.code).strip()
        if not code:
            continue
        qty = max(int(hardware_entry.qty), 1)
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
        for target in _as_iterable((hardware_entry.model_dump(mode="json")).get("mount_targets")):
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

        if not mount_targets:
            mount_targets = _synthesize_mount_targets(code, qty, components)

        lower = code.lower()
        hardware.append(
            HardwareItem(
                code=code,
                qty=qty,
                id=f"hardware_{lower}",
                anchor=anchor,
                mesh_path=None,
                svg_path=None,
                mount_targets=mount_targets,
            )
        )

    joints: list[JointSpec] = []
    for entry in source_joints:
        joint_id = str(entry.get("id", "")).strip()
        parent_face_source_id = str(entry.get("parent_face_id", "")).strip()
        child_face_source_id = str(entry.get("child_face_id", "")).strip()
        if not joint_id or not parent_face_source_id or not child_face_source_id:
            raise ValueError("Inference joint must include id, parent_face_id and child_face_id")

        parent_face_id = source_face_to_uuid.get(parent_face_source_id)
        child_face_id = source_face_to_uuid.get(child_face_source_id)

        if parent_face_id is None:
            parent_component_id = source_component_id_to_uuid.get(_component_id_from_face_id(parent_face_source_id), "")
            parent_face = _face_from_id(parent_face_source_id)
            if parent_component_id and parent_face is not None:
                parent_face_id = face_index.get((parent_component_id, parent_face))
        if child_face_id is None:
            child_component_id = source_component_id_to_uuid.get(_component_id_from_face_id(child_face_source_id), "")
            child_face = _face_from_id(child_face_source_id)
            if child_component_id and child_face is not None:
                child_face_id = face_index.get((child_component_id, child_face))

        if parent_face_id is None or child_face_id is None:
            raise ValueError(f"Inference joint '{joint_id}' references unknown faces")
        if parent_face_id not in face_ids or child_face_id not in face_ids:
            raise ValueError(f"Inference joint '{joint_id}' references unknown faces")

        joint_rule = entry.get("joint_rule")
        parsed_rule = None
        if isinstance(joint_rule, str) and joint_rule.strip():
            try:
                parsed_rule = JointRule(joint_rule.strip().lower())
            except ValueError as exc:
                raise ValueError(f"Inference joint '{joint_id}' has invalid joint_rule") from exc

        joints.append(
            JointSpec(
                id=str(uuid.uuid4()),
                parent_face_id=parent_face_id,
                child_face_id=child_face_id,
                joint_rule=parsed_rule,
                offset_u=_safe_float(entry.get("offset_u"), 0.0),
                offset_v=_safe_float(entry.get("offset_v"), 0.0),
                clearance=max(_safe_float(entry.get("clearance"), 0.0), 0.0),
            )
        )

    joints = _normalize_joints(joints, component_by_id)
    joints = _ensure_minimum_connectivity_joints(components, joints)

    warnings: list[str] = []

    _extend_validation_warnings(
        warnings=warnings,
        components=components,
        joints=joints,
        product=product,
    )

    assign_component_transforms(product=product, components=components, joints=joints)

    return ProjectModel(
        product=product,
        materials=materials,
        components=components,
        hardware=hardware,
        joints=joints,
        features=[],
        warnings=warnings,
    )


class VisionModelBuilder:
    def __init__(
        self,
        assign_component_transforms: Callable[[ProductSpec, list[Component], list[JointSpec]], None] = _assign_component_transforms,
    ) -> None:
        self._assign_component_transforms = assign_component_transforms

    def build_project_model_from_inference(
        self,
        inference: InferenceOutput,
        *,
        project_name: str,
        fallback_type: str = "cabinet",
        material_thickness: float = 18.0,
    ) -> ProjectModel:
        return _build_project_model_from_inference(
            inference,
            project_name=project_name,
            fallback_type=fallback_type,
            material_thickness=material_thickness,
            assign_component_transforms=self._assign_component_transforms,
        )


_default_vision_model_builder = VisionModelBuilder()


def build_project_model_from_inference(
    inference: InferenceOutput,
    *,
    project_name: str,
    fallback_type: str = "cabinet",
    material_thickness: float = 18.0,
) -> ProjectModel:
    return _default_vision_model_builder.build_project_model_from_inference(
        inference,
        project_name=project_name,
        fallback_type=fallback_type,
        material_thickness=material_thickness,
    )
