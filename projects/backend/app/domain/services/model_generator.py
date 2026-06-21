from __future__ import annotations

import uuid
from importlib import import_module

from app.domain import ComponentSpec, HardwareSpec, ProductSpec, ProjectDesign
from app.domain.services.product_types import PROFILE_PRODUCT_TYPES, generator_profile_for_inferred_type
from app.presentation.schemas.project_design import (
    ComponentCategory,
    ComponentKind,
    FaceSpec,
    FeatureSpec,
    HardwareAnchor,
    HardwareMountFace,
    HardwareMountTarget,
    JointRule,
    JointSpec,
    MaterialSpec,
)
from app.domain.services.rule_validator import ProjectRuleValidator, ValidationResult


def _assign_component_transforms(product: ProductSpec, components: list[ComponentSpec], joints: list[JointSpec]) -> None:
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


def _set_component_faces(components: list[ComponentSpec]) -> None:
    for component in components:
        component.faces = [
            FaceSpec(id=_face_id(component.id, face), component_id=component.id, normal=face)
            for face in FACE_ORDER
        ]


class ModelGenerator:
    def __init__(self) -> None:
        self.validator = ProjectRuleValidator()

    def generate(self, spec: ProductSpec) -> ProjectDesign:
        if not getattr(spec, "id", None):
            spec.id = str(uuid.uuid4())
        if not getattr(spec, "sku", None):
            spec.sku = f"SKU-{spec.id[:8]}"

        default_material = MaterialSpec(
            id="material_board_default",
            thickness_mm=spec.material_thickness,
            texture_map_url=None,
        )

        components = self._build_components(spec)
        _set_component_faces(components)
        for component in components:
            component.material_id = default_material.id
            component.category = self._category_for_kind(component.kind)

        joints = self._build_joints(spec, components)
        _assign_component_transforms(product=spec, components=components, joints=joints)
        features = self._build_features(joints)
        self._assign_uuid_component_ids(components, joints, features)
        hardware = self._build_hardware(spec, components)
        warnings = self._generate_warnings(spec)
        return ProjectDesign(
            product=spec,
            materials=[default_material],
            components=components,
            hardware=hardware,
            joints=joints,
            features=features,
            warnings=warnings,
        )

    def validate(self, model: ProjectDesign) -> ValidationResult:
        return self.validator.validate(model)

    def apply_update(self, model: ProjectDesign, **updates: float) -> ProjectDesign:
        spec = model.product.model_copy(deep=True)
        for key, value in updates.items():
            if value is not None:
                setattr(spec, key, value)

        components = self._build_components(spec)
        _set_component_faces(components)
        for component in components:
            component.material_id = model.materials[0].id if model.materials else "material_board_default"
            component.category = self._category_for_kind(component.kind)

        joints = self._build_joints(spec, components)
        _assign_component_transforms(product=spec, components=components, joints=joints)
        features = self._build_features(joints)
        self._assign_uuid_component_ids(components, joints, features)
        hardware = self._build_hardware(spec, components)
        warnings = self._generate_warnings(spec)

        return ProjectDesign(
            product=spec,
            materials=model.materials,
            components=components,
            hardware=hardware,
            joints=joints,
            features=features,
            warnings=warnings,
        )

    def _assign_uuid_component_ids(
        self,
        components: list[ComponentSpec],
        joints: list[JointSpec],
        features: list[FeatureSpec],
    ) -> None:
        old_face_index: dict[str, tuple[str, HardwareMountFace]] = {
            face.id: (component.id, face.normal)
            for component in components
            for face in component.faces
        }

        id_map: dict[str, str] = {}
        for component in components:
            original_id = component.id
            uuid_id = str(uuid.uuid4())
            id_map[original_id] = uuid_id
            component.id = uuid_id
            component.faces = [
                FaceSpec(
                    id=_face_id(uuid_id, face.normal),
                    component_id=uuid_id,
                    normal=face.normal,
                )
                for face in component.faces
            ]

        new_face_index: dict[tuple[str, HardwareMountFace], str] = {
            (component.id, face.normal): face.id
            for component in components
            for face in component.faces
        }

        for joint in joints:
            parent_component_id, parent_face = old_face_index.get(
                joint.parent_face_id,
                (
                    _component_id_from_face_id(joint.parent_face_id),
                    HardwareMountFace(joint.parent_face_id.split(":")[1]),
                ),
            )
            child_component_id, child_face = old_face_index.get(
                joint.child_face_id,
                (
                    _component_id_from_face_id(joint.child_face_id),
                    HardwareMountFace(joint.child_face_id.split(":")[1]),
                ),
            )
            mapped_parent_id = id_map.get(parent_component_id, parent_component_id)
            mapped_child_id = id_map.get(child_component_id, child_component_id)
            joint.parent_face_id = new_face_index[(mapped_parent_id, parent_face)]
            joint.child_face_id = new_face_index[(mapped_child_id, child_face)]
            joint.id = str(uuid.uuid4())

        for feature in features:
            feature.component_id = id_map.get(feature.component_id, feature.component_id)

    def _category_for_kind(self, kind: ComponentKind) -> ComponentCategory:
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

    def _build_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        profile = self._profile_for_spec(spec)
        if profile == "desk":
            return self._build_desk_components(spec)
        if profile == "shelf":
            return self._build_shelf_components(spec)
        return self._build_cabinet_components(spec)

    def _profile_for_spec(self, spec: ProductSpec) -> str:
        profile = generator_profile_for_inferred_type(getattr(spec, "inferred_type", ""))
        if profile is not None:
            return profile
        supported = ",".join(f"'{name}'" for name in PROFILE_PRODUCT_TYPES)
        raise ValueError(f"Product spec must include inferred_type in {{{supported}}}")

    def _build_cabinet_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        inner_width = spec.target_width - (2 * spec.material_thickness)
        shelf_depth = max(spec.target_depth - 2, 1)
        usable_front_height = max(spec.target_height - (2 * spec.material_thickness), spec.material_thickness)
        mixed_four_front_layout = spec.door_count == 3 and spec.drawer_count == 1
        components = [
            ComponentSpec(
                id="left_side",
                kind=ComponentKind.LEFT_SIDE,
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="right_side",
                kind=ComponentKind.RIGHT_SIDE,
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="top",
                kind=ComponentKind.TOP_PANEL,
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="bottom",
                kind=ComponentKind.BOTTOM_PANEL,
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="back_panel",
                kind=ComponentKind.BACK_PANEL,
                width=max(inner_width, 1),
                height=max(spec.target_height - (2 * spec.material_thickness), 1),
                depth=spec.material_thickness,
            ),
        ]

        for idx in range(spec.shelf_count):
            components.append(
                ComponentSpec(
                    id=f"shelf_{idx + 1}",
                    kind=ComponentKind.SHELF,
                    width=inner_width,
                    height=spec.material_thickness,
                    depth=shelf_depth,
                )
            )

        if spec.door_count > 0:
            if mixed_four_front_layout:
                left_column_width = max(min(inner_width * 0.58, inner_width - spec.material_thickness * 2), spec.material_thickness * 4)
                right_column_width = max(inner_width - left_column_width, spec.material_thickness * 3)
                top_left_door_height = max(usable_front_height * 0.32, spec.material_thickness * 3)
                bottom_left_door_height = max(usable_front_height * 0.45, spec.material_thickness * 3)
                right_door_height = max(usable_front_height * 0.62, spec.material_thickness * 3)

                components.append(
                    ComponentSpec(
                        id="door_panel_1",
                        kind=ComponentKind.DOOR_PANEL,
                        width=left_column_width,
                        height=top_left_door_height,
                        depth=max(spec.material_thickness / 2, 1),
                    )
                )
                components.append(
                    ComponentSpec(
                        id="door_panel_2",
                        kind=ComponentKind.DOOR_PANEL,
                        width=bottom_left_door_height and left_column_width,
                        height=bottom_left_door_height,
                        depth=max(spec.material_thickness / 2, 1),
                    )
                )
                components.append(
                    ComponentSpec(
                        id="door_panel_3",
                        kind=ComponentKind.DOOR_PANEL,
                        width=right_column_width,
                        height=right_door_height,
                        depth=max(spec.material_thickness / 2, 1),
                    )
                )
            else:
                door_width = max(inner_width / max(spec.door_count, 1), spec.material_thickness * 2)
                for idx in range(spec.door_count):
                    components.append(
                        ComponentSpec(
                            id=f"door_panel_{idx + 1}",
                            kind=ComponentKind.DOOR_PANEL,
                            width=door_width,
                            height=usable_front_height,
                            depth=max(spec.material_thickness / 2, 1),
                        )
                    )

        if spec.divider_count > 0:
            divider_height = max(spec.target_height - (2 * spec.material_thickness), 1)
            divider_depth = max(spec.target_depth - 2, 1)
            for idx in range(spec.divider_count):
                components.append(
                    ComponentSpec(
                        id=f"divider_panel_{idx + 1}",
                        kind=ComponentKind.DIVIDER_PANEL,
                        width=spec.material_thickness,
                        height=divider_height,
                        depth=divider_depth,
                    )
                )

        if spec.drawer_count > 0:
            if mixed_four_front_layout:
                left_column_width = max(min(inner_width * 0.58, inner_width - spec.material_thickness * 2), spec.material_thickness * 4)
                drawer_height = max(usable_front_height * 0.16, spec.material_thickness * 2)
                components.append(
                    ComponentSpec(
                        id="drawer_front_1",
                        kind=ComponentKind.DRAWER_FRONT,
                        width=left_column_width,
                        height=drawer_height,
                        depth=max(spec.material_thickness / 2, 1),
                    )
                )
            else:
                drawer_height = max(
                    min(usable_front_height / max(spec.drawer_count + 1, 1), spec.target_height * 0.2),
                    spec.material_thickness * 2,
                )
                drawer_width = max(inner_width - (spec.material_thickness * 0.75), spec.material_thickness * 3)
                for idx in range(spec.drawer_count):
                    components.append(
                        ComponentSpec(
                            id=f"drawer_front_{idx + 1}",
                            kind=ComponentKind.DRAWER_FRONT,
                            width=drawer_width,
                            height=drawer_height,
                            depth=max(spec.material_thickness / 2, 1),
                        )
                    )

        return components

    def _build_shelf_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        # Open shelf variant: no full back panel by default, more shelf levels.
        inner_width = spec.target_width - (2 * spec.material_thickness)
        shelf_depth = max(spec.target_depth - 2, 1)
        components = [
            ComponentSpec(
                id="left_side",
                kind=ComponentKind.LEFT_SIDE,
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="right_side",
                kind=ComponentKind.RIGHT_SIDE,
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="top",
                kind=ComponentKind.TOP_PANEL,
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="bottom",
                kind=ComponentKind.BOTTOM_PANEL,
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
        ]

        for idx in range(spec.shelf_count):
            components.append(
                ComponentSpec(
                    id=f"shelf_{idx + 1}",
                    kind="shelf",
                    width=inner_width,
                    height=spec.material_thickness,
                    depth=shelf_depth,
                )
            )
        return components

    def _build_desk_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        top_thickness = spec.material_thickness
        leg_width = max(spec.material_thickness * 1.4, 40)
        leg_depth = max(spec.material_thickness * 1.4, 40)
        leg_height = max(spec.target_height - top_thickness, 1)
        inset_x = max(spec.material_thickness * 1.8, 30)
        inset_z = max(spec.material_thickness * 1.2, 20)
        span_width = max(spec.target_width - (2 * inset_x + 2 * leg_width), 80)

        return [
            ComponentSpec(
                id="top",
                kind=ComponentKind.TOP_PANEL,
                width=spec.target_width,
                height=top_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="left_leg_front",
                kind=ComponentKind.LEFT_LEG_FRONT,
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="right_leg_front",
                kind=ComponentKind.RIGHT_LEG_FRONT,
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="left_leg_back",
                kind=ComponentKind.LEFT_LEG_BACK,
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="right_leg_back",
                kind=ComponentKind.RIGHT_LEG_BACK,
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="back_apron",
                kind=ComponentKind.BACK_PANEL,
                width=span_width,
                height=max(spec.material_thickness * 2, 36),
                depth=spec.material_thickness,
            ),
            ComponentSpec(
                id="front_apron",
                kind=ComponentKind.FRONT_PANEL,
                width=span_width,
                height=max(spec.material_thickness * 2, 36),
                depth=spec.material_thickness,
            ),
        ]

    def _build_hardware(self, spec: ProductSpec, components: list[ComponentSpec]) -> list[HardwareSpec]:
        door_components = [component for component in components if component.kind == ComponentKind.DOOR_PANEL]
        drawer_components = [component for component in components if component.kind == ComponentKind.DRAWER_FRONT]
        front_components = [component for component in components if component.kind == ComponentKind.FRONT_PANEL]
        front_like_components = [
            component
            for component in components
            if component.kind in {ComponentKind.DOOR_PANEL, ComponentKind.DRAWER_FRONT, ComponentKind.FRONT_PANEL}
        ]

        def _preferred_components(anchor: HardwareAnchor) -> list[ComponentSpec]:
            if anchor == HardwareAnchor.CAM_LOCK_15MM and (drawer_components or door_components):
                return drawer_components or door_components
            if anchor == HardwareAnchor.CORNER_BRACKET_40 and front_components:
                return front_components
            if front_like_components:
                return front_like_components
            return []

        def _default_mount_targets(anchor: HardwareAnchor, qty: int) -> list[HardwareMountTarget]:
            targets: list[HardwareMountTarget] = []
            preferred = _preferred_components(anchor)
            if not preferred:
                return targets

            def _offset(component: ComponentSpec, component_idx: int, local_idx: int) -> tuple[float, float, float]:
                width = max(component.width, 1.0)
                height = max(component.height, 1.0)
                depth = max(component.depth, 1.0)

                x_inset = max(10.0, min(85.0, width * 0.38))
                y_inset = max(16.0, min(220.0, height * 0.35))
                z_local = round(depth * 0.5, 3)

                if anchor.name == "HINGE_SOFT_CLOSE_110":
                    side = -1.0 if component_idx % 2 == 0 else 1.0
                    y_slots = [-y_inset, y_inset, 0.0]
                    return (round(side * x_inset, 3), round(y_slots[local_idx % len(y_slots)], 3), z_local)

                if anchor == HardwareAnchor.CAM_LOCK_15MM:
                    corners = [(-x_inset, -y_inset), (x_inset, -y_inset), (-x_inset, y_inset), (x_inset, y_inset)]
                    x, y = corners[local_idx % len(corners)]
                    return (round(x, 3), round(y, 3), z_local)

                if anchor == HardwareAnchor.WOOD_SCREW_4X40:
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

                broad = [(-x_inset, -y_inset), (x_inset, -y_inset), (-x_inset, y_inset), (x_inset, y_inset), (0.0, 0.0)]
                x, y = broad[local_idx % len(broad)]
                return (round(x, 3), round(y, 3), z_local)

            for idx in range(max(qty, 0)):
                component_idx = idx % len(preferred)
                component = preferred[component_idx]
                local_x, local_y, local_z = _offset(component, component_idx=component_idx, local_idx=idx // len(preferred))
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

        def _hardware(anchor: HardwareAnchor, qty: int) -> HardwareSpec:
            code = anchor.name
            lower = code.lower()
            return HardwareSpec(
                id=f"hardware_{lower}",
                code=code,
                qty=qty,
                anchor=anchor,
                mesh_path=f"assets/hardware/{lower}.glb",
                svg_path=f"assets/hardware/{lower}.svg",
                mount_targets=_default_mount_targets(anchor, qty),
            )

        profile = self._profile_for_spec(spec)
        if profile == "desk":
            return [
                _hardware(anchor=HardwareAnchor.CORNER_BRACKET_40, qty=8),
                _hardware(anchor=HardwareAnchor.WOOD_SCREW_4X40, qty=24),
            ]

        # Common for cabinet and shelf
        shelf_hardware_qty = spec.shelf_count * 4
        cam_lock_qty = 8  # Top/Bottom panels

        hardware_list = [
            _hardware(anchor=HardwareAnchor.CAM_LOCK_15MM, qty=cam_lock_qty),
            _hardware(anchor=HardwareAnchor.SHELF_PIN_5MM, qty=shelf_hardware_qty),
            _hardware(
                anchor=HardwareAnchor.WOOD_SCREW_4X40,
                qty=(cam_lock_qty + shelf_hardware_qty) * 2,
            ),
        ]
        return hardware_list

    def _build_joints(self, spec: ProductSpec, components: list[ComponentSpec]) -> list[JointSpec]:
        by_id = {component.id: component for component in components}
        joints: list[JointSpec] = []
        face_index = {
            (component.id, face.normal): face.id
            for component in components
            for face in component.faces
        }

        def add_joint(
            parent_id: str,
            child_id: str,
            parent_face: HardwareMountFace,
            child_face: HardwareMountFace,
            offset_u: float = 0.0,
            offset_v: float = 0.0,
            rule: str | None = None,
        ) -> None:
            if parent_id not in by_id or child_id not in by_id:
                return
            joints.append(
                JointSpec(
                    id=str(uuid.uuid4()),
                    parent_face_id=face_index[(parent_id, parent_face)],
                    child_face_id=face_index[(child_id, child_face)],
                    joint_rule=rule,
                    offset_u=round(offset_u, 3),
                    offset_v=round(offset_v, 3),
                    clearance=0.0,
                )
            )

        profile = self._profile_for_spec(spec)
        if profile == "desk":
            leg_ids = [
                "left_leg_front",
                "right_leg_front",
                "left_leg_back",
                "right_leg_back",
            ]
            for leg_id in leg_ids:
                add_joint("top", leg_id, HardwareMountFace.NEG_Y, HardwareMountFace.POS_Y)
            add_joint("top", "front_apron", HardwareMountFace.NEG_Y, HardwareMountFace.POS_Y)
            add_joint("top", "back_apron", HardwareMountFace.NEG_Y, HardwareMountFace.POS_Y)
            return joints

        add_joint(
            "left_side",
            "top",
            HardwareMountFace.POS_Y,
            HardwareMountFace.NEG_X,
            rule=JointRule.OVERLAP,
        )
        add_joint(
            "right_side",
            "top",
            HardwareMountFace.POS_Y,
            HardwareMountFace.POS_X,
            rule=JointRule.OVERLAP,
        )
        add_joint(
            "left_side",
            "bottom",
            HardwareMountFace.NEG_Y,
            HardwareMountFace.NEG_X,
            rule=JointRule.INSET,
        )
        add_joint(
            "right_side",
            "bottom",
            HardwareMountFace.NEG_Y,
            HardwareMountFace.POS_X,
            rule=JointRule.INSET,
        )

        if "back_panel" in by_id:
            add_joint(
                "left_side",
                "back_panel",
                HardwareMountFace.NEG_Z,
                HardwareMountFace.NEG_X,
                rule=JointRule.FLUSH_BACK,
            )
            add_joint(
                "right_side",
                "back_panel",
                HardwareMountFace.NEG_Z,
                HardwareMountFace.POS_X,
                rule=JointRule.FLUSH_BACK,
            )

        shelf_ids = [component.id for component in components if component.id.startswith("shelf_")]
        shelf_count = len(shelf_ids)
        usable_height = max(spec.target_height - (2 * spec.material_thickness), spec.material_thickness)
        for idx, shelf_id in enumerate(shelf_ids):
            shelf_anchor_y = ((0.5 - ((idx + 1) / (shelf_count + 1))) * usable_height) if shelf_count > 0 else 0.0
            add_joint(
                "left_side",
                shelf_id,
                HardwareMountFace.POS_X,
                HardwareMountFace.NEG_X,
                offset_v=shelf_anchor_y,
                rule=JointRule.BETWEEN,
            )
            add_joint(
                "right_side",
                shelf_id,
                HardwareMountFace.NEG_X,
                HardwareMountFace.POS_X,
                offset_v=shelf_anchor_y,
                rule=JointRule.BETWEEN,
            )

        door_ids = [component.id for component in components if component.id.startswith("door_panel_")]
        door_count = len(door_ids)
        mixed_four_front_layout = spec.door_count == 3 and spec.drawer_count == 1 and door_count == 3
        divider_ids = [component.id for component in components if component.id.startswith("divider_panel_")]
        divider_count = len(divider_ids)
        inner_width = max(spec.target_width - (2 * spec.material_thickness), spec.material_thickness)
        if mixed_four_front_layout and divider_count > 0:
            left_column_width = max(min(inner_width * 0.58, inner_width - spec.material_thickness * 2), spec.material_thickness * 4)
            divider_anchor_x = -inner_width / 2 + left_column_width
            for divider_id in divider_ids:
                add_joint(
                    "bottom",
                    divider_id,
                    HardwareMountFace.POS_Y,
                    HardwareMountFace.NEG_Y,
                    offset_u=divider_anchor_x,
                    rule=JointRule.BETWEEN,
                )
        else:
            for idx, divider_id in enumerate(divider_ids):
                divider_anchor_x = (((idx + 1) / (divider_count + 1)) - 0.5) * inner_width if divider_count > 0 else 0.0
                add_joint(
                    "bottom",
                    divider_id,
                    HardwareMountFace.POS_Y,
                    HardwareMountFace.NEG_Y,
                    offset_u=divider_anchor_x,
                    rule=JointRule.BETWEEN,
                )

        if mixed_four_front_layout:
            usable_height = max(spec.target_height - (2 * spec.material_thickness), spec.material_thickness)
            left_column_width = max(min(inner_width * 0.58, inner_width - spec.material_thickness * 2), spec.material_thickness * 4)
            right_column_width = max(inner_width - left_column_width, spec.material_thickness * 3)

            top_door = by_id.get("door_panel_1")
            bottom_left_door = by_id.get("door_panel_2")
            right_door = by_id.get("door_panel_3")
            drawer = by_id.get("drawer_front_1")

            left_center_x = -inner_width / 2 + left_column_width / 2
            right_center_x = inner_width / 2 - right_column_width / 2

            if top_door is not None:
                top_center_y = usable_height / 2 - top_door.height / 2
                add_joint(
                    "top",
                    "door_panel_1",
                    HardwareMountFace.POS_Z,
                    HardwareMountFace.NEG_Z,
                    offset_u=left_center_x,
                    offset_v=top_center_y,
                    rule=JointRule.MOUNT,
                )

            if drawer is not None:
                drawer_center_y = usable_height / 2 - (top_door.height if top_door else usable_height * 0.32) - drawer.height / 2
                add_joint(
                    "bottom",
                    "drawer_front_1",
                    HardwareMountFace.POS_Z,
                    HardwareMountFace.NEG_Z,
                    offset_u=left_center_x,
                    offset_v=drawer_center_y,
                    rule=JointRule.MOUNT,
                )

            if bottom_left_door is not None:
                bottom_left_center_y = -usable_height / 2 + bottom_left_door.height / 2
                add_joint(
                    "left_side",
                    "door_panel_2",
                    HardwareMountFace.POS_Z,
                    HardwareMountFace.NEG_Z,
                    offset_u=left_center_x,
                    offset_v=bottom_left_center_y,
                    rule=JointRule.MOUNT,
                )

            if right_door is not None:
                right_center_y = -usable_height / 2 + right_door.height / 2
                add_joint(
                    "right_side",
                    "door_panel_3",
                    HardwareMountFace.POS_Z,
                    HardwareMountFace.NEG_Z,
                    offset_u=right_center_x,
                    offset_v=right_center_y,
                    rule=JointRule.MOUNT,
                )
        else:
            for idx, door_id in enumerate(door_ids):
                door_anchor_x = (((idx + 0.5) / max(door_count, 1)) - 0.5) * inner_width
                parent_for_door = "left_side" if door_anchor_x < 0 else "right_side"
                add_joint(
                    parent_for_door,
                    door_id,
                    HardwareMountFace.POS_Z,
                    HardwareMountFace.NEG_Z,
                    offset_u=door_anchor_x,
                    rule=JointRule.MOUNT,
                )

        drawer_ids = [component.id for component in components if component.id.startswith("drawer_front_")]
        drawer_count = len(drawer_ids)
        front_zone_height = usable_height * 0.45
        if not mixed_four_front_layout:
            for idx, drawer_id in enumerate(drawer_ids):
                drawer_anchor_y = (
                    (-usable_height / 2 + spec.material_thickness * 1.8)
                    + ((idx + 0.5) / max(drawer_count, 1)) * front_zone_height
                )
                add_joint(
                    "bottom",
                    drawer_id,
                    HardwareMountFace.POS_Z,
                    HardwareMountFace.NEG_Z,
                    offset_v=drawer_anchor_y,
                    rule=JointRule.MOUNT,
                )

        return joints

    def _build_features(self, joints: list[JointSpec]) -> list[FeatureSpec]:
        features: list[FeatureSpec] = []
        for index, joint in enumerate(joints):
            features.append(
                FeatureSpec(
                    component_id=_component_id_from_face_id(joint.child_face_id),
                    face_index=((index % 6) + 1),
                    u_coord=round((index % 4) * 12.5, 2),
                    v_coord=round((index % 3) * 9.5, 2),
                    operation_type="drill_5mm",
                )
            )
        return features

    def _generate_warnings(self, spec: ProductSpec) -> list[str]:
        shelf_spacing = (spec.target_height - (2 * spec.material_thickness)) / max(spec.shelf_count + 1, 1)
        warnings: list[str] = []
        if shelf_spacing > 420:
            warnings.append("Shelf spacing is large; consider adding a divider or extra shelf.")
        return warnings
