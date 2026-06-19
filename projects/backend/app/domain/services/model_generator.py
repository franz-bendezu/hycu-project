from __future__ import annotations

import uuid

from app.domain import ComponentSpec, HardwareSpec, ProductSpec, ProjectDesign
from app.presentation.schemas.project_design import (
    ComponentKind,
    FeatureSpec,
    HardwareAnchor,
    JointRule,
    JointSpec,
    MaterialSpec,
)
from app.domain.services.rule_validator import ProjectRuleValidator, ValidationResult


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
        for component in components:
            component.material_id = default_material.id

        joints = self._build_joints(spec, components)
        features = self._build_features(joints)
        self._assign_uuid_component_ids(components, joints, features)
        hardware = self._build_hardware(spec)
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
        for component in components:
            component.material_id = model.materials[0].id if model.materials else "material_board_default"

        joints = self._build_joints(spec, components)
        features = self._build_features(joints)
        self._assign_uuid_component_ids(components, joints, features)
        hardware = self._build_hardware(spec)
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
        id_map: dict[str, str] = {}
        for component in components:
            original_id = component.id
            uuid_id = str(uuid.uuid4())
            id_map[original_id] = uuid_id
            component.id = uuid_id

        for joint in joints:
            joint.parent_id = id_map.get(joint.parent_id, joint.parent_id)
            joint.child_id = id_map.get(joint.child_id, joint.child_id)

        for feature in features:
            feature.component_id = id_map.get(feature.component_id, feature.component_id)

    def _build_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        profile = self._profile_for_spec(spec)
        if profile == "desk":
            return self._build_desk_components(spec)
        if profile == "shelf":
            return self._build_shelf_components(spec)
        return self._build_cabinet_components(spec)

    def _profile_for_spec(self, spec: ProductSpec) -> str:
        profile = (getattr(spec, "inferred_type", "") or "").strip().lower()
        if profile in {"cabinet", "desk", "shelf"}:
            return profile
        # Legacy fallback for models created before inferred_type existed.
        if spec.shelf_count == 0:
            return "desk"
        if spec.shelf_count >= 4:
            return "shelf"
        return "cabinet"

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

    def _build_hardware(self, spec: ProductSpec) -> list[HardwareSpec]:
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

        def add_joint(
            parent_id: str,
            child_id: str,
            pos_x: float,
            pos_y: float,
            pos_z: float,
            rule: str | None = None,
        ) -> None:
            if parent_id not in by_id or child_id not in by_id:
                return
            joints.append(
                JointSpec(
                    parent_id=parent_id,
                    child_id=child_id,
                    joint_rule=rule,
                    pos_x=round(pos_x, 3),
                    pos_y=round(pos_y, 3),
                    pos_z=round(pos_z, 3),
                    rot_x=0.0,
                    rot_y=0.0,
                    rot_z=0.0,
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
                add_joint("top", leg_id, 0.0, -spec.target_height / 2, 0.0)
            add_joint("top", "front_apron", 0.0, -spec.target_height / 2 + 80, spec.target_depth / 2 - 18)
            add_joint("top", "back_apron", 0.0, -spec.target_height / 2 + 80, -spec.target_depth / 2 + 18)
            return joints

        add_joint(
            "left_side",
            "top",
            0.0,
            spec.target_height / 2 - spec.material_thickness / 2,
            0.0,
            rule=JointRule.OVERLAP,
        )
        add_joint(
            "right_side",
            "top",
            0.0,
            spec.target_height / 2 - spec.material_thickness / 2,
            0.0,
            rule=JointRule.OVERLAP,
        )
        add_joint(
            "left_side",
            "bottom",
            0.0,
            -spec.target_height / 2 + spec.material_thickness / 2,
            0.0,
            rule=JointRule.INSET,
        )
        add_joint(
            "right_side",
            "bottom",
            0.0,
            -spec.target_height / 2 + spec.material_thickness / 2,
            0.0,
            rule=JointRule.INSET,
        )

        if "back_panel" in by_id:
            add_joint(
                "left_side",
                "back_panel",
                0.0,
                0.0,
                -spec.target_depth / 2 + spec.material_thickness / 2,
                rule=JointRule.FLUSH_BACK,
            )
            add_joint(
                "right_side",
                "back_panel",
                0.0,
                0.0,
                -spec.target_depth / 2 + spec.material_thickness / 2,
                rule=JointRule.FLUSH_BACK,
            )

        shelf_ids = [component.id for component in components if component.id.startswith("shelf_")]
        shelf_count = len(shelf_ids)
        usable_height = max(spec.target_height - (2 * spec.material_thickness), spec.material_thickness)
        for idx, shelf_id in enumerate(shelf_ids):
            shelf_anchor_y = ((0.5 - ((idx + 1) / (shelf_count + 1))) * usable_height) if shelf_count > 0 else 0.0
            add_joint("left_side", shelf_id, 0.0, shelf_anchor_y, 0.0, rule=JointRule.BETWEEN)
            add_joint("right_side", shelf_id, 0.0, shelf_anchor_y, 0.0, rule=JointRule.BETWEEN)

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
                add_joint("bottom", divider_id, divider_anchor_x, 0.0, 0.0, rule=JointRule.BETWEEN)
        else:
            for idx, divider_id in enumerate(divider_ids):
                divider_anchor_x = (((idx + 1) / (divider_count + 1)) - 0.5) * inner_width if divider_count > 0 else 0.0
                add_joint("bottom", divider_id, divider_anchor_x, 0.0, 0.0, rule=JointRule.BETWEEN)

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
                    left_center_x,
                    top_center_y,
                    spec.target_depth / 2 - spec.material_thickness / 4,
                    rule=JointRule.MOUNT,
                )

            if drawer is not None:
                drawer_center_y = usable_height / 2 - (top_door.height if top_door else usable_height * 0.32) - drawer.height / 2
                add_joint(
                    "bottom",
                    "drawer_front_1",
                    left_center_x,
                    drawer_center_y,
                    spec.target_depth / 2 - spec.material_thickness / 4,
                    rule=JointRule.MOUNT,
                )

            if bottom_left_door is not None:
                bottom_left_center_y = -usable_height / 2 + bottom_left_door.height / 2
                add_joint(
                    "left_side",
                    "door_panel_2",
                    left_center_x,
                    bottom_left_center_y,
                    spec.target_depth / 2 - spec.material_thickness / 4,
                    rule=JointRule.MOUNT,
                )

            if right_door is not None:
                right_center_y = -usable_height / 2 + right_door.height / 2
                add_joint(
                    "right_side",
                    "door_panel_3",
                    right_center_x,
                    right_center_y,
                    spec.target_depth / 2 - spec.material_thickness / 4,
                    rule=JointRule.MOUNT,
                )
        else:
            for idx, door_id in enumerate(door_ids):
                door_anchor_x = (((idx + 0.5) / max(door_count, 1)) - 0.5) * inner_width
                parent_for_door = "left_side" if door_anchor_x < 0 else "right_side"
                add_joint(
                    parent_for_door,
                    door_id,
                    door_anchor_x,
                    0.0,
                    spec.target_depth / 2 - spec.material_thickness / 4,
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
                    0.0,
                    drawer_anchor_y,
                    spec.target_depth / 2 - spec.material_thickness / 4,
                    rule=JointRule.MOUNT,
                )

        return joints

    def _build_features(self, joints: list[JointSpec]) -> list[FeatureSpec]:
        features: list[FeatureSpec] = []
        for index, joint in enumerate(joints):
            features.append(
                FeatureSpec(
                    component_id=joint.child_id,
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
