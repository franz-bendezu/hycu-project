from __future__ import annotations

import uuid

from app.domain import ComponentSpec, HardwareSpec, ProductSpec, ProjectDesign
from app.presentation.schemas.project_design import FeatureSpec, JointSpec, MaterialSpec
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
            component.length_formula = f"{component.height:.1f}"
            component.width_formula = f"{component.width:.1f}"

        joints = self._build_joints(spec, components)
        features = self._build_features(joints)
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
        return self.generate(spec)

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
        components = [
            ComponentSpec(
                id="left_side",
                kind="left_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="right_side",
                kind="right_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="top",
                kind="top_panel",
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="bottom",
                kind="bottom_panel",
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="back_panel",
                kind="back_panel",
                width=max(inner_width, 1),
                height=max(spec.target_height - (2 * spec.material_thickness), 1),
                depth=spec.material_thickness,
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

    def _build_shelf_components(self, spec: ProductSpec) -> list[ComponentSpec]:
        # Open shelf variant: no full back panel by default, more shelf levels.
        inner_width = spec.target_width - (2 * spec.material_thickness)
        shelf_depth = max(spec.target_depth - 2, 1)
        components = [
            ComponentSpec(
                id="left_side",
                kind="left_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="right_side",
                kind="right_side",
                width=spec.material_thickness,
                height=spec.target_height,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="top",
                kind="top_panel",
                width=inner_width,
                height=spec.material_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="bottom",
                kind="bottom_panel",
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
                kind="top_panel",
                width=spec.target_width,
                height=top_thickness,
                depth=spec.target_depth,
            ),
            ComponentSpec(
                id="left_leg_front",
                kind="left_leg_front",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="right_leg_front",
                kind="right_leg_front",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="left_leg_back",
                kind="left_leg_back",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="right_leg_back",
                kind="right_leg_back",
                width=leg_width,
                height=leg_height,
                depth=leg_depth,
            ),
            ComponentSpec(
                id="back_apron",
                kind="back_panel",
                width=span_width,
                height=max(spec.material_thickness * 2, 36),
                depth=spec.material_thickness,
            ),
            ComponentSpec(
                id="front_apron",
                kind="front_panel",
                width=span_width,
                height=max(spec.material_thickness * 2, 36),
                depth=spec.material_thickness,
            ),
        ]

    def _build_hardware(self, spec: ProductSpec) -> list[HardwareSpec]:
        def _hardware(code: str, qty: int, joint_type: str) -> HardwareSpec:
            lower = code.lower()
            return HardwareSpec(
                id=f"hardware_{lower}",
                code=code,
                qty=qty,
                mesh_path=f"assets/hardware/{lower}.glb",
                svg_path=f"assets/hardware/{lower}.svg",
                joint_type=joint_type,
            )

        profile = self._profile_for_spec(spec)
        if profile == "desk":
            return [
                _hardware(code="CORNER_BRACKET_40", qty=8, joint_type="bracket"),
                _hardware(code="WOOD_SCREW_4X40", qty=24, joint_type="screw"),
            ]
        if profile == "shelf":
            return [
                _hardware(code="CAM_LOCK_15MM", qty=max(12, spec.shelf_count * 4), joint_type="cam_lock"),
                _hardware(code="WOOD_SCREW_4X40", qty=max(20, spec.shelf_count * 8), joint_type="screw"),
            ]
        return [
            _hardware(code="CAM_LOCK_15MM", qty=max(8, spec.shelf_count * 4), joint_type="cam_lock"),
            _hardware(code="WOOD_SCREW_4X40", qty=max(16, spec.shelf_count * 8), joint_type="screw"),
        ]

    def _build_joints(self, spec: ProductSpec, components: list[ComponentSpec]) -> list[JointSpec]:
        by_id = {component.id: component for component in components}
        joints: list[JointSpec] = []

        def add_joint(parent_id: str, child_id: str, pos_x: float, pos_y: float, pos_z: float) -> None:
            if parent_id not in by_id or child_id not in by_id:
                return
            joints.append(
                JointSpec(
                    parent_id=parent_id,
                    child_id=child_id,
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

        add_joint("left_side", "top", 0.0, spec.target_height / 2 - spec.material_thickness / 2, 0.0)
        add_joint("right_side", "top", 0.0, spec.target_height / 2 - spec.material_thickness / 2, 0.0)
        add_joint("left_side", "bottom", 0.0, -spec.target_height / 2 + spec.material_thickness / 2, 0.0)
        add_joint("right_side", "bottom", 0.0, -spec.target_height / 2 + spec.material_thickness / 2, 0.0)

        if "back_panel" in by_id:
            add_joint("left_side", "back_panel", 0.0, 0.0, -spec.target_depth / 2 + spec.material_thickness / 2)
            add_joint("right_side", "back_panel", 0.0, 0.0, -spec.target_depth / 2 + spec.material_thickness / 2)

        shelf_ids = [component.id for component in components if component.id.startswith("shelf_")]
        for shelf_id in shelf_ids:
            add_joint("left_side", shelf_id, 0.0, 0.0, 0.0)
            add_joint("right_side", shelf_id, 0.0, 0.0, 0.0)

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
