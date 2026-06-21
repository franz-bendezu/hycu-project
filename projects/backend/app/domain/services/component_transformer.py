from __future__ import annotations

from dataclasses import dataclass

from app.presentation.schemas.project_design import Component, ComponentKind, HardwareMountFace, JointSpec, ProductSpec


def _component_id_from_face_id(face_id: str) -> str:
    return face_id.split(":", 1)[0]


def _avg_child_joint_offsets(joints: list[JointSpec]) -> dict[str, tuple[float, float]]:
    sums: dict[str, list[float]] = {}
    for joint in joints:
        child_id = _component_id_from_face_id(joint.child_face_id)
        row = sums.setdefault(child_id, [0.0, 0.0, 0.0])
        row[0] += joint.offset_u
        row[1] += joint.offset_v
        row[2] += 1.0

    result: dict[str, tuple[float, float]] = {}
    for component_id, row in sums.items():
        count = row[2]
        if count <= 0:
            continue
        result[component_id] = (
            row[0] / count,
            row[1] / count,
        )
    return result


def _face_from_joint(joint: JointSpec) -> HardwareMountFace | None:
    try:
        return HardwareMountFace(joint.child_face_id.split(":")[1])
    except (ValueError, IndexError):
        return None


def _face_from_face_id(face_id: str) -> HardwareMountFace | None:
    try:
        return HardwareMountFace(face_id.split(":")[1])
    except (ValueError, IndexError):
        return None


def _face_center(component: Component, face: HardwareMountFace) -> tuple[float, float, float]:
    if face == HardwareMountFace.POS_X:
        return (component.pos_x + component.width / 2, component.pos_y, component.pos_z)
    if face == HardwareMountFace.NEG_X:
        return (component.pos_x - component.width / 2, component.pos_y, component.pos_z)
    if face == HardwareMountFace.POS_Y:
        return (component.pos_x, component.pos_y + component.height / 2, component.pos_z)
    if face == HardwareMountFace.NEG_Y:
        return (component.pos_x, component.pos_y - component.height / 2, component.pos_z)
    if face == HardwareMountFace.POS_Z:
        return (component.pos_x, component.pos_y, component.pos_z + component.depth / 2)
    return (component.pos_x, component.pos_y, component.pos_z - component.depth / 2)


def _face_normal(face: HardwareMountFace) -> tuple[float, float, float]:
    if face == HardwareMountFace.POS_X:
        return (1.0, 0.0, 0.0)
    if face == HardwareMountFace.NEG_X:
        return (-1.0, 0.0, 0.0)
    if face == HardwareMountFace.POS_Y:
        return (0.0, 1.0, 0.0)
    if face == HardwareMountFace.NEG_Y:
        return (0.0, -1.0, 0.0)
    if face == HardwareMountFace.POS_Z:
        return (0.0, 0.0, 1.0)
    return (0.0, 0.0, -1.0)


def _face_uv_axes(face: HardwareMountFace) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if face in {HardwareMountFace.POS_Z, HardwareMountFace.NEG_Z}:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    if face in {HardwareMountFace.POS_X, HardwareMountFace.NEG_X}:
        return ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    return ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def _child_center_from_parent_joint(
    parent: Component,
    child: Component,
    parent_face: HardwareMountFace,
    child_face: HardwareMountFace,
    offset_u: float,
    offset_v: float,
    clearance: float,
) -> tuple[float, float, float]:
    px, py, pz = _face_center(parent, parent_face)
    nx, ny, nz = _face_normal(parent_face)
    (ux, uy, uz), (vx, vy, vz) = _face_uv_axes(parent_face)

    target_x = px + (ux * offset_u) + (vx * offset_v) + (nx * clearance)
    target_y = py + (uy * offset_u) + (vy * offset_v) + (ny * clearance)
    target_z = pz + (uz * offset_u) + (vz * offset_v) + (nz * clearance)

    if child_face == HardwareMountFace.POS_X:
        return (target_x - child.width / 2, target_y, target_z)
    if child_face == HardwareMountFace.NEG_X:
        return (target_x + child.width / 2, target_y, target_z)
    if child_face == HardwareMountFace.POS_Y:
        return (target_x, target_y - child.height / 2, target_z)
    if child_face == HardwareMountFace.NEG_Y:
        return (target_x, target_y + child.height / 2, target_z)
    if child_face == HardwareMountFace.POS_Z:
        return (target_x, target_y, target_z - child.depth / 2)
    return (target_x, target_y, target_z + child.depth / 2)


def _canonical_position(
    component: Component,
    product: ProductSpec,
    kind_index: dict[ComponentKind, int],
    kind_counts: dict[ComponentKind, int],
) -> tuple[float, float, float]:
    w = product.target_width
    h = product.target_height
    d = product.target_depth
    t = product.material_thickness

    if component.kind == ComponentKind.LEFT_SIDE:
        return (-w / 2 + component.width / 2, 0.0, 0.0)
    if component.kind == ComponentKind.RIGHT_SIDE:
        return (w / 2 - component.width / 2, 0.0, 0.0)
    if component.kind == ComponentKind.TOP_PANEL:
        return (0.0, h / 2 - component.height / 2, 0.0)
    if component.kind == ComponentKind.BOTTOM_PANEL:
        return (0.0, -h / 2 + component.height / 2, 0.0)
    if component.kind == ComponentKind.BACK_PANEL:
        return (0.0, 0.0, -d / 2 + component.depth / 2)

    if component.kind == ComponentKind.DIVIDER_PANEL:
        divider_count = max(kind_counts.get(ComponentKind.DIVIDER_PANEL, 1), 1)
        idx = kind_index.get(ComponentKind.DIVIDER_PANEL, 0)
        inner_width = max(w - t * 2, t * 2)
        spacing = inner_width / (divider_count + 1)
        x_pos = -inner_width / 2 + spacing * (idx + 1)
        return (x_pos, 0.0, 0.0)

    if component.kind == ComponentKind.SHELF:
        shelf_count = max(kind_counts.get(ComponentKind.SHELF, 1), 1)
        idx = kind_index.get(ComponentKind.SHELF, 0)
        y_pos = ((idx + 1) / (shelf_count + 1)) * (h - t) - h / 2
        return (0.0, y_pos, 0.0)

    if component.kind in {ComponentKind.DOOR_PANEL, ComponentKind.FRONT_PANEL}:
        count = max(kind_counts.get(component.kind, 1), 1)
        idx = kind_index.get(component.kind, 0)
        inner_width = max(w - t * 2, t * 3)
        x_pos = (((idx + 0.5) / count) - 0.5) * inner_width
        z_pos = d / 2 - max(component.depth, t / 2) / 2
        return (x_pos, 0.0, z_pos)

    if component.kind == ComponentKind.DRAWER_FRONT:
        count = max(kind_counts.get(ComponentKind.DRAWER_FRONT, 1), 1)
        idx = kind_index.get(ComponentKind.DRAWER_FRONT, 0)
        front_zone_height = h * 0.45
        fallback_h = front_zone_height / count
        y_pos = -h / 2 + t * 1.8 + fallback_h * (idx + 0.5)
        z_pos = d / 2 - max(component.depth, t / 2) / 2
        return (0.0, y_pos, z_pos)

    if component.kind in {
        ComponentKind.LEFT_LEG_FRONT,
        ComponentKind.LEFT_LEG_BACK,
        ComponentKind.RIGHT_LEG_FRONT,
        ComponentKind.RIGHT_LEG_BACK,
    }:
        x_pos = -w / 2 + component.width / 2 + t
        if component.kind in {ComponentKind.RIGHT_LEG_FRONT, ComponentKind.RIGHT_LEG_BACK}:
            x_pos = w / 2 - component.width / 2 - t
        z_pos = d / 2 - component.depth / 2 - t
        if component.kind in {ComponentKind.LEFT_LEG_BACK, ComponentKind.RIGHT_LEG_BACK}:
            z_pos = -d / 2 + component.depth / 2 + t
        y_pos = -h / 2 + component.height / 2
        return (x_pos, y_pos, z_pos)

    return (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class JointPlacementCandidate:
    child_id: str
    x: float
    y: float
    z: float


class ComponentTransformSolver:
    def __init__(self, *, product: ProductSpec, components: list[Component], joints: list[JointSpec]) -> None:
        self.product = product
        self.components = components
        self.joints = joints
        self.by_id = {component.id: component for component in components}

    def run(self) -> None:
        self._seed_canonical_transforms()
        self._apply_joint_offset_hints()
        self._apply_face_constraints()

    @staticmethod
    def _set_pose(component: Component, x: float, y: float, z: float) -> None:
        component.pos_x = round(x, 3)
        component.pos_y = round(y, 3)
        component.pos_z = round(z, 3)
        component.rot_x_deg = 0.0
        component.rot_y_deg = 0.0
        component.rot_z_deg = 0.0

    def _seed_canonical_transforms(self) -> None:
        kind_counts: dict[ComponentKind, int] = {}
        for component in self.components:
            kind_counts[component.kind] = kind_counts.get(component.kind, 0) + 1

        kind_seen: dict[ComponentKind, int] = {}
        for component in self.components:
            kind_idx = kind_seen.get(component.kind, 0)
            kind_seen[component.kind] = kind_idx + 1

            x, y, z = _canonical_position(component, self.product, {component.kind: kind_idx}, kind_counts)
            self._set_pose(component, x, y, z)

    def _apply_joint_offset_hints(self) -> None:
        joint_offsets = _avg_child_joint_offsets(self.joints)
        child_face_by_component: dict[str, HardwareMountFace] = {}
        for joint in self.joints:
            component_id = _component_id_from_face_id(joint.child_face_id)
            if component_id in child_face_by_component:
                continue
            face = _face_from_joint(joint)
            if face is not None:
                child_face_by_component[component_id] = face

        for component in self.components:
            offsets = joint_offsets.get(component.id)
            if offsets is None:
                continue

            offset_u, offset_v = offsets
            face = child_face_by_component.get(component.id)
            x = component.pos_x
            y = component.pos_y
            z = component.pos_z

            if face in {HardwareMountFace.POS_Z, HardwareMountFace.NEG_Z}:
                x += offset_u
                y += offset_v
            elif face in {HardwareMountFace.POS_X, HardwareMountFace.NEG_X}:
                z += offset_u
                y += offset_v
            elif face in {HardwareMountFace.POS_Y, HardwareMountFace.NEG_Y}:
                x += offset_u
                z += offset_v

            self._set_pose(component, x, y, z)

    def _build_joint_candidates(self) -> dict[str, list[JointPlacementCandidate]]:
        candidates: dict[str, list[JointPlacementCandidate]] = {}

        for joint in self.joints:
            parent_id = _component_id_from_face_id(joint.parent_face_id)
            child_id = _component_id_from_face_id(joint.child_face_id)
            parent = self.by_id.get(parent_id)
            child = self.by_id.get(child_id)
            if parent is None or child is None or parent_id == child_id:
                continue

            parent_face = _face_from_face_id(joint.parent_face_id)
            child_face = _face_from_face_id(joint.child_face_id)
            if parent_face is None or child_face is None:
                continue

            x, y, z = _child_center_from_parent_joint(
                parent,
                child,
                parent_face,
                child_face,
                joint.offset_u,
                joint.offset_v,
                max(0.0, joint.clearance),
            )
            candidates.setdefault(child.id, []).append(JointPlacementCandidate(child_id=child.id, x=x, y=y, z=z))

        return candidates

    def _apply_face_constraints(self) -> None:
        candidates_by_child = self._build_joint_candidates()
        for child_id, candidates in candidates_by_child.items():
            if not candidates:
                continue

            child = self.by_id.get(child_id)
            if child is None:
                continue

            count = float(len(candidates))
            avg_x = sum(item.x for item in candidates) / count
            avg_y = sum(item.y for item in candidates) / count
            avg_z = sum(item.z for item in candidates) / count
            self._set_pose(child, avg_x, avg_y, avg_z)


def assign_component_transforms(
    *,
    product: ProductSpec,
    components: list[Component],
    joints: list[JointSpec],
) -> None:
    solver = ComponentTransformSolver(product=product, components=components, joints=joints)
    solver.run()
