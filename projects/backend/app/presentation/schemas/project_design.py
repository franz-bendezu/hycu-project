from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JointRule(str, Enum):
    OVERLAP = "overlap"
    INSET = "inset"
    BETWEEN = "between"
    FLUSH_BACK = "flush_back"
    MOUNT = "mount"


class ComponentKind(str, Enum):
    LEFT_SIDE = "left_side"
    RIGHT_SIDE = "right_side"
    TOP_PANEL = "top_panel"
    BOTTOM_PANEL = "bottom_panel"
    BACK_PANEL = "back_panel"
    SHELF = "shelf"
    DIVIDER_PANEL = "divider_panel"
    FRONT_PANEL = "front_panel"
    DOOR_PANEL = "door_panel"
    DRAWER_FRONT = "drawer_front"
    LEFT_LEG_FRONT = "left_leg_front"
    RIGHT_LEG_FRONT = "right_leg_front"
    LEFT_LEG_BACK = "left_leg_back"
    RIGHT_LEG_BACK = "right_leg_back"


class ComponentCategory(str, Enum):
    STRUCTURAL = "structural"
    FRONT = "front"
    INTERNAL = "internal"
    SUPPORT = "support"


class HardwareAnchor(str, Enum):
    # Screws & Fasteners
    WOOD_SCREW_4X40 = "WOOD_SCREW_4X40"

    # Brackets
    CORNER_BRACKET_40 = "CORNER_BRACKET_40"

    # Connectors
    CAM_LOCK_15MM = "CAM_LOCK_15MM"

    # Pins & Supports
    SHELF_PIN_5MM = "SHELF_PIN_5MM"


class HardwareMountFace(str, Enum):
    POS_X = "+x"
    NEG_X = "-x"
    POS_Y = "+y"
    NEG_Y = "-y"
    POS_Z = "+z"
    NEG_Z = "-z"


class HardwareMountTarget(BaseModel):
    component_id: str
    face: HardwareMountFace
    local_x: float = 0.0
    local_y: float = 0.0
    local_z: float = 0.0
    normal_offset_mm: float = 2.0


class ProductSpec(BaseModel):
    id: str | None = None
    sku: str | None = None
    name: str = Field(..., min_length=1, max_length=120)
    inferred_type: Literal[
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
    ] = "cabinet"
    target_width: float = Field(default=800, gt=0)
    target_height: float = Field(default=1200, gt=0)
    target_depth: float = Field(default=450, gt=0)
    material_thickness: float = Field(default=18, ge=8, le=50)
    shelf_count: int = Field(default=3, ge=0, le=20)
    divider_count: int = Field(default=0, ge=0, le=8)
    door_count: int = Field(default=0, ge=0, le=12)
    drawer_count: int = Field(default=0, ge=0, le=12)


class Component(BaseModel):
    id: str
    kind: ComponentKind
    category: ComponentCategory = ComponentCategory.STRUCTURAL
    material_id: str | None = None
    width: float
    height: float
    depth: float


class HardwareItem(BaseModel):
    code: str
    qty: int
    id: str | None = None
    anchor: HardwareAnchor | None = None
    mesh_path: str | None = None
    svg_path: str | None = None
    mount_targets: list[HardwareMountTarget] = Field(default_factory=list)


class MaterialSpec(BaseModel):
    id: str
    thickness_mm: float = Field(..., gt=0)
    texture_map_url: str | None = None


class JointSpec(BaseModel):
    parent_id: str
    child_id: str
    joint_rule: JointRule | None = None
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0


class FeatureSpec(BaseModel):
    component_id: str
    face_index: int = Field(..., ge=1, le=6)
    u_coord: float = 0.0
    v_coord: float = 0.0
    operation_type: str


class ProjectModel(BaseModel):
    product: ProductSpec
    materials: list[MaterialSpec] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)
    hardware: list[HardwareItem] = Field(default_factory=list)
    joints: list[JointSpec] = Field(default_factory=list)
    features: list[FeatureSpec] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
