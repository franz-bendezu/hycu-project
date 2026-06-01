from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductSpec(BaseModel):
    id: str | None = None
    sku: str | None = None
    name: str = Field(..., min_length=1, max_length=120)
    inferred_type: Literal["cabinet", "desk", "shelf"] = "cabinet"
    target_width: float = Field(default=800, gt=0)
    target_height: float = Field(default=1200, gt=0)
    target_depth: float = Field(default=450, gt=0)
    material_thickness: float = Field(default=18, ge=8, le=50)
    shelf_count: int = Field(default=3, ge=0, le=20)


class Component(BaseModel):
    id: str
    kind: str
    material_id: str | None = None
    length_formula: str | None = None
    width_formula: str | None = None
    width: float
    height: float
    depth: float


class HardwareItem(BaseModel):
    code: str
    qty: int
    id: str | None = None
    mesh_path: str | None = None
    svg_path: str | None = None
    joint_type: str | None = None


class MaterialSpec(BaseModel):
    id: str
    thickness_mm: float = Field(..., gt=0)
    texture_map_url: str | None = None


class JointSpec(BaseModel):
    parent_id: str
    child_id: str
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
