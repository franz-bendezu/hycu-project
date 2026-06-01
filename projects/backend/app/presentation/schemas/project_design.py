from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductSpec(BaseModel):
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
    width: float
    height: float
    depth: float


class HardwareItem(BaseModel):
    code: str
    qty: int


class ProjectModel(BaseModel):
    product: ProductSpec
    components: list[Component] = Field(default_factory=list)
    hardware: list[HardwareItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
