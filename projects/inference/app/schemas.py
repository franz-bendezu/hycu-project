from __future__ import annotations

from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class InferRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "InferRequest":
        if len(self.image_urls) > 12:
            raise ValueError("Maximum 12 images per request")
        for idx, value in enumerate(self.image_urls):
            if len(value) < 8:
                raise ValueError(f"image_urls[{idx}] is too short")
        return self


class BenchmarkItemRequest(BaseModel):
    item_id: str | None = None
    image_urls: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "BenchmarkItemRequest":
        if len(self.image_urls) > 12:
            raise ValueError("Maximum 12 images per item")
        for idx, value in enumerate(self.image_urls):
            if len(value) < 8:
                raise ValueError(f"image_urls[{idx}] is too short")
        return self


class BenchmarkRequest(BaseModel):
    items: list[BenchmarkItemRequest] = Field(..., min_length=1, max_length=64)


class BenchmarkItemSummary(BaseModel):
    item_id: str
    detected_type: ProductType
    confidence: float = Field(..., ge=0.0, le=1.0)
    component_coverage: float = Field(..., ge=0.0, le=1.0)
    physical_validity_score: float = Field(..., ge=0.0, le=1.0)
    escalation_strategy: str
    human_review_required: bool


class BenchmarkResponse(BaseModel):
    items_analyzed: int = Field(..., ge=1)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_component_coverage: float = Field(..., ge=0.0, le=1.0)
    avg_physical_validity: float = Field(..., ge=0.0, le=1.0)
    human_review_rate: float = Field(..., ge=0.0, le=1.0)
    escalation_strategy_counts: dict[str, int] = Field(default_factory=dict)
    item_results: list[BenchmarkItemSummary] = Field(default_factory=list)


class ComponentKind(StrEnum):
    PANEL = "panel"
    SUPPORT = "support"
    HARDWARE = "hardware"
    ASSEMBLY = "assembly"


class DoorType(StrEnum):
    HINGED = "hinged"
    SLIDING = "sliding"
    UNKNOWN = "unknown"


class ProductType(StrEnum):
    CABINET = "cabinet"
    WARDROBE = "wardrobe"
    BOOKCASE = "bookcase"
    DESK = "desk"
    TABLE = "table"
    SHELF = "shelf"
    NIGHTSTAND = "nightstand"
    DRESSER = "dresser"
    SIDEBOARD = "sideboard"
    TV_STAND = "tv_stand"


class SegmentationBackend(StrEnum):
    BOX_RASTERIZER = "box-rasterizer"
    SAM2 = "sam2"


class JointType(StrEnum):
    CAM_LOCK = "cam_lock"
    SHELF_PIN = "shelf_pin"
    SCREW = "screw"
    HINGE = "hinge"
    SLIDING_TRACK = "sliding_track"
    TELESCOPIC_SLIDE = "telescopic_slide"
    BRACKET = "bracket"


class InteriorVisibility(StrEnum):
    INTERIOR_NOT_VISIBLE = "interior_not_visible"
    INTERIOR_PARTIALLY_VISIBLE = "interior_partially_visible"
    INTERIOR_FULLY_VISIBLE = "interior_fully_visible"


class HardwareCode(StrEnum):
    CAM_LOCK_15MM = "CAM_LOCK_15MM"
    SHELF_PIN_5MM = "SHELF_PIN_5MM"
    HINGE_SOFT_CLOSE_110 = "HINGE_SOFT_CLOSE_110"
    WOOD_SCREW_4X16 = "WOOD_SCREW_4X16"
    SLIDING_DOOR_TRACK_SET = "SLIDING_DOOR_TRACK_SET"
    TELESCOPIC_SLIDE_400 = "TELESCOPIC_SLIDE_400"
    WOOD_SCREW_4X40 = "WOOD_SCREW_4X40"
    CORNER_BRACKET_40 = "CORNER_BRACKET_40"
    HARDWARE_REVIEW_REQUIRED = "HARDWARE_REVIEW_REQUIRED"


class Component(BaseModel):
    id: str
    name: str
    kind: ComponentKind
    quantity: int = Field(..., ge=1)
    # Positioning metadata for 3D reconstruction
    position_label: str | None = None  # e.g., "left", "right", "top", "bottom"
    box_corners: tuple[float, float, float, float] | None = None  # [x1, y1, x2, y2] normalized
    dimensions_mm: tuple[float, float, float] | None = None  # [length, width, thickness]
    relative_position: tuple[float, float, float] | None = None  # [x, y, z] in furniture-local frame
    bbox_3d: tuple[float, float, float, float, float, float] | None = None  # [cx, cy, cz, sx, sy, sz]
    orientation_euler_deg: tuple[float, float, float] | None = None  # [roll, pitch, yaw]
    visible_in_views: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)


class InteriorAssessment(BaseModel):
    visibility: InteriorVisibility
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    unknown_interior: bool


class DoorAssessment(BaseModel):
    type: DoorType
    count_uncertain: bool


class UncertaintyAssessment(BaseModel):
    hardware_uncertain: bool


class JointEvidence(BaseModel):
    id: str
    parent_component_id: str
    child_component_id: str
    joint_type: JointType
    count: int = Field(..., ge=1)
    anchor_parent: tuple[float, float, float] | None = None
    anchor_child: tuple[float, float, float] | None = None
    orientation_axis: tuple[float, float, float] | None = None
    orientation_degrees: float | None = None
    fit_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_view_ids: list[int] = Field(default_factory=list)


class HardwareRecommendation(BaseModel):
    code: HardwareCode
    qty: int = Field(..., ge=1)
    reason: str | None = None


class ImageEvidence(BaseModel):
    image_url: str = Field(..., min_length=8)
    width_px: int
    height_px: int
    detected_type: ProductType
    confidence: float
    raw_detections: list[dict] = Field(default_factory=list)


class InferResponse(BaseModel):
    schema_version: str = "1.1.0"
    coordinate_frame: Literal["furniture_local_v1"] = "furniture_local_v1"
    detected_type: ProductType
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[Component]
    component_index: dict[str, Component]
    interior: InteriorAssessment
    door: DoorAssessment
    uncertainty: UncertaintyAssessment
    joints: list[JointEvidence]
    hardware: list[HardwareRecommendation]
    image_url: str = Field(..., min_length=8)
    images_analyzed: int = Field(default=1, ge=1)
    image_results: list[ImageEvidence] = Field(default_factory=list)
    evidence: list[ImageEvidence] = Field(default_factory=list)
    deterministic_hash: str = Field(default="")
    constraints_report: dict[str, float | int | bool] = Field(default_factory=dict)
    review_flags: list[str] = Field(default_factory=list)
    validation_metrics: dict[str, float | int | bool] = Field(default_factory=dict)
    escalation: dict[str, str | float | bool | list[str]] = Field(default_factory=dict)
