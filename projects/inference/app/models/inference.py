from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ComponentKind(StrEnum):
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


class ComponentCategory(StrEnum):
    STRUCTURAL = "structural"
    FRONT = "front"
    INTERNAL = "internal"
    SUPPORT = "support"


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


class EscalationStrategy(StrEnum):
    FAST_2D_FUSION = "fast_2d_fusion"
    ESCALATE_GEOMETRY_OPTIMIZATION = "escalate_geometry_optimization"
    ESCALATE_MVS_REFINEMENT = "escalate_mvs_refinement"
    GEOMETRY_REFINEMENT_REQUIRES_DEPTH = "geometry_refinement_requires_depth"
    GEOMETRY_REFINEMENT_APPLIED = "geometry_refinement_applied"
    MVS_REFINEMENT_APPLIED = "mvs_refinement_applied"


class HardwareMountFace(StrEnum):
    POS_X = "+x"
    NEG_X = "-x"
    POS_Y = "+y"
    NEG_Y = "-y"
    POS_Z = "+z"
    NEG_Z = "-z"


class JointRule(StrEnum):
    OVERLAP = "overlap"
    INSET = "inset"
    BETWEEN = "between"
    FLUSH_BACK = "flush_back"
    MOUNT = "mount"


class RawDetection(BaseModel):
    model_config = ConfigDict(extra="allow")

    box: tuple[float, float, float, float]
    score: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    label: str | None = None
    image_width_px: int | None = None
    image_height_px: int | None = None
    track_id: int | None = None
    track_iou: float | None = None
    view_index: int | None = None
    mask_area_px: int | None = None
    mask_fill_ratio: float | None = None


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
    category: ComponentCategory | None = None
    material_id: str | None = None
    width: float = Field(default=1.0, gt=0)
    height: float = Field(default=1.0, gt=0)
    depth: float = Field(default=1.0, gt=0)
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
    parent_face_id: str
    child_face_id: str
    joint_rule: JointRule | None = None
    offset_u: float = 0.0
    offset_v: float = 0.0
    clearance: float = 0.0
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


class FaceEvidence(BaseModel):
    id: str
    component_id: str
    normal: HardwareMountFace


class ImageEvidence(BaseModel):
    image_url: str = Field(..., min_length=8)
    width_px: int
    height_px: int
    detected_type: ProductType
    confidence: float
    raw_detections: list[RawDetection] = Field(default_factory=list)


class InferResponse(BaseModel):
    schema_version: str = "1.1.0"
    coordinate_frame: Literal["furniture_local_v1"] = "furniture_local_v1"
    detected_type: ProductType
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[Component]
    faces: list[FaceEvidence] = Field(default_factory=list)
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


class BenchmarkItemSummary(BaseModel):
    item_id: str
    detected_type: ProductType
    confidence: float = Field(..., ge=0.0, le=1.0)
    component_coverage: float = Field(..., ge=0.0, le=1.0)
    physical_validity_score: float = Field(..., ge=0.0, le=1.0)
    escalation_strategy: EscalationStrategy
    human_review_required: bool


class BenchmarkResponse(BaseModel):
    items_analyzed: int = Field(..., ge=1)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_component_coverage: float = Field(..., ge=0.0, le=1.0)
    avg_physical_validity: float = Field(..., ge=0.0, le=1.0)
    human_review_rate: float = Field(..., ge=0.0, le=1.0)
    escalation_strategy_counts: dict[str, int] = Field(default_factory=dict)
    item_results: list[BenchmarkItemSummary] = Field(default_factory=list)
