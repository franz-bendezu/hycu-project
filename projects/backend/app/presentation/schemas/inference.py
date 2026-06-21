from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from app.presentation.schemas.project_design import (
    ComponentCategory,
    ComponentKind,
    HardwareMountFace,
    JointRule,
)


class InferenceComponent(BaseModel):
    id: str
    kind: ComponentKind
    category: ComponentCategory | None = None
    material_id: str | None = None
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    depth: float = Field(..., gt=0)


class InferenceFaceEvidence(BaseModel):
    id: str
    component_id: str
    normal: HardwareMountFace


class InferenceInteriorAssessment(BaseModel):
    visibility: Literal[
        "interior_not_visible", "interior_partially_visible", "interior_fully_visible"
    ]
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    unknown_interior: bool


class InferenceDoorAssessment(BaseModel):
    type: Literal["hinged", "sliding", "unknown"]
    count_uncertain: bool


class InferenceUncertaintyAssessment(BaseModel):
    hardware_uncertain: bool


class InferenceJointEvidence(BaseModel):
    id: str
    parent_face_id: str
    child_face_id: str
    joint_rule: JointRule | None = None
    offset_u: float = 0.0
    offset_v: float = 0.0
    clearance: float = 0.0


class InferenceHardwareRecommendation(BaseModel):
    code: Literal[
        "CAM_LOCK_15MM",
        "SHELF_PIN_5MM",
        "HINGE_SOFT_CLOSE_110",
        "WOOD_SCREW_4X16",
        "SLIDING_DOOR_TRACK_SET",
        "TELESCOPIC_SLIDE_400",
        "WOOD_SCREW_4X40",
        "CORNER_BRACKET_40",
        "HARDWARE_REVIEW_REQUIRED",
    ]
    qty: int = Field(..., ge=1)
    reason: str | None = None


class InferenceImageOutput(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[InferenceComponent]
    faces: list[InferenceFaceEvidence]
    joints: list[InferenceJointEvidence]
    interior: InferenceInteriorAssessment | None = None
    door: InferenceDoorAssessment | None = None
    uncertainty: InferenceUncertaintyAssessment | None = None
    hardware: list[InferenceHardwareRecommendation] | None = None
    image_url: str = Field(..., min_length=8)


class InferenceImageEvidence(BaseModel):
    image_url: str = Field(..., min_length=8)
    width_px: int = Field(..., ge=1)
    height_px: int = Field(..., ge=1)
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_detections: list[dict] = Field(default_factory=list)


class InferenceOutput(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[InferenceComponent]
    faces: list[InferenceFaceEvidence]
    joints: list[InferenceJointEvidence]
    interior: InferenceInteriorAssessment | None = None
    door: InferenceDoorAssessment | None = None
    uncertainty: InferenceUncertaintyAssessment | None = None
    hardware: list[InferenceHardwareRecommendation] | None = None
    image_url: str = Field(..., min_length=8)
    images_analyzed: int | None = Field(default=None, ge=1)
    image_results: list[InferenceImageEvidence] | None = None
    evidence: list[InferenceImageEvidence] | None = None
