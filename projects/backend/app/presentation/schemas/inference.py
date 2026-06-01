from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InferenceComponent(BaseModel):
    id: str
    name: str
    kind: Literal["panel", "support", "hardware", "assembly"]
    quantity: int = Field(..., ge=1)


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
    parent_component_id: str
    child_component_id: str
    joint_type: Literal[
        "cam_lock",
        "shelf_pin",
        "screw",
        "hinge",
        "sliding_track",
        "telescopic_slide",
        "bracket",
    ]
    count: int = Field(..., ge=1)


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
    component_index: dict[str, InferenceComponent] | None = None
    interior: InferenceInteriorAssessment | None = None
    door: InferenceDoorAssessment | None = None
    uncertainty: InferenceUncertaintyAssessment | None = None
    joints: list[InferenceJointEvidence] | None = None
    hardware: list[InferenceHardwareRecommendation] | None = None
    image_url: str = Field(..., min_length=8)


class InferenceOutput(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[InferenceComponent]
    component_index: dict[str, InferenceComponent] | None = None
    interior: InferenceInteriorAssessment | None = None
    door: InferenceDoorAssessment | None = None
    uncertainty: InferenceUncertaintyAssessment | None = None
    joints: list[InferenceJointEvidence] | None = None
    hardware: list[InferenceHardwareRecommendation] | None = None
    image_url: str = Field(..., min_length=8)
    images_analyzed: int | None = Field(default=None, ge=1)
    image_results: list[InferenceImageOutput] | None = None
