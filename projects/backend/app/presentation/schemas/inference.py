from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceComponent(BaseModel):
    name: str
    kind: str
    quantity: int = Field(..., ge=1)


class InferenceImageOutput(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[InferenceComponent]
    image_url: str = Field(..., min_length=8)


class InferenceOutput(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    components: list[InferenceComponent]
    image_url: str = Field(..., min_length=8)
    images_analyzed: int | None = Field(default=None, ge=1)
    image_results: list[InferenceImageOutput] | None = None
