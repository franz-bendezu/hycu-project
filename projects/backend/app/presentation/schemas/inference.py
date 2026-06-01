from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceOutput(BaseModel):
    detected_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_width: float = Field(..., gt=0)
    suggested_height: float = Field(..., gt=0)
    suggested_depth: float = Field(..., gt=0)
    image_url: str = Field(..., min_length=8)
