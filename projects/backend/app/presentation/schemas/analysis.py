from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnalyzeRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1, description="Batch image URLs")
    project_name: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_sources(self) -> "AnalyzeRequest":
        if len(self.image_urls) > 12:
            raise ValueError("Maximum 12 images per request")
        return self


class AnalyzeResponse(BaseModel):
    job_id: str
    status: Literal["queued", "complete", "failed"]
