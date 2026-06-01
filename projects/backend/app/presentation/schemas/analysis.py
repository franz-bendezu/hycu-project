from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image_url: str = Field(..., min_length=8, description="Image URL to analyze")
    project_name: str | None = Field(default=None, min_length=1, max_length=120)


class AnalyzeResponse(BaseModel):
    job_id: str
    status: Literal["queued", "complete", "failed"]
