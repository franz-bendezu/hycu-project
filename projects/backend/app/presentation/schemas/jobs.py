from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.presentation.schemas.inference import InferenceOutput


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "complete", "failed"]
    result: InferenceOutput | None = None
    project_id: str | None = None
    asset_id: str | None = None
