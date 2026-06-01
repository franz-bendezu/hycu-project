from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.presentation.schemas.jobs import JobResponse
from app.presentation.schemas.project_design import ProjectModel


class CreateProjectRequest(BaseModel):
    job_id: str | None = None
    name: str | None = Field(default=None, min_length=1, max_length=120)


class CreateProjectResponse(BaseModel):
    project_id: str
    model: ProjectModel


class CreateProjectAssetResponse(BaseModel):
    asset_id: str
    file_name: str
    content_type: str
    size_bytes: int


class ProjectAssetResponse(BaseModel):
    asset_id: str
    file_name: str
    content_type: str
    size_bytes: int


class ProjectAssetsResponse(BaseModel):
    project_id: str
    assets: list[ProjectAssetResponse]


class ProjectJobsResponse(BaseModel):
    project_id: str
    jobs: list[JobResponse]


class CreateProjectJobRequest(BaseModel):
    asset_id: str


class CreateProjectJobResponse(BaseModel):
    project_id: str
    asset_id: str
    job_id: str
    status: Literal["queued", "complete", "failed"]


class UpdateProjectRequest(BaseModel):
    target_width: float | None = Field(default=None, gt=0)
    target_height: float | None = Field(default=None, gt=0)
    target_depth: float | None = Field(default=None, gt=0)
    material_thickness: float | None = Field(default=None, ge=8, le=50)
    shelf_count: int | None = Field(default=None, ge=0, le=20)


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]


class ProjectResponse(BaseModel):
    project_id: str
    model: ProjectModel


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    created_at: str


class ProjectsListResponse(BaseModel):
    projects: list[ProjectSummary]
