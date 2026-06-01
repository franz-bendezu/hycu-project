from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.domain import JobEntity, ProjectAssetEntity, ProjectEntity
from app.infrastructure.persistence.models import Job as JobModel, Project, ProjectAsset
from app.application.errors import NotFoundError


def get_project_or_error(project_id: str, db: Session) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise NotFoundError("Project not found")
    return project


def project_entity(project: Project) -> ProjectEntity:
    return ProjectEntity.from_record(project)


def asset_entity(asset: ProjectAsset | None) -> ProjectAssetEntity | None:
    if asset is None:
        return None
    return ProjectAssetEntity.from_record(asset)


def job_entity(job: JobModel | None) -> JobEntity | None:
    if job is None:
        return None
    return JobEntity.from_record(job)


def project_record(entity: ProjectEntity) -> Project:
    return Project(id=entity.id, name=entity.name, model_json=entity.design.model_dump_json())


def asset_record(entity: ProjectAssetEntity) -> ProjectAsset:
    return ProjectAsset(
        id=entity.id,
        project_id=entity.project_id,
        file_name=entity.file_name,
        content_type=entity.content_type,
        size_bytes=entity.size_bytes,
        image_data=entity.image_data,
    )


def job_record(entity: JobEntity) -> JobModel:
    return JobModel(
        id=entity.id,
        status=entity.status,
        result_json=json.dumps(entity.result) if entity.result is not None else None,
        project_name=entity.project_name,
        project_id=entity.project_id,
        asset_id=entity.asset_id,
    )


def sync_project_record(record: Project, entity: ProjectEntity) -> None:
    record.name = entity.name
    record.model_json = entity.design.model_dump_json()


def sync_job_record(record: JobModel, entity: JobEntity) -> None:
    record.status = entity.status
    record.result_json = json.dumps(entity.result) if entity.result is not None else None
    record.project_name = entity.project_name
    record.project_id = entity.project_id
    record.asset_id = entity.asset_id
