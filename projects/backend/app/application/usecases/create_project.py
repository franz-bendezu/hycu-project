from __future__ import annotations

from app.domain.services import ModelGenerator
from app.domain import ProductSpec, ProjectEntity
from app.application.errors import ConflictError, NotFoundError, ValidationError
from app.application.commands import CreateProjectCommand
from app.application.ports import JobRepository, ProjectRepository


def _shelf_count_from_detected_type(detected_type: str | None, fallback: int = 3) -> int:
    normalized = (detected_type or "").strip().lower()
    if normalized == "desk":
        return 0
    if normalized == "cabinet":
        return 2
    if normalized == "shelf":
        return 4
    return fallback


def _inferred_type_from_detected_type(detected_type: str | None, fallback: str = "cabinet") -> str:
    normalized = (detected_type or "").strip().lower()
    if normalized in {"cabinet", "desk", "shelf"}:
        return normalized
    return fallback


class CreateProjectUseCase:
    def __init__(
        self,
        generator: ModelGenerator,
        project_repository: ProjectRepository,
        job_repository: JobRepository,
    ) -> None:
        self.generator = generator
        self.project_repository = project_repository
        self.job_repository = job_repository

    def execute(
        self,
        command: CreateProjectCommand,
    ) -> ProjectEntity:
        default_name = command.name or "Generated Project"

        if command.job_id:
            job = self.job_repository.get_job(command.job_id)
            if not job:
                raise NotFoundError("Job not found")
            if job.status != "complete":
                raise ConflictError("Job is not complete")

            result = job.result or {}
            default_name = command.name or job.project_name or default_name
            inferred_type = _inferred_type_from_detected_type(result.get("detected_type"))
            spec = ProductSpec(
                name=default_name,
                inferred_type=inferred_type,
                target_width=result["suggested_width"],
                target_height=result["suggested_height"],
                target_depth=result["suggested_depth"],
                shelf_count=_shelf_count_from_detected_type(inferred_type, fallback=3),
            )
        else:
            spec = ProductSpec(name=default_name)

        model = self.generator.generate(spec)
        report = self.generator.validate(model)
        if not report.valid:
            raise ValidationError("Generated model failed validation")

        project = ProjectEntity.create(name=default_name, model=model)
        self.project_repository.add_project(project)
        self.project_repository.save()

        return project
