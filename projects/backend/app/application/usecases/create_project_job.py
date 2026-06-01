from __future__ import annotations

from app.domain.services import ModelGenerator
from app.domain import JobEntity, ProductSpec
from app.application.errors import ConflictError, IntegrationError, NotFoundError
from app.application.commands import CreateProjectJobCommand
from app.application.ports import (
    InferenceGateway,
    JobRepository,
    ProjectAssetRepository,
    ProjectRepository,
)


def _shelf_count_from_detected_type(detected_type: str | None, fallback: int) -> int:
    normalized = (detected_type or "").strip().lower()
    if normalized == "desk":
        return 0
    if normalized == "cabinet":
        return 2
    if normalized == "shelf":
        return 4
    return fallback


def _inferred_type_from_detected_type(detected_type: str | None, fallback: str) -> str:
    normalized = (detected_type or "").strip().lower()
    if normalized in {"cabinet", "desk", "shelf"}:
        return normalized
    return fallback


class CreateProjectJobUseCase:
    def __init__(
        self,
        generator: ModelGenerator,
        project_repository: ProjectRepository,
        job_repository: JobRepository,
        asset_repository: ProjectAssetRepository,
        inference_gateway: InferenceGateway,
    ) -> None:
        self.generator = generator
        self.project_repository = project_repository
        self.job_repository = job_repository
        self.asset_repository = asset_repository
        self.inference_gateway = inference_gateway

    def execute(
        self,
        command: CreateProjectJobCommand,
    ) -> JobEntity:
        project = self.project_repository.get_project(command.project_id)
        if project is None:
            raise NotFoundError("Project not found")
        try:
            asset = project.ensure_asset_belongs(self.asset_repository.get_project_asset(command.asset_id))
        except ValueError as exc:
            raise NotFoundError(str(exc)) from exc

        active_job = self.job_repository.get_active_project_job(command.project_id)
        try:
            project.ensure_can_start_job(active_job)
        except ValueError as exc:
            raise ConflictError(str(exc)) from exc

        job = project.start_job(asset)
        self.job_repository.add_job(job)
        self.job_repository.save()

        try:
            result = self.inference_gateway.infer(asset.image_data)
        except Exception as exc:
            job.fail()
            self.job_repository.update_job(job)
            self.job_repository.save()
            raise IntegrationError(str(exc)) from exc

        current_model = project.design
        inferred_type = _inferred_type_from_detected_type(
            result.get("detected_type"),
            fallback=current_model.product.inferred_type,
        )
        spec = ProductSpec(
            name=current_model.product.name,
            inferred_type=inferred_type,
            target_width=result["suggested_width"],
            target_height=result["suggested_height"],
            target_depth=result["suggested_depth"],
            shelf_count=_shelf_count_from_detected_type(
                inferred_type,
                fallback=current_model.product.shelf_count,
            ),
        )
        updated_model = self.generator.generate(spec)

        project.apply_inference_result(job=job, result=result, model=updated_model)
        self.job_repository.update_job(job)
        self.project_repository.update_project(project)
        self.job_repository.save()

        return job
