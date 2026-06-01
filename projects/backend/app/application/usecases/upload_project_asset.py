from __future__ import annotations

from app.domain import ProjectAssetEntity
from app.application.errors import NotFoundError, UnsupportedMediaTypeError, ValidationError
from app.application.commands import UploadProjectAssetCommand
from app.application.ports import ProjectAssetRepository, ProjectRepository


class UploadProjectAssetUseCase:
    def __init__(
        self,
        project_repository: ProjectRepository,
        asset_repository: ProjectAssetRepository,
    ) -> None:
        self.project_repository = project_repository
        self.asset_repository = asset_repository

    def execute(
        self,
        command: UploadProjectAssetCommand,
    ) -> ProjectAssetEntity:
        project = self.project_repository.get_project(command.project_id)
        if project is None:
            raise NotFoundError("Project not found")
        if not command.content_type or not command.content_type.startswith("image/"):
            raise UnsupportedMediaTypeError("Uploaded file must be an image")
        if not command.payload:
            raise ValidationError("Uploaded file is empty")

        asset = project.create_asset(
            file_name=command.file_name or "uploaded-image",
            content_type=command.content_type,
            payload=command.payload,
        )
        self.asset_repository.add_project_asset(asset)
        self.asset_repository.save()

        return asset
