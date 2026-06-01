from __future__ import annotations

from app.domain import ProjectAssetEntity
from app.application.errors import NotFoundError
from app.application.queries import ProjectQuery
from app.application.ports import ProjectAssetRepository, ProjectRepository


class ListProjectAssetsUseCase:
    def __init__(
        self,
        project_repository: ProjectRepository,
        asset_repository: ProjectAssetRepository,
    ) -> None:
        self.project_repository = project_repository
        self.asset_repository = asset_repository

    def execute(self, query: ProjectQuery) -> list[ProjectAssetEntity]:
        project = self.project_repository.get_project(query.project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return self.asset_repository.list_project_assets(query.project_id)
