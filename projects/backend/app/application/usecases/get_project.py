from __future__ import annotations

from app.domain import ProjectEntity
from app.application.errors import NotFoundError
from app.application.queries import ProjectQuery
from app.application.ports import ProjectRepository


class GetProjectUseCase:
    def __init__(self, project_repository: ProjectRepository) -> None:
        self.project_repository = project_repository

    def execute(self, query: ProjectQuery) -> ProjectEntity:
        project = self.project_repository.get_project(query.project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return project
