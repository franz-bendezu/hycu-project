from __future__ import annotations

from app.domain import ProjectEntity
from app.application.queries import ListProjectsQuery
from app.application.ports import ProjectRepository


class ListProjectsUseCase:
    def __init__(self, project_repository: ProjectRepository) -> None:
        self.project_repository = project_repository

    def execute(self, _: ListProjectsQuery) -> list[ProjectEntity]:
        return self.project_repository.list_projects()
