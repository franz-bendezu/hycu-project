from __future__ import annotations

from app.domain.services import ModelGenerator
from app.domain.services import ValidationResult
from app.application.errors import NotFoundError
from app.application.queries import ProjectQuery
from app.application.ports import ProjectRepository


class ValidateProjectUseCase:
    def __init__(self, generator: ModelGenerator, project_repository: ProjectRepository) -> None:
        self.generator = generator
        self.project_repository = project_repository

    def execute(self, query: ProjectQuery) -> ValidationResult:
        project = self.project_repository.get_project(query.project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return self.generator.validate(project.design)
