from __future__ import annotations

from app.domain.services import ModelGenerator
from app.domain import ProjectEntity
from app.application.errors import NotFoundError, ValidationError
from app.application.commands import UpdateProjectCommand
from app.application.ports import ProjectRepository


class UpdateProjectUseCase:
    def __init__(self, generator: ModelGenerator, project_repository: ProjectRepository) -> None:
        self.generator = generator
        self.project_repository = project_repository

    def execute(self, command: UpdateProjectCommand) -> ProjectEntity:
        project = self.project_repository.get_project(command.project_id)
        if project is None:
            raise NotFoundError("Project not found")

        updated = self.generator.apply_update(project.design, **command.as_updates())
        report = self.generator.validate(updated)
        if not report.valid:
            raise ValidationError("Project update failed validation")

        project.apply_design_model(updated)
        self.project_repository.update_project(project)
        self.project_repository.save()

        return project
