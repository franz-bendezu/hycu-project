from __future__ import annotations

from app.domain import JobEntity
from app.application.errors import NotFoundError
from app.application.queries import ProjectQuery
from app.application.ports import JobRepository, ProjectRepository


class ListProjectJobsUseCase:
    def __init__(
        self,
        project_repository: ProjectRepository,
        job_repository: JobRepository,
    ) -> None:
        self.project_repository = project_repository
        self.job_repository = job_repository

    def execute(self, query: ProjectQuery) -> list[JobEntity]:
        project = self.project_repository.get_project(query.project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return self.job_repository.list_project_jobs(query.project_id)
