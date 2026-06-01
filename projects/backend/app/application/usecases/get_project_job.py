from __future__ import annotations

from app.domain import JobEntity
from app.application.errors import NotFoundError
from app.application.queries import ProjectJobQuery
from app.application.ports import JobRepository, ProjectRepository


class GetProjectJobUseCase:
    def __init__(
        self,
        project_repository: ProjectRepository,
        job_repository: JobRepository,
    ) -> None:
        self.project_repository = project_repository
        self.job_repository = job_repository

    def execute(self, query: ProjectJobQuery) -> JobEntity:
        project = self.project_repository.get_project(query.project_id)
        if project is None:
            raise NotFoundError("Project not found")
        job = self.job_repository.get_project_job(query.project_id, query.job_id)
        if job is None:
            raise NotFoundError("Project job not found")
        return job
