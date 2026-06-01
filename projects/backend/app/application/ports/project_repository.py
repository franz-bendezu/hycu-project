from __future__ import annotations

from typing import Protocol

from app.domain import ProjectEntity


class ProjectRepository(Protocol):
    def list_projects(self) -> list[ProjectEntity]: ...

    def get_project(self, project_id: str) -> ProjectEntity | None: ...

    def add_project(self, project: ProjectEntity) -> None: ...

    def update_project(self, project: ProjectEntity) -> None: ...

    def save(self) -> None: ...
