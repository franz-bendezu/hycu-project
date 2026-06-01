from __future__ import annotations

from typing import Protocol

from app.domain import ProjectAssetEntity


class ProjectAssetRepository(Protocol):
    def get_project_asset(self, asset_id: str) -> ProjectAssetEntity | None: ...

    def list_project_assets(self, project_id: str) -> list[ProjectAssetEntity]: ...

    def add_project_asset(self, asset: ProjectAssetEntity) -> None: ...

    def save(self) -> None: ...
