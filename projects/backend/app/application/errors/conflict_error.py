from __future__ import annotations

from app.application.errors.use_case_error import UseCaseError


class ConflictError(UseCaseError):
    pass
