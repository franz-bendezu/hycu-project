from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text

from app.infrastructure.persistence.database import Base, engine
from app.api.routers.analysis import router as analysis_router
from app.api.routers.health import router as health_router
from app.api.routers.projects import router as projects_router

# Import models so SQLAlchemy registers them before create_all
import app.infrastructure.persistence.models  # noqa: F401

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Vision-to-Blueprint Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(projects_router)


@app.exception_handler(HTTPException)
def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
	message = exc.detail if isinstance(exc.detail, str) else "Request failed"
	return JSONResponse(
		status_code=exc.status_code,
		content={
			"error": {
				"code": f"HTTP_{exc.status_code}",
				"message": message,
			}
		},
	)


@app.exception_handler(RequestValidationError)
def handle_validation_exception(_: Request, exc: RequestValidationError) -> JSONResponse:
	first = exc.errors()[0] if exc.errors() else None
	message = first.get("msg", "Validation failed") if first else "Validation failed"
	return JSONResponse(
		status_code=422,
		content={
			"error": {
				"code": "REQUEST_VALIDATION_ERROR",
				"message": message,
			}
		},
	)


@app.exception_handler(Exception)
def handle_unexpected_exception(_: Request, __: Exception) -> JSONResponse:
	return JSONResponse(
		status_code=500,
		content={
			"error": {
				"code": "INTERNAL_SERVER_ERROR",
				"message": "Unexpected server error",
			}
		},
	)
