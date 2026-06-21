from __future__ import annotations

import base64
import datetime
import json
import uuid
from pydantic import ValidationError

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy.orm import Session

from app.infrastructure.persistence.database import get_db
from app.infrastructure.gateways.inference_gateway import (
    InferenceGatewayError,
    get_inference_gateway,
)
from app.domain.services.model_generator import ModelGenerator
from ...domain.services.vision_model_builder import build_project_model_from_inference
from ...domain.services.artifact_builder import (
    build_blueprint_pdf,
    build_bom_csv,
    build_nesting_dxf,
    build_export_zip,
    content_disposition,
)
from app.infrastructure.persistence.models import (
    Job as JobModel,
    JobAssetResult,
    Project,
    ProjectAsset,
)
from app.presentation.schemas.project_design import ProductSpec, ProjectModel
from app.presentation.schemas.inference import InferenceOutput
from app.presentation.schemas.projects import (
    CreateProjectAssetResponse,
    CreateProjectJobRequest,
    CreateProjectJobResponse,
    CreateProjectRequest,
    CreateProjectResponse,
    ProjectAssetResponse,
    ProjectAssetsResponse,
    ProjectJobsResponse,
    ProjectResponse,
    ProjectSummary,
    ProjectsListResponse,
    UpdateProjectRequest,
    ValidateResponse,
)
from app.presentation.schemas.jobs import JobResponse

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])
generator = ModelGenerator()
client = get_inference_gateway()


def _build_or_extract_project_model(
    *,
    inference_result: InferenceOutput,
    project_name: str,
    fallback_type: str,
    material_thickness: float = 18.0,
) -> ProjectModel:
    model = build_project_model_from_inference(
        inference_result,
        project_name=project_name,
        fallback_type=fallback_type,
        material_thickness=material_thickness,
    )
    return model

def _get_project_or_404(project_id: str, db: Session) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _validated_result_or_422(raw: str | None) -> dict | None:
    if not raw:
        return None
    payload = json.loads(raw)
    if isinstance(payload, dict):
        try:
            InferenceOutput.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Stored job result does not match strict inference schema: {exc.errors()[0]['msg']}",
            ) from exc
    return payload


@router.get("", response_model=ProjectsListResponse)
def list_projects(db: Session = Depends(get_db)) -> ProjectsListResponse:
    rows = db.query(Project).order_by(Project.created_at.desc()).all()
    return ProjectsListResponse(
        projects=[
            ProjectSummary(
                project_id=p.id,
                name=p.name,
                created_at=p.created_at.isoformat(),
            )
            for p in rows
        ]
    )


@router.post("", response_model=CreateProjectResponse)
def create_project(
    payload: CreateProjectRequest, db: Session = Depends(get_db)
) -> CreateProjectResponse:
    default_name = payload.name or "Generated Project"
    result: dict | None = None

    if payload.job_id:
        job = db.get(JobModel, payload.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "complete":
            raise HTTPException(status_code=409, detail="Job is not complete")

        result = json.loads(job.result_json or "{}")
        try:
            inference_result = InferenceOutput.model_validate(result)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Inference payload does not match DTO schema: {exc.errors()[0]['msg']}",
            ) from exc
        default_name = payload.name or job.project_name or default_name
        model = _build_or_extract_project_model(
            inference_result=inference_result,
            project_name=default_name,
            fallback_type="cabinet",
        )
        job.result_json = json.dumps(result)
        report = generator.validate(model)
        if not report.valid:
            raise HTTPException(status_code=422, detail="Vision-first model failed validation")
    else:
        spec = ProductSpec(name=default_name)
        model = generator.generate(spec)
        report = generator.validate(model)
        if not report.valid:
            raise HTTPException(status_code=422, detail="Generated model failed validation")

    project_id = str(uuid.uuid4())
    project = Project(id=project_id, name=default_name, model_json=model.model_dump_json())
    db.add(project)
    db.commit()

    return CreateProjectResponse(
        project_id=project_id,
        model=model,
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.post("/{project_id}/assets", response_model=CreateProjectAssetResponse)
async def upload_project_asset(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> CreateProjectAssetResponse:
    _get_project_or_404(project_id, db)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    encoded = base64.b64encode(payload).decode("ascii")
    asset_id = str(uuid.uuid4())
    file_name = file.filename or "uploaded-image"
    content_type = file.content_type

    asset = ProjectAsset(
        id=asset_id,
        project_id=project_id,
        file_name=file_name,
        content_type=content_type,
        size_bytes=len(payload),
        image_data=f"data:{content_type};base64,{encoded}",
    )
    db.add(asset)
    db.commit()

    return CreateProjectAssetResponse(
        asset_id=asset_id,
        file_name=file_name,
        content_type=content_type,
        size_bytes=len(payload),
    )


@router.get("/{project_id}/assets", response_model=ProjectAssetsResponse)
def list_project_assets(
    project_id: str, db: Session = Depends(get_db)
) -> ProjectAssetsResponse:
    _get_project_or_404(project_id, db)
    assets = (
        db.query(ProjectAsset)
        .filter(ProjectAsset.project_id == project_id, ProjectAsset.deleted_at.is_(None))
        .all()
    )
    return ProjectAssetsResponse(
        project_id=project_id,
        assets=[
            ProjectAssetResponse(
                asset_id=a.id,
                file_name=a.file_name,
                content_type=a.content_type,
                size_bytes=a.size_bytes,
                image_url=a.image_data,
            )
            for a in assets
        ],
    )


@router.delete("/{project_id}/assets/{asset_id}", status_code=204)
def delete_project_asset(project_id: str, asset_id: str, db: Session = Depends(get_db)) -> Response:
    _get_project_or_404(project_id, db)
    asset = db.get(ProjectAsset, asset_id)
    if not asset or asset.project_id != project_id or asset.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Project asset not found")

    asset.deleted_at = datetime.datetime.now(datetime.UTC)
    db.commit()
    return Response(status_code=204)


@router.post("/{project_id}/jobs", response_model=CreateProjectJobResponse)
def create_project_job(
    project_id: str,
    payload: CreateProjectJobRequest,
    db: Session = Depends(get_db),
) -> CreateProjectJobResponse:
    project = _get_project_or_404(project_id, db)
    _ = payload
    assets = (
        db.query(ProjectAsset)
        .filter(ProjectAsset.project_id == project_id, ProjectAsset.deleted_at.is_(None))
        .all()
    )
    if not assets:
        raise HTTPException(status_code=404, detail="Project has no assets")

    image_data_list = [asset.image_data for asset in assets]

    try:
        job_id, inference_result = client.submit(image_data_list)
    except InferenceGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = inference_result.model_dump(mode="json")
    result["source_asset_ids"] = [asset.id for asset in assets]

    current_model = ProjectModel.model_validate_json(project.model_json)
    updated_model = _build_or_extract_project_model(
        inference_result=inference_result,
        project_name=current_model.product.name,
        fallback_type=current_model.product.inferred_type,
        material_thickness=current_model.product.material_thickness,
    )

    job = JobModel(
        id=job_id,
        status="complete",
        result_json=json.dumps(result),
        project_name=project.name,
        project_id=project_id,
        asset_id=None,
    )
    db.add(job)

    asset_ids = result.get("source_asset_ids")
    image_results = inference_result.image_results
    if isinstance(asset_ids, list) and isinstance(image_results, list):
        pair_count = min(len(asset_ids), len(image_results))
        for idx in range(pair_count):
            asset_id = asset_ids[idx]
            image_result = image_results[idx].model_dump(mode="json")
            if not isinstance(asset_id, str):
                continue
            db.add(
                JobAssetResult(
                    job_id=job_id,
                    asset_id=asset_id,
                    status="complete",
                    result_json=json.dumps(image_result),
                )
            )

    report = generator.validate(updated_model)
    if not report.valid:
        raise HTTPException(status_code=422, detail="Vision-first model failed validation")
    project.model_json = updated_model.model_dump_json()
    db.commit()

    return CreateProjectJobResponse(
        project_id=project_id,
        asset_id=None,
        asset_count=len(assets),
        job_id=job_id,
        status="complete",
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.get("/{project_id}/jobs", response_model=ProjectJobsResponse)
def list_project_jobs(project_id: str, db: Session = Depends(get_db)) -> ProjectJobsResponse:
    _get_project_or_404(project_id, db)

    jobs = (
        db.query(JobModel)
        .filter(JobModel.project_id == project_id)
        .order_by(JobModel.id.desc())
        .all()
    )

    return ProjectJobsResponse(
        project_id=project_id,
        jobs=[
            JobResponse(
                job_id=job.id,
                status=job.status,
                result=_validated_result_or_422(job.result_json),
                project_id=job.project_id,
                asset_id=job.asset_id,
            )
            for job in jobs
        ],
    )


@router.get("/{project_id}/jobs/{job_id}", response_model=JobResponse)
def get_project_job(
    project_id: str, job_id: str, db: Session = Depends(get_db)
) -> JobResponse:
    _get_project_or_404(project_id, db)
    job = db.get(JobModel, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(status_code=404, detail="Project job not found")

    result = json.loads(job.result_json) if job.result_json else None
    if job.result_json:
        result = _validated_result_or_422(job.result_json)
    asset_results_rows = (
        db.query(JobAssetResult)
        .filter(JobAssetResult.job_id == job_id)
        .order_by(JobAssetResult.asset_id.asc())
        .all()
    )
    return JobResponse(
        job_id=job_id,
        status=job.status,
        result=result,
        project_id=job.project_id,
        asset_id=job.asset_id,
        asset_results=[
            {
                "job_id": row.job_id,
                "asset_id": row.asset_id,
                "status": row.status,
                "result": json.loads(row.result_json) if row.result_json else None,
            }
            for row in asset_results_rows
        ]
        if asset_results_rows
        else None,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    report = generator.validate(model)
    return ProjectResponse(
        project_id=project_id,
        model=model,
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    payload: UpdateProjectRequest,
    db: Session = Depends(get_db),
) -> ProjectResponse:
    project = _get_project_or_404(project_id, db)
    current_model = ProjectModel.model_validate_json(project.model_json)

    updated = generator.apply_update(current_model, **payload.model_dump())
    report = generator.validate(updated)
    if not report.valid:
        raise HTTPException(status_code=422, detail="Project update failed validation")

    project.model_json = updated.model_dump_json()
    db.commit()

    return ProjectResponse(
        project_id=project_id,
        model=updated,
        validation=ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings),
    )


@router.post("/{project_id}/validate", response_model=ValidateResponse)
def validate_project(
    project_id: str, db: Session = Depends(get_db)
) -> ValidateResponse:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    report = generator.validate(model)
    return ValidateResponse(valid=report.valid, errors=report.errors, warnings=report.warnings)


@router.get("/{project_id}/blueprint.pdf")
def download_blueprint(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = build_blueprint_pdf(project_id, model)
    return Response(
        content=payload,
        media_type="application/pdf",
        headers=content_disposition(f"{project_id}-blueprint.pdf"),
    )


@router.get("/{project_id}/bom.csv")
def download_bom(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = build_bom_csv(model)
    return Response(
        content=payload,
        media_type="text/csv",
        headers=content_disposition(f"{project_id}-bom.csv"),
    )


@router.get("/{project_id}/nesting.dxf")
def download_nesting(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = build_nesting_dxf(model)
    return Response(
        content=payload,
        media_type="application/dxf",
        headers=content_disposition(f"{project_id}-nesting.dxf"),
    )


@router.get("/{project_id}/export")
def download_export_package(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project_or_404(project_id, db)
    model = ProjectModel.model_validate_json(project.model_json)
    payload = build_export_zip(project_id, model)
    return Response(
        content=payload,
        media_type="application/zip",
        headers=content_disposition(f"{project_id}-fabrication-package.zip"),
    )
