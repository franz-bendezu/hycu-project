from __future__ import annotations

import base64
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.infrastructure.persistence.database import get_db
from app.infrastructure.gateways.inference_gateway import (
    InferenceGatewayError,
    get_inference_gateway,
)
from app.infrastructure.persistence.models import Job as JobModel
from app.presentation.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.presentation.schemas.jobs import JobResponse

router = APIRouter(prefix="/api/v1", tags=["analysis"])
client = get_inference_gateway()


def _submit_analysis(
    image_url: str,
    project_name: str | None,
    db: Session,
) -> AnalyzeResponse:
    try:
        job_id, result = client.submit(image_url)
    except InferenceGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    job = JobModel(
        id=job_id,
        status="complete",
        result_json=json.dumps(result),
        project_name=project_name,
    )
    db.add(job)
    db.commit()
    return AnalyzeResponse(job_id=job_id, status="complete")


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalyzeResponse:
    return _submit_analysis(payload.image_url, payload.project_name, db)


@router.post("/analyze-upload", response_model=AnalyzeResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    project_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    encoded = base64.b64encode(payload).decode("ascii")
    image_url = f"data:{file.content_type};base64,{encoded}"
    return _submit_analysis(image_url, project_name, db)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    job = db.get(JobModel, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = json.loads(job.result_json) if job.result_json else None
    return JobResponse(
        job_id=job_id,
        status=job.status,
        result=result,
        project_id=job.project_id,
        asset_id=job.asset_id,
    )

