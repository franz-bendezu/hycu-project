# Vision-to-Blueprint Implementation Skeleton

This repository now contains a runnable starter architecture aligned with class implementation priorities:

- Training project for model training/export workflow.
- Inference project for deployed inference endpoint.
- Backend (FastAPI) for orchestration, rules, model generation, and API gateway.
- Frontend (React + TypeScript + Vite) that calls backend only.
- Infrastructure starter (Dev Container + Docker Compose services).

## Project Tree

- projects/training: training/export pipeline project (not deployed in runtime)
- projects/inference: deployed inference API service
- projects/backend: orchestration API, rules, model generation, project flow
- projects/frontend: web application (consumes backend only)
- projects/infra: additional compose setup
- .devcontainer: containerized development environment (no local Python or Node install required)

## Dev Container First (recommended)

Use this workflow if you do not want to install Python or Node locally.
This setup now includes Docker-in-Docker (DIND), so you can run docker commands from inside the devcontainer.

1. Ensure Docker Desktop is running.
2. Open this folder in VS Code.
3. Run Reopen in Container.
4. Wait for post-create setup to finish (backend pip install and frontend npm install).
5. Verify Docker in the container: docker version

After container starts, run these commands in VS Code terminal:

1. Backend:
	- cd projects/backend
	- uvicorn app.main:app --reload --port 8000
2. Frontend (new terminal):
	- cd projects/frontend
	- npm run dev -- --host 0.0.0.0 --port 5173

Services started by devcontainer compose:

- inference on 9000
- postgres on 5432
- redis on 6379

You can also run Docker commands inside the container, for example:

- docker ps
- docker build -t local-test .

## Backend Run (local optional)

1. Open terminal in backend folder.
2. Create virtual environment.
3. Install dependencies from requirements.txt.
4. Run: uvicorn app.main:app --reload --port 8000

Folder path: projects/backend

## Backend Endpoints

- GET /health
- POST /api/v1/analyze
- GET /api/v1/jobs/{job_id}
- POST /api/v1/projects
- GET /api/v1/projects/{project_id}
- PATCH /api/v1/projects/{project_id}
- POST /api/v1/projects/{project_id}/validate

## Inference Endpoints

- GET /health
- POST /infer

## Runtime Architecture

1. Frontend calls backend only.
2. Backend calls inference.
3. training is used offline to train/export models, then artifacts are deployed to inference.

## Frontend Run (local)

1. Open terminal in frontend folder.
2. Install dependencies.
3. Run: npm run dev

Folder path: projects/frontend

Optional environment variable:

- VITE_BACKEND_URL (defaults to http://localhost:8000)

Copy projects/frontend/.env.example to .env when needed.

## Docker Compose

From projects/infra folder run docker compose up --build.

This starts backend, postgres, and redis.
