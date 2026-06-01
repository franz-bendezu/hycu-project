import { createContext, FormEvent, ReactNode, useContext, useEffect, useMemo, useState } from "react";

import {
  ProjectAppearance,
  createProjectAnalysisJob,
  createProject,
  getProject,
  getJob,
  healthCheck,
  ProductSpec,
  ProjectModel,
  uploadProjectAsset,
  updateProject,
  validateProject,
  ValidateResponse,
} from "../api/backend";
import type { QueuedImage } from "../components/ImageBatchUploader";

type DraftSpec = Pick<
  ProductSpec,
  "target_width" | "target_height" | "target_depth" | "shelf_count" | "material_thickness"
>;

export type ProjectAnalysisStatus = "idle" | "analyzing" | "complete" | "failed";

export type ProjectRecord = {
  key: string;
  name: string;
  createdAt: string;
  analysisStatus: ProjectAnalysisStatus;
  jobId?: string;
  backendProjectId?: string;
  imageCount: number;
  analyzedCount: number;
  failedCount: number;
  lastError?: string;
};

type WorkflowContextValue = {
  backendStatus: "unknown" | "ok" | "down";
  imageUrl: string;
  setImageUrl: (value: string) => void;
  projects: ProjectRecord[];
  selectedProjectKey: string;
  createNewProject: (name: string) => string;
  selectProject: (projectKey: string) => void;
  getProjectByKey: (projectKey: string) => ProjectRecord | undefined;
  queuedImages: QueuedImage[];
  addImageFiles: (files: File[]) => void;
  removeImageFile: (id: string) => void;
  clearImageQueue: () => void;
  setActiveJob: (nextJobId: string) => void;
  projectName: string;
  setProjectName: (value: string) => void;
  jobId: string;
  projectId: string;
  jobStatus: string;
  projectModel: ProjectModel | null;
  draftSpec: DraftSpec | null;
  setDraftSpec: (value: DraftSpec | null) => void;
  appearance: ProjectAppearance;
  setAppearance: (value: ProjectAppearance) => void;
  validation: ValidateResponse | null;
  activeAction: string;
  error: string;
  clearError: () => void;
  onHealthCheck: () => Promise<void>;
  onAnalyzeUrl: (event: FormEvent) => Promise<void>;
  onAnalyzeQueuedImages: () => Promise<void>;
  onCreateProject: () => Promise<void>;
  onUpdateProject: (event: FormEvent) => Promise<void>;
  onValidateProject: () => Promise<void>;
};

const WorkflowContext = createContext<WorkflowContextValue | undefined>(undefined);

function applyDraftFromModel(model: ProjectModel): DraftSpec {
  return {
    target_width: model.product.target_width,
    target_height: model.product.target_height,
    target_depth: model.product.target_depth,
    material_thickness: model.product.material_thickness,
    shelf_count: model.product.shelf_count,
  };
}

export function WorkflowProvider({ children }: { children: ReactNode }): React.JSX.Element {
  const [backendStatus, setBackendStatus] = useState<"unknown" | "ok" | "down">("unknown");
  const [imageUrl, setImageUrl] = useState("");
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [selectedProjectKey, setSelectedProjectKey] = useState("");
  const [queuedImages, setQueuedImages] = useState<QueuedImage[]>([]);
  const [projectName, setProjectName] = useState("My Cabinet");
  const [jobId, setJobId] = useState("");
  const [projectId, setProjectId] = useState("");
  const [jobStatus, setJobStatus] = useState("");
  const [projectModel, setProjectModel] = useState<ProjectModel | null>(null);
  const [draftSpec, setDraftSpec] = useState<DraftSpec | null>(null);
  const [appearance, setAppearance] = useState<ProjectAppearance>({ finish: "matte" });
  const [validation, setValidation] = useState<ValidateResponse | null>(null);
  const [activeAction, setActiveAction] = useState("");
  const [error, setError] = useState("");

  function resetProjectFlow(): void {
    setJobId("");
    setProjectId("");
    setJobStatus("");
    setProjectModel(null);
    setDraftSpec(null);
    setAppearance({ finish: "matte" });
    setValidation(null);
  }

  function clearError(): void {
    setError("");
  }

  function getProjectByKey(projectKey: string): ProjectRecord | undefined {
    return projects.find((project) => project.key === projectKey);
  }

  function createNewProject(name: string): string {
    const normalizedName = name.trim() || `Project ${projects.length + 1}`;
    const key = crypto.randomUUID();
    const created: ProjectRecord = {
      key,
      name: normalizedName,
      createdAt: new Date().toISOString(),
      analysisStatus: "idle",
      imageCount: 0,
      analyzedCount: 0,
      failedCount: 0,
    };

    setProjects((prev) => [created, ...prev]);
    setSelectedProjectKey(key);
    setProjectName(normalizedName);
    return key;
  }

  function selectProject(projectKey: string): void {
    // projectKey is now the backend project_id UUID
    const existingByBackendId = projects.find((p) => p.backendProjectId === projectKey);
    if (existingByBackendId) {
      setSelectedProjectKey(existingByBackendId.key);
      setProjectName(existingByBackendId.name);
      setProjectId(projectKey);
      if (existingByBackendId.jobId) {
        setJobId(existingByBackendId.jobId);
      }
      return;
    }

    // Not in local cache — load from backend
    setProjectId(projectKey);
    setSelectedProjectKey(projectKey);
    getProject(projectKey)
      .then((res) => {
        const name = res.model.product.name;
        setProjectName(name);
        setProjectModel(res.model);
        setDraftSpec(applyDraftFromModel(res.model));
        // Insert a synthetic local record so other consumers work
        setProjects((prev) => {
          if (prev.some((p) => p.backendProjectId === projectKey)) return prev;
          return [
            {
              key: projectKey,
              name,
              createdAt: new Date().toISOString(),
              analysisStatus: "idle",
              backendProjectId: projectKey,
              imageCount: 0,
              analyzedCount: 0,
              failedCount: 0,
            },
            ...prev,
          ];
        });
      })
      .catch(() => {
        // project not found — leave state as-is, view will show 404
      });
  }

  function ensureSelectedProject(): string {
    if (selectedProjectKey) {
      return selectedProjectKey;
    }
    return createNewProject(projectName);
  }

  function updateProjectRecord(projectKey: string, updates: Partial<ProjectRecord>): void {
    setProjects((prev) =>
      prev.map((project) => (project.key === projectKey ? { ...project, ...updates } : project))
    );
  }

  function setActiveJob(nextJobId: string): void {
    setJobId(nextJobId);
    setJobStatus("complete");
    if (selectedProjectKey) {
      updateProjectRecord(selectedProjectKey, {
        jobId: nextJobId,
        analysisStatus: "complete",
        analyzedCount: 1,
        failedCount: 0,
      });
    }
  }

  function addImageFiles(files: File[]): void {
    const maxSizeBytes = 10 * 1024 * 1024;
    const validFiles = files.filter((file) => file.type.startsWith("image/") && file.size <= maxSizeBytes);

    if (validFiles.length < files.length) {
      setError("Some files were skipped. Only image files up to 10 MB are accepted.");
    }

    const nextItems: QueuedImage[] = validFiles.map((file) => ({
      id: `${file.name}-${file.size}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      name: file.name,
      size: file.size,
      file,
      previewUrl: URL.createObjectURL(file),
      status: "ready",
    }));

    setQueuedImages((prev) => [...prev, ...nextItems]);
  }

  function removeImageFile(id: string): void {
    setQueuedImages((prev) => {
      const target = prev.find((item) => item.id === id);
      if (target) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((item) => item.id !== id);
    });
  }

  function clearImageQueue(): void {
    setQueuedImages((prev) => {
      prev.forEach((item) => URL.revokeObjectURL(item.previewUrl));
      return [];
    });
  }

  async function onHealthCheck(): Promise<void> {
    setError("");
    setActiveAction("Checking backend health...");
    try {
      const result = await healthCheck();
      setBackendStatus(result.status === "ok" ? "ok" : "unknown");
    } catch (e) {
      setBackendStatus("down");
      setError((e as Error).message);
    } finally {
      setActiveAction("");
    }
  }

  useEffect(() => {
    void onHealthCheck();
  }, []);

  async function onAnalyzeUrl(event: FormEvent): Promise<void> {
    event.preventDefault();
    setError("URL analysis is disabled. Upload project assets and run analysis jobs.");
  }

  async function onAnalyzeQueuedImages(): Promise<void> {
    if (queuedImages.length === 0) {
      setError("Add image files before running batch analysis.");
      return;
    }

    if (!projectId) {
      setError("Create backend project first.");
      return;
    }

    setError("");
    setValidation(null);
    setJobId("");
    setJobStatus("");
    setActiveAction(`Analyzing ${queuedImages.length} image(s)...`);
    const projectKey = ensureSelectedProject();

    updateProjectRecord(projectKey, {
      name: projectName,
      analysisStatus: "analyzing",
      imageCount: queuedImages.length,
      analyzedCount: 0,
      failedCount: 0,
      lastError: undefined,
    });

    let firstSuccessfulJobId = "";
    let analyzedCount = 0;
    let failedCount = 0;

    for (const item of queuedImages) {
      setQueuedImages((prev) =>
        prev.map((row) =>
          row.id === item.id
            ? {
                ...row,
                status: "analyzing",
                message: undefined,
              }
            : row
        )
      );

      try {
        const asset = await uploadProjectAsset(projectId, item.file);
        const createdJob = await createProjectAnalysisJob(projectId, asset.asset_id);
        const jobResult = await getJob(createdJob.job_id);
        const latestProject = await getProject(projectId);
        setProjectModel(latestProject.model);
        setDraftSpec(applyDraftFromModel(latestProject.model));

        if (!firstSuccessfulJobId) {
          firstSuccessfulJobId = createdJob.job_id;
          setJobId(createdJob.job_id);
          setJobStatus(jobResult.status);
        }
        analyzedCount += 1;
        updateProjectRecord(projectKey, {
          analyzedCount,
          failedCount,
          analysisStatus: "analyzing",
          jobId: firstSuccessfulJobId || createdJob.job_id,
        });

        setQueuedImages((prev) =>
          prev.map((row) =>
            row.id === item.id
              ? {
                  ...row,
                  status: "complete",
                  jobId: createdJob.job_id,
                  message: "Analysis complete",
                }
              : row
          )
        );
      } catch (e) {
        failedCount += 1;
        updateProjectRecord(projectKey, {
          analyzedCount,
          failedCount,
          analysisStatus: "analyzing",
          lastError: (e as Error).message,
        });
        setQueuedImages((prev) =>
          prev.map((row) =>
            row.id === item.id
              ? {
                  ...row,
                  status: "failed",
                  message: (e as Error).message,
                }
              : row
          )
        );
      }
    }

    if (!firstSuccessfulJobId) {
      setError("All image analyses failed. Check backend availability or use image URLs.");
      updateProjectRecord(projectKey, {
        analysisStatus: "failed",
        analyzedCount,
        failedCount,
      });
    } else {
      updateProjectRecord(projectKey, {
        analysisStatus: "complete",
        jobId: firstSuccessfulJobId,
        analyzedCount,
        failedCount,
      });
    }

    setActiveAction("");
  }

  async function onCreateProject(): Promise<void> {
    if (projectId) {
      setError("Backend project already exists for this project.");
      return;
    }

    setError("");
    setValidation(null);
    setActiveAction("Creating backend project...");

    try {
      const created = await createProject(projectName);
      setProjectId(created.project_id);
      setProjectModel(created.model);
      setDraftSpec(applyDraftFromModel(created.model));
      if (selectedProjectKey) {
        updateProjectRecord(selectedProjectKey, {
          backendProjectId: created.project_id,
          name: created.model.product.name,
        });
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setActiveAction("");
    }
  }

  async function onUpdateProject(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!projectId || !draftSpec) {
      setError("Create a project before editing dimensions.");
      return;
    }

    setError("");
    setValidation(null);
    setActiveAction("Updating project dimensions...");

    try {
      const updated = await updateProject(projectId, draftSpec);
      setProjectModel(updated.model);
      setDraftSpec(applyDraftFromModel(updated.model));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setActiveAction("");
    }
  }

  async function onValidateProject(): Promise<void> {
    if (!projectId) {
      setError("Create a project first.");
      return;
    }

    setError("");
    setActiveAction("Validating project rules...");
    try {
      const result = await validateProject(projectId);
      setValidation(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setActiveAction("");
    }
  }

  const value = useMemo<WorkflowContextValue>(
    () => ({
      backendStatus,
      imageUrl,
      setImageUrl,
      projects,
      selectedProjectKey,
      createNewProject,
      selectProject,
      getProjectByKey,
      queuedImages,
      addImageFiles,
      removeImageFile,
      clearImageQueue,
      setActiveJob,
      projectName,
      setProjectName,
      jobId,
      projectId,
      jobStatus,
      projectModel,
      draftSpec,
      setDraftSpec,
      appearance,
      setAppearance,
      validation,
      activeAction,
      error,
      clearError,
      onHealthCheck,
      onAnalyzeUrl,
      onAnalyzeQueuedImages,
      onCreateProject,
      onUpdateProject,
      onValidateProject,
    }),
    [
      backendStatus,
      imageUrl,
      projects,
      selectedProjectKey,
      queuedImages,
      projectName,
      jobId,
      projectId,
      jobStatus,
      projectModel,
      draftSpec,
      appearance,
      validation,
      activeAction,
      error,
    ]
  );

  return <WorkflowContext.Provider value={value}>{children}</WorkflowContext.Provider>;
}

export function useVisionWorkflow(): WorkflowContextValue {
  const value = useContext(WorkflowContext);
  if (!value) {
    throw new Error("useVisionWorkflow must be used within WorkflowProvider");
  }
  return value;
}
