const BASE_URL =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_BACKEND_URL ||
  "http://localhost:8000";

export type HealthResponse = { status: string };

export type AnalyzeResponse = {
  job_id: string;
  status: "queued" | "complete" | "failed";
};

export enum InferenceComponentKind {
  PANEL = "panel",
  SUPPORT = "support",
  HARDWARE = "hardware",
  ASSEMBLY = "assembly",
}

export enum InferenceDoorType {
  HINGED = "hinged",
  SLIDING = "sliding",
  UNKNOWN = "unknown",
}

export enum InferenceJointType {
  CAM_LOCK = "cam_lock",
  SHELF_PIN = "shelf_pin",
  SCREW = "screw",
  HINGE = "hinge",
  SLIDING_TRACK = "sliding_track",
  TELESCOPIC_SLIDE = "telescopic_slide",
  BRACKET = "bracket",
}

export enum InferenceInteriorVisibility {
  INTERIOR_NOT_VISIBLE = "interior_not_visible",
  INTERIOR_PARTIALLY_VISIBLE = "interior_partially_visible",
  INTERIOR_FULLY_VISIBLE = "interior_fully_visible",
}

export enum InferenceHardwareCode {
  CAM_LOCK_15MM = "CAM_LOCK_15MM",
  SHELF_PIN_5MM = "SHELF_PIN_5MM",
  HINGE_SOFT_CLOSE_110 = "HINGE_SOFT_CLOSE_110",
  WOOD_SCREW_4X16 = "WOOD_SCREW_4X16",
  SLIDING_DOOR_TRACK_SET = "SLIDING_DOOR_TRACK_SET",
  TELESCOPIC_SLIDE_400 = "TELESCOPIC_SLIDE_400",
  WOOD_SCREW_4X40 = "WOOD_SCREW_4X40",
  CORNER_BRACKET_40 = "CORNER_BRACKET_40",
  HARDWARE_REVIEW_REQUIRED = "HARDWARE_REVIEW_REQUIRED",
}

export enum ProductInferredType {
  CABINET = "cabinet",
  WARDROBE = "wardrobe",
  BOOKCASE = "bookcase",
  DESK = "desk",
  TABLE = "table",
  SHELF = "shelf",
  NIGHTSTAND = "nightstand",
  DRESSER = "dresser",
  SIDEBOARD = "sideboard",
  TV_STAND = "tv_stand",
}

export enum ProjectComponentKind {
  LEFT_SIDE = "left_side",
  RIGHT_SIDE = "right_side",
  TOP_PANEL = "top_panel",
  BOTTOM_PANEL = "bottom_panel",
  BACK_PANEL = "back_panel",
  SHELF = "shelf",
  DIVIDER_PANEL = "divider_panel",
  FRONT_PANEL = "front_panel",
  DOOR_PANEL = "door_panel",
  DRAWER_FRONT = "drawer_front",
  LEFT_LEG_FRONT = "left_leg_front",
  RIGHT_LEG_FRONT = "right_leg_front",
  LEFT_LEG_BACK = "left_leg_back",
  RIGHT_LEG_BACK = "right_leg_back",
}

export enum ProjectComponentCategory {
  STRUCTURAL = "structural",
  FRONT = "front",
  INTERNAL = "internal",
  SUPPORT = "support",
}

export enum ProjectJointRule {
  OVERLAP = "overlap",
  INSET = "inset",
  BETWEEN = "between",
  FLUSH_BACK = "flush_back",
  MOUNT = "mount",
}

export type InferenceOutput = {
  detected_type: string;
  confidence: number;
  suggested_width: number;
  suggested_height: number;
  suggested_depth: number;
  components?: Array<{ id: string; name: string; kind: InferenceComponentKind; quantity: number }>;
  component_index?: Record<string, { id: string; name: string; kind: InferenceComponentKind; quantity: number }>;
  interior?: {
    visibility: InferenceInteriorVisibility;
    coverage_ratio: number;
    unknown_interior: boolean;
  };
  door?: {
    type: InferenceDoorType;
    count_uncertain: boolean;
  };
  uncertainty?: {
    hardware_uncertain: boolean;
  };
  joints?: Array<{
    id: string;
    parent_component_id: string;
    child_component_id: string;
    joint_type: InferenceJointType;
    count: number;
  }>;
  hardware?: Array<{
    code: InferenceHardwareCode;
    qty: number;
    reason?: string | null;
  }>;
  image_url: string;
  images_analyzed?: number;
  image_results?: Array<{
    detected_type: string;
    confidence: number;
    suggested_width: number;
    suggested_height: number;
    suggested_depth: number;  
    components: Array<{ id: string; name: string; kind: InferenceComponentKind; quantity: number }>;
    component_index?: Record<string, { id: string; name: string; kind: InferenceComponentKind; quantity: number }>;
    interior?: {
      visibility: InferenceInteriorVisibility;
      coverage_ratio: number;
      unknown_interior: boolean;
    };
    door?: {
      type: InferenceDoorType;
      count_uncertain: boolean;
    };
    uncertainty?: {
      hardware_uncertain: boolean;
    };
    joints?: Array<{
      id: string;
      parent_component_id: string;
      child_component_id: string;
      joint_type: InferenceJointType;
      count: number;
    }>;
    hardware?: Array<{
      code: InferenceHardwareCode;
      qty: number;
      reason?: string | null;
    }>;
    image_url: string;
  }>;
};

export type JobResponse = {
  job_id: string;
  status: "queued" | "complete" | "failed";
  result: InferenceOutput | null;
  project_id?: string | null;
  asset_id?: string | null;
  asset_results?: Array<{
    job_id: string;
    asset_id: string;
    status: "queued" | "complete" | "failed";
    result: InferenceOutput["image_results"] extends Array<infer T> ? T | null : null;
  }> | null;
};

export type ProjectJobsResponse = {
  project_id: string;
  jobs: JobResponse[];
};

export type ProductSpec = {
  id?: string | null;
  sku?: string | null;
  name: string;
  inferred_type?: ProductInferredType;
  target_width: number;
  target_height: number;
  target_depth: number;
  material_thickness: number;
  shelf_count: number;
  divider_count?: number;
  door_count?: number;
  drawer_count?: number;
};

export type ProjectAppearance = {
  finish: "matte" | "satin" | "gloss";
};

export type ProjectModel = {
  product: ProductSpec;
  materials?: Array<{
    id: string;
    thickness_mm: number;
    texture_map_url?: string | null;
  }>;
  components: Array<{
    id: string;
    kind: ProjectComponentKind;
    category?: ProjectComponentCategory;
    material_id?: string | null;
    width: number;
    height: number;
    depth: number;
    pos_x: number;
    pos_y: number;
    pos_z: number;
    rot_x_deg: number;
    rot_y_deg: number;
    rot_z_deg: number;
    faces?: Array<{
      id: string;
      component_id: string;
      normal: "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
    }>;
  }>;
  hardware: Array<{
    id?: string | null;
    code: string;
    qty: number;
    anchor?: string | null;
    mesh_path?: string | null;
    svg_path?: string | null;
    joint_type?: string | null;
    mount_targets?: Array<{
      component_id: string;
      face: "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
      local_x: number;
      local_y: number;
      local_z: number;
      normal_offset_mm?: number;
    }>;
  }>;
  joints?: Array<{
    id: string;
    parent_face_id: string;
    child_face_id: string;
    joint_rule?: ProjectJointRule | null;
    offset_u: number;
    offset_v: number;
    clearance: number;
  }>;
  features?: Array<{
    component_id: string;
    face_index: number;
    u_coord: number;
    v_coord: number;
    operation_type: string;
  }>;
  warnings: string[];
};

export type CreateProjectResponse = {
  project_id: string;
  model: ProjectModel;
  validation: ValidateResponse;
};

export type CreateProjectAssetResponse = {
  asset_id: string;
  file_name: string;
  content_type: string;
  size_bytes: number;
};

export type ProjectAssetSummary = {
  asset_id: string;
  file_name: string;
  content_type: string;
  size_bytes: number;
  image_url?: string | null;
};

export type ProjectAssetsResponse = {
  project_id: string;
  assets: ProjectAssetSummary[];
};

export type CreateProjectJobResponse = {
  project_id: string;
  asset_id?: string | null;
  asset_count: number;
  job_id: string;
  status: "queued" | "complete" | "failed";
  validation: ValidateResponse;
};

export type ProjectResponse = {
  project_id: string;
  model: ProjectModel;
  validation: ValidateResponse;
};

export type ProjectSummary = {
  project_id: string;
  name: string;
  created_at: string;
};

export type ProjectsListResponse = {
  projects: ProjectSummary[];
};

export type ValidateResponse = {
  valid: boolean;
  errors: string[];
  warnings: string[];
  checks?: {
    load_span?: {
      is_safe: boolean;
      message: string;
      recommended_dividers?: number;
    };
    joints?: Array<{
      id: string;
      is_valid: boolean;
      message: string;
    }>;
  };
};

export type ExportArtifactKind = "blueprint" | "bom" | "nesting" | "package";

const EXPORT_KIND_CONFIG: Record<
  ExportArtifactKind,
  { path: string; fallbackName: string }
> = {
  blueprint: {
    path: "blueprint.pdf",
    fallbackName: "blueprint.pdf",
  },
  bom: {
    path: "bom.csv",
    fallbackName: "bom.csv",
  },
  nesting: {
    path: "nesting.dxf",
    fallbackName: "nesting.dxf",
  },
  package: {
    path: "export",
    fallbackName: "fabrication-package.zip",
  },
};

type ApiErrorResponse = {
  error?: {
    code?: string;
    message?: string;
  };
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init);
  const contentType = response.headers.get("content-type") || "";
  const hasJson = contentType.includes("application/json");
  const payload = hasJson ? await response.json() : null;

  if (!response.ok) {
    const apiError = payload as ApiErrorResponse | null;
    const message = apiError?.error?.message || `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}

export async function healthCheck(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export async function analyzeImages(imageUrls: string[]): Promise<AnalyzeResponse> {
  return requestJson<AnalyzeResponse>("/api/v1/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_urls: imageUrls }),
  });
}

export async function analyzeImageFilesBatch(
  files: File[],
  projectName?: string
): Promise<AnalyzeResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  if (projectName?.trim()) {
    form.append("project_name", projectName.trim());
  }

  return requestJson<AnalyzeResponse>("/api/v1/analyze-upload-batch", {
    method: "POST",
    body: form,
  });
}

export async function getJob(jobId: string): Promise<JobResponse> {
  return requestJson<JobResponse>(`/api/v1/jobs/${jobId}`);
}

export async function createProject(name?: string): Promise<CreateProjectResponse> {
  return requestJson<CreateProjectResponse>("/api/v1/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function uploadProjectAsset(projectId: string, file: File): Promise<CreateProjectAssetResponse> {
  const form = new FormData();
  form.append("file", file);
  return requestJson<CreateProjectAssetResponse>(`/api/v1/projects/${projectId}/assets`, {
    method: "POST",
    body: form,
  });
}

export async function listProjectAssets(projectId: string): Promise<ProjectAssetsResponse> {
  return requestJson<ProjectAssetsResponse>(`/api/v1/projects/${projectId}/assets`);
}

export async function deleteProjectAsset(projectId: string, assetId: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/api/v1/projects/${projectId}/assets/${assetId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as ApiErrorResponse;
      throw new Error(payload.error?.message || `Request failed with status ${response.status}`);
    }
    throw new Error(`Request failed with status ${response.status}`);
  }
}

export async function createProjectAnalysisJob(
  projectId: string
): Promise<CreateProjectJobResponse> {
  return requestJson<CreateProjectJobResponse>(`/api/v1/projects/${projectId}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export async function listProjectJobs(projectId: string): Promise<ProjectJobsResponse> {
  return requestJson<ProjectJobsResponse>(`/api/v1/projects/${projectId}/jobs`);
}

export async function getProject(projectId: string): Promise<ProjectResponse> {
  return requestJson<ProjectResponse>(`/api/v1/projects/${projectId}`);
}

export async function updateProject(
  projectId: string,
  updates: Partial<
    Pick<
      ProductSpec,
      "target_width" | "target_height" | "target_depth" | "shelf_count" | "material_thickness"
    >
  >
): Promise<ProjectResponse> {
  return requestJson<ProjectResponse>(`/api/v1/projects/${projectId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
}

export async function validateProject(projectId: string): Promise<ValidateResponse> {
  return requestJson<ValidateResponse>(`/api/v1/projects/${projectId}/validate`, {
    method: "POST",
  });
}

export async function listProjects(): Promise<ProjectsListResponse> {
  return requestJson<ProjectsListResponse>("/api/v1/projects");
}

export async function downloadProjectArtifact(
  projectId: string,
  kind: ExportArtifactKind
): Promise<{ blob: Blob; fileName: string }> {
  const config = EXPORT_KIND_CONFIG[kind];
  const response = await fetch(`${BASE_URL}/api/v1/projects/${projectId}/${config.path}`);

  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as ApiErrorResponse;
      throw new Error(payload.error?.message || `Request failed with status ${response.status}`);
    }
    throw new Error(`Request failed with status ${response.status}`);
  }

  const contentDisposition = response.headers.get("content-disposition") || "";
  const fileNameMatch = contentDisposition.match(/filename=\"?([^\";]+)\"?/i);
  const fileName = fileNameMatch?.[1] || config.fallbackName;

  return {
    blob: await response.blob(),
    fileName,
  };
}
