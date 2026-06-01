import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createProjectAnalysisJob,
  createProject,
  healthCheck,
  uploadProjectAsset,
  updateProject,
  validateProject,
} from "./backend";

describe("backend API client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("calls health endpoint and returns payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ status: "ok" }),
      })
    );

    const result = await healthCheck();
    expect(result.status).toBe("ok");
  });

  it("surfaces backend error envelope message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        headers: { get: () => "application/json" },
        json: async () => ({ error: { code: "HTTP_404", message: "Job not found" } }),
      })
    );

    await expect(createProjectAnalysisJob("p1", "a1")).rejects.toThrow("Job not found");
  });

  it("uses correct methods and payloads for create-update-validate", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({
          project_id: "p1",
          model: { product: { name: "A", target_width: 800, target_height: 1200, target_depth: 450, material_thickness: 18, shelf_count: 3 }, components: [], hardware: [], warnings: [] },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({
          project_id: "p1",
          model: { product: { name: "A", target_width: 900, target_height: 1200, target_depth: 450, material_thickness: 18, shelf_count: 3 }, components: [], hardware: [], warnings: [] },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ valid: true, errors: [], warnings: [] }),
      });

    vi.stubGlobal("fetch", fetchMock);

    await createProject("My Project");
    await updateProject("p1", { target_width: 900 });
    await validateProject("p1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("/api/v1/projects"),
      expect.objectContaining({ method: "POST" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining("/api/v1/projects/p1"),
      expect.objectContaining({ method: "PATCH" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      expect.stringContaining("/api/v1/projects/p1/validate"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("uploads project asset and creates project analysis job", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => ({ asset_id: "a1", file_name: "cabinet.jpg", content_type: "image/jpeg", size_bytes: 16 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const file = new File(["fake image bytes"], "cabinet.jpg", { type: "image/jpeg" });
    const result = await uploadProjectAsset("p1", file);

    expect(result.asset_id).toBe("a1");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/projects/p1/assets"),
      expect.objectContaining({ method: "POST", body: expect.any(FormData) })
    );

    fetchMock.mockResolvedValueOnce({
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => ({ project_id: "p1", asset_id: "a1", job_id: "j1", status: "complete" }),
    });

    const job = await createProjectAnalysisJob("p1", "a1");
    expect(job.job_id).toBe("j1");
  });
});
