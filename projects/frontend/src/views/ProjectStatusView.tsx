import { useEffect } from "react";
import { Link, useParams } from "react-router-dom";

import { SectionCard } from "../components/SectionCard";
import { StatusPill } from "../components/StatusPill";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";

function statusTone(status: "idle" | "analyzing" | "complete" | "failed"): "neutral" | "warning" | "ok" | "error" {
  if (status === "complete") {
    return "ok";
  }
  if (status === "failed") {
    return "error";
  }
  if (status === "analyzing") {
    return "warning";
  }
  return "neutral";
}

export function ProjectStatusView(): React.JSX.Element {
  const { projectKey = "" } = useParams();
  const { getProjectByKey, selectProject } = useVisionWorkflow();

  useEffect(() => {
    if (projectKey) {
      selectProject(projectKey);
    }
  }, [projectKey]);

  const project = projectKey ? getProjectByKey(projectKey) : undefined;

  if (!project) {
    return (
      <section className="content-grid" style={{ gridTemplateColumns: "1fr" }}>
        <SectionCard title="Project not found">
          <p>This project route does not exist.</p>
          <Link to="/" className="button secondary">
            Back to home
          </Link>
        </SectionCard>
      </section>
    );
  }

  return (
    <section className="content-grid" style={{ gridTemplateColumns: "1fr" }}>
      <SectionCard title={project.name} subtitle="Project analysis status and routing hub">
        <div style={{ display: "grid", gap: "0.55rem" }}>
          <p style={{ margin: 0 }}>
            Created: <strong>{new Date(project.createdAt).toLocaleString()}</strong>
          </p>
          <StatusPill label={`Analysis: ${project.analysisStatus}`} tone={statusTone(project.analysisStatus)} />
          <p style={{ margin: 0 }}>
            Images queued: <strong>{project.imageCount}</strong>
          </p>
          <p style={{ margin: 0 }}>
            Completed: <strong>{project.analyzedCount}</strong> | Failed: <strong>{project.failedCount}</strong>
          </p>
          <p style={{ margin: 0 }}>
            Job ID: <strong>{project.jobId || "-"}</strong>
          </p>
          <p style={{ margin: 0 }}>
            Backend Project ID: <strong>{project.backendProjectId || "-"}</strong>
          </p>
          {project.lastError ? <p style={{ color: "#9f1f1f", margin: 0 }}>Last error: {project.lastError}</p> : null}

          <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap" }}>
            <Link to={`/projects/${project.key}/workspace`} className="button">
              Open workspace
            </Link>
            <Link to={`/projects/${project.key}/fabrication-output`} className="button secondary">
              View fabrication output
            </Link>
            <Link to="/" className="button secondary">
              Back to home
            </Link>
          </div>
        </div>
      </SectionCard>
    </section>
  );
}
