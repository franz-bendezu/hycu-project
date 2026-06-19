import { NavLink, useParams } from "react-router-dom";

import { StatusPill } from "./StatusPill";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";

function toneForStatus(status: "idle" | "analyzing" | "complete" | "failed"): "neutral" | "warning" | "ok" | "error" {
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

export function ProjectNavigation(): React.JSX.Element | null {
  const { projectKey = "" } = useParams();
  const { getProjectByKey } = useVisionWorkflow();

  if (!projectKey) {
    return null;
  }

  const project = getProjectByKey(projectKey);

  return (
    <section className="panel project-nav-shell" aria-label="Project navigation">
      <div className="project-nav-header">
        <div>
          <p className="project-nav-kicker">Current project</p>
          <h2>{project?.name || "Unknown project"}</h2>
        </div>
        {project ? (
          <StatusPill label={`Analysis: ${project.analysisStatus}`} tone={toneForStatus(project.analysisStatus)} />
        ) : null}
      </div>

      <nav className="project-nav-row">
        <NavLink
          to={`/projects/${projectKey}`}
          end
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          <span className="project-nav-title">Overview</span>
          <span className="project-nav-subtitle">Status and checks</span>
        </NavLink>
        <NavLink
          to={`/projects/${projectKey}/workspace`}
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          <span className="project-nav-title">Analysis</span>
          <span className="project-nav-subtitle">Upload and detect</span>
        </NavLink>
        <NavLink
          to={`/projects/${projectKey}/jobs`}
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          <span className="project-nav-title">Jobs</span>
          <span className="project-nav-subtitle">History and results</span>
        </NavLink>
        <NavLink
          to={`/projects/${projectKey}/model`}
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          <span className="project-nav-title">Model</span>
          <span className="project-nav-subtitle">3D and topology</span>
        </NavLink>
        <NavLink
          to={`/projects/${projectKey}/fabrication-output`}
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          <span className="project-nav-title">Fabrication</span>
          <span className="project-nav-subtitle">BOM and blueprint</span>
        </NavLink>
      </nav>
    </section>
  );
}
