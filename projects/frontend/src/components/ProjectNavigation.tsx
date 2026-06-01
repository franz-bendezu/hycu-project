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
          Status
        </NavLink>
        <NavLink
          to={`/projects/${projectKey}/workspace`}
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          Workspace
        </NavLink>
        <NavLink
          to={`/projects/${projectKey}/proposal-coverage`}
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          Diagram Coverage
        </NavLink>
        <NavLink
          to={`/projects/${projectKey}/fabrication-output`}
          className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
        >
          Fabrication Output
        </NavLink>
      </nav>
    </section>
  );
}
