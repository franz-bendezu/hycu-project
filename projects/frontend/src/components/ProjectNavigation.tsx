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

type NavItem = {
  title: string;
  subtitle: string;
  to: string;
  step: string;
  end?: boolean;
};

export function ProjectNavigation(): React.JSX.Element | null {
  const { projectKey = "" } = useParams();
  const { getProjectByKey } = useVisionWorkflow();

  if (!projectKey) {
    return null;
  }

  const project = getProjectByKey(projectKey);
  const navItems: NavItem[] = [
    {
      title: "Overview",
      subtitle: "Status and checks",
      to: `/projects/${projectKey}`,
      step: "01",
      end: true,
    },
    {
      title: "Analysis",
      subtitle: "Upload and detect",
      to: `/projects/${projectKey}/workspace`,
      step: "02",
    },
    {
      title: "Jobs",
      subtitle: "History and results",
      to: `/projects/${projectKey}/jobs`,
      step: "03",
    },
    {
      title: "Model",
      subtitle: "3D and topology",
      to: `/projects/${projectKey}/model`,
      step: "04",
    },
    {
      title: "Fabrication",
      subtitle: "BOM and blueprint",
      to: `/projects/${projectKey}/fabrication-output`,
      step: "05",
    },
  ];

  return (
    <section className="panel project-nav-shell" aria-label="Project navigation">
      <div className="project-nav-header">
        <div className="project-nav-title-wrap">
          <p className="project-nav-kicker">Current project</p>
          <h2>{project?.name || "Unknown project"}</h2>
          <div className="project-nav-meta">
            <p className="project-nav-project-id">Project: {projectKey}</p>
            <NavLink to="/" className="project-nav-back-link">
              Back to Home
            </NavLink>
          </div>
        </div>
        {project ? (
          <StatusPill label={`Analysis: ${project.analysisStatus}`} tone={toneForStatus(project.analysisStatus)} />
        ) : null}
      </div>

      <nav className="project-nav-row" aria-label="Project workflow sections">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => (isActive ? "project-nav-link active" : "project-nav-link")}
          >
            <span className="project-nav-step">{item.step}</span>
            <span className="project-nav-title">{item.title}</span>
            <span className="project-nav-subtitle">{item.subtitle}</span>
          </NavLink>
        ))}
      </nav>
    </section>
  );
}
