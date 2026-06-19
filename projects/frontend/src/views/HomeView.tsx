import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { createProject, listProjects, ProjectSummary } from "../api/backend";
import { SectionCard } from "../components/SectionCard";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";

export function HomeView(): React.JSX.Element {
  const navigate = useNavigate();
  const { createNewProject } = useVisionWorkflow();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");

  const { data, isLoading, isError } = useQuery({
    queryKey: ["projects"],
    queryFn: () => listProjects().then((r) => r.projects),
  });
  const projects: ProjectSummary[] = data ?? [];

  const mutation = useMutation({
    mutationFn: (projectName: string) => createProject(projectName || undefined),
    onSuccess: (created, projectName) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      createNewProject(projectName || "Project");
      setName("");
      navigate(`/projects/${created.project_id}`);
    },
  });

  function onSubmit(event: FormEvent): void {
    event.preventDefault();
    mutation.mutate(name.trim());
  }

  return (
    <section className="content-grid" style={{ gridTemplateColumns: "1fr" }}>
      <SectionCard
        title="Create New Project"
        subtitle="Project is saved to the backend immediately on creation"
      >
        <form onSubmit={onSubmit}>
          <Label htmlFor="home-project-name">Project name</Label>
          <Input
            id="home-project-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Kitchen Cabinet A"
            required
          />
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? "Creating…" : "Create project"}
          </Button>
        </form>
        {mutation.isError ? (
          <p style={{ color: "var(--color-error, red)" }}>Failed to create project. Is the backend running?</p>
        ) : null}
      </SectionCard>

      <SectionCard title="Projects" subtitle="All projects stored in the backend database">
        {isLoading ? (
          <p className="ux-note">Loading your projects list...</p>
        ) : isError ? (
          <p style={{ color: "var(--color-error, red)" }}>Failed to load projects from server.</p>
        ) : projects.length === 0 ? (
          <div className="empty-state-panel" role="status" aria-live="polite">
            <h4>No projects yet</h4>
            <p>Create your first project above to start image analysis and 3D modeling.</p>
          </div>
        ) : (
          <ul className="project-list">
            {projects.map((project) => (
              <li key={project.project_id} className="project-card">
                <div>
                  <h4 style={{ margin: 0 }}>{project.name}</h4>
                  <p style={{ margin: "0.2rem 0 0", fontSize: "0.8rem", opacity: 0.7 }}>
                    {new Date(project.created_at).toLocaleString()}
                  </p>
                </div>
                <Link className="button" to={`/projects/${project.project_id}`}>
                  Open
                </Link>
              </li>
            ))}
          </ul>
        )}
      </SectionCard>
    </section>
  );
}