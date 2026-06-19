import { SectionCard } from "../components/SectionCard";
import { StatusPill } from "../components/StatusPill";
import { Button } from "../components/ui/button";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";

function toneForStatus(status: "queued" | "complete" | "failed"): "neutral" | "warning" | "ok" | "error" {
  if (status === "complete") {
    return "ok";
  }
  if (status === "failed") {
    return "error";
  }
  return "warning";
}

export function JobsView(): React.JSX.Element {
  const { projectJobs, jobId, setActiveJob } = useVisionWorkflow();

  return (
    <section className="content-grid" style={{ gridTemplateColumns: "1fr" }}>
      <SectionCard
        title="Job Results"
        subtitle="Batch jobs created for this project"
      >
        {projectJobs.length === 0 ? (
          <div className="empty-state-panel" role="status" aria-live="polite">
            <h4>No jobs yet</h4>
            <p>Run image analysis from the Analysis tab and your job history will appear here.</p>
          </div>
        ) : (
          <div className="jobs-grid">
            {projectJobs.map((job) => (
              <div
                key={job.job_id}
                className="job-card"
              >
                <p style={{ margin: 0 }}>
                  Job ID: <strong>{job.job_id}</strong>
                </p>
                <StatusPill label={job.status} tone={toneForStatus(job.status)} />
                <p style={{ margin: 0 }}>
                  Detected type: <strong>{job.result?.detected_type || "-"}</strong>
                </p>
                <p style={{ margin: 0 }}>
                  Images analyzed: <strong>{job.result?.images_analyzed || 0}</strong>
                </p>
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => setActiveJob(job.job_id)}
                  disabled={job.job_id === jobId}
                >
                  {job.job_id === jobId ? "Active job" : "Use as active job"}
                </Button>
              </div>
            ))}
          </div>
        )}
      </SectionCard>
    </section>
  );
}