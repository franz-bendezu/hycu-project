import { SectionCard } from "../components/SectionCard";
import { StatusPill } from "../components/StatusPill";
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
          <p>No jobs yet. Run analysis from the Analysis route.</p>
        ) : (
          <div style={{ display: "grid", gap: "0.6rem" }}>
            {projectJobs.map((job) => (
              <div
                key={job.job_id}
                style={{
                  border: "1px solid rgba(20, 38, 68, 0.14)",
                  borderRadius: "12px",
                  padding: "0.7rem 0.8rem",
                  display: "grid",
                  gap: "0.35rem",
                }}
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
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setActiveJob(job.job_id)}
                  disabled={job.job_id === jobId}
                >
                  {job.job_id === jobId ? "Active job" : "Use as active job"}
                </button>
              </div>
            ))}
          </div>
        )}
      </SectionCard>
    </section>
  );
}