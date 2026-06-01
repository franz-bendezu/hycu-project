import { ImageBatchUploader } from "./ImageBatchUploader";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";
import { SectionCard } from "./SectionCard";

export function WorkspaceFlow(): React.JSX.Element {
  const {
    projectJobs,
    queuedImages,
    addImageFiles,
    removeImageFile,
    clearImageQueue,
    setActiveJob,
    onAnalyzeQueuedImages,
    jobId,
    jobStatus,
  } = useVisionWorkflow();

  return (
    <div className="flow-column">
      <SectionCard
        title="1. Upload assets and run analysis jobs"
        subtitle="Each uploaded image is saved to project assets before inference"
      >
        <ImageBatchUploader
          queuedImages={queuedImages}
          onAddFiles={addImageFiles}
          onRemoveFile={removeImageFile}
          onClear={clearImageQueue}
          onAnalyze={onAnalyzeQueuedImages}
        />
        {jobId ? (
          <p>
            Latest job: <strong>{jobId}</strong> ({jobStatus || "unknown"})
          </p>
        ) : null}
        {projectJobs.length > 0 ? (
          <div style={{ marginTop: "0.8rem", display: "grid", gap: "0.55rem" }}>
            <p style={{ margin: 0 }}>
              <strong>Results per job</strong>
            </p>
            {projectJobs.map((job) => (
              <div
                key={job.job_id}
                style={{
                  border: "1px solid rgba(20, 38, 68, 0.14)",
                  borderRadius: "12px",
                  padding: "0.65rem 0.75rem",
                  display: "grid",
                  gap: "0.35rem",
                }}
              >
                <p style={{ margin: 0 }}>
                  Job: <strong>{job.job_id}</strong> ({job.status})
                </p>
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
                  {job.job_id === jobId ? "Active" : "Use this job"}
                </button>
              </div>
            ))}
          </div>
        ) : null}
      </SectionCard>

      <SectionCard
        title="2. Parametric generation"
        subtitle="Model reflects latest analyzed asset dimensions"
      >
        <p style={{ margin: 0 }}>
          Run analysis here, then switch to the Model route for topology adjustments.
        </p>
      </SectionCard>

    </div>
  );
}
