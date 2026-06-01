import { FormEvent } from "react";

import { ProductSpec } from "../api/backend";
import { ImageBatchUploader } from "./ImageBatchUploader";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";
import { SectionCard } from "./SectionCard";

function parseNumberInput(raw: string): number {
  const value = Number(raw);
  return Number.isNaN(value) ? 0 : value;
}

type DraftSpec = Pick<
  ProductSpec,
  "target_width" | "target_height" | "target_depth" | "shelf_count" | "material_thickness"
>;

function updateDraftValue(
  draft: DraftSpec | null,
  key: keyof DraftSpec,
  raw: string
): DraftSpec {
  return {
    target_width: key === "target_width" ? parseNumberInput(raw) : draft?.target_width ?? 0,
    target_height: key === "target_height" ? parseNumberInput(raw) : draft?.target_height ?? 0,
    target_depth: key === "target_depth" ? parseNumberInput(raw) : draft?.target_depth ?? 0,
    material_thickness: key === "material_thickness" ? parseNumberInput(raw) : draft?.material_thickness ?? 18,
    shelf_count: key === "shelf_count" ? parseNumberInput(raw) : draft?.shelf_count ?? 0,
  };
}

export function WorkspaceFlow(): React.JSX.Element {
  const {
    queuedImages,
    addImageFiles,
    removeImageFile,
    clearImageQueue,
    setActiveJob,
    projectName,
    onAnalyzeUrl,
    onAnalyzeQueuedImages,
    onUpdateProject,
    onValidateProject,
    jobId,
    jobStatus,
    projectId,
    draftSpec,
    setDraftSpec,
    appearance,
    setAppearance,
  } = useVisionWorkflow();

  function handleNumberField(
    key: keyof DraftSpec,
    event: FormEvent<HTMLInputElement>
  ): void {
    setDraftSpec(updateDraftValue(draftSpec, key, event.currentTarget.value));
  }

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
          onUseAsActiveJob={setActiveJob}
        />
        {jobId ? (
          <p>
            Latest job: <strong>{jobId}</strong> ({jobStatus || "unknown"})
          </p>
        ) : null}
      </SectionCard>

      <SectionCard
        title="2. Parametric generation"
        subtitle="Model reflects latest analyzed asset dimensions"
      >
        <p style={{ margin: 0 }}>
          Run at least one analysis job to refresh the project model from inference output.
        </p>
      </SectionCard>

      <SectionCard
        title="3. Topology adjustments"
        subtitle="Sliders and fields mirror activity diagram loop"
      >
        <form onSubmit={onUpdateProject}>
          <label htmlFor="width">Target width</label>
          <input
            id="width"
            type="number"
            value={draftSpec?.target_width ?? ""}
            onChange={(e) => handleNumberField("target_width", e)}
            disabled={!projectId}
            min={1}
            step={1}
          />

          <label htmlFor="height">Target height</label>
          <input
            id="height"
            type="number"
            value={draftSpec?.target_height ?? ""}
            onChange={(e) => handleNumberField("target_height", e)}
            disabled={!projectId}
            min={1}
            step={1}
          />

          <label htmlFor="depth">Target depth</label>
          <input
            id="depth"
            type="number"
            value={draftSpec?.target_depth ?? ""}
            onChange={(e) => handleNumberField("target_depth", e)}
            disabled={!projectId}
            min={1}
            step={1}
          />

          <label htmlFor="shelf-count">Shelf count</label>
          <input
            id="shelf-count"
            type="number"
            value={draftSpec?.shelf_count ?? ""}
            onChange={(e) => handleNumberField("shelf_count", e)}
            disabled={!projectId}
            min={0}
            step={1}
          />

          <label htmlFor="material-thickness">Material thickness (mm)</label>
          <input
            id="material-thickness"
            type="number"
            value={draftSpec?.material_thickness ?? ""}
            onChange={(e) => handleNumberField("material_thickness", e)}
            disabled={!projectId}
            min={8}
            max={50}
            step={1}
          />

          <label htmlFor="finish">Finish</label>
          <select
            id="finish"
            value={appearance.finish}
            onChange={(event) => setAppearance({ finish: event.currentTarget.value as "matte" | "satin" | "gloss" })}
            disabled={!projectId}
          >
            <option value="matte">Matte</option>
            <option value="satin">Satin</option>
            <option value="gloss">Gloss</option>
          </select>

          <p style={{ margin: 0, fontSize: "0.85rem", opacity: 0.8 }}>
            Finish selection is currently frontend-only and will be sent once backend appearance fields are enabled.
          </p>

          <button type="submit" disabled={!projectId || !draftSpec}>
            Save dimension update
          </button>
        </form>
      </SectionCard>

      <SectionCard title="4. Validation" subtitle="Rule validator feedback from backend engine">
        <button onClick={onValidateProject} disabled={!projectId}>
          Validate project
        </button>
      </SectionCard>
    </div>
  );
}
