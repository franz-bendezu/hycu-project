import { useState } from "react";

import { buildFabricationPackage } from "../domain/fabrication";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";
import { ParametricViewer } from "./ParametricViewer";
import { SectionCard } from "./SectionCard";
import { StatusPill } from "./StatusPill";

enum ModelViewMode {
  THREE_D = "3d",
  JSON = "json",
}

export function ModelInspector(): React.JSX.Element {
  const { projectId, projectModel, validation, queuedImages, jobId } = useVisionWorkflow();
  const [viewMode, setViewMode] = useState<ModelViewMode>(ModelViewMode.THREE_D);
  const completedAnalyses = queuedImages.filter((item) => item.status === "complete").length;
  const failedAnalyses = queuedImages.filter((item) => item.status === "failed").length;

  return (
    <div className="flow-column">
      <SectionCard
        title="Real-time 3D workspace"
        subtitle="Diagram parity: display model and update visuals after each change"
      >
        <div className="view-mode-switch" role="tablist" aria-label="Model view mode">
          <button
            type="button"
            role="tab"
            aria-selected={viewMode === ModelViewMode.THREE_D}
            className={viewMode === ModelViewMode.THREE_D ? "primary" : "secondary"}
            onClick={() => setViewMode(ModelViewMode.THREE_D)}
          >
            3D
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={viewMode === ModelViewMode.JSON}
            className={viewMode === ModelViewMode.JSON ? "primary" : "secondary"}
            onClick={() => setViewMode(ModelViewMode.JSON)}
          >
            JSON
          </button>
        </div>

        {viewMode === ModelViewMode.THREE_D ? (
          <ParametricViewer model={projectModel} validation={validation} />
        ) : (
          <div className="json-preview-shell">
            <pre className="json-preview-content">
              {projectModel ? JSON.stringify(projectModel, null, 2) : "No model yet."}
            </pre>
          </div>
        )}
        <div className="model-stats-grid">
          <p className="model-stat-pill">
            {projectModel
              ? `Model: ${projectModel.product.name} (${projectModel.components.length} components)`
              : "No model yet. Create a project to render preview."}
          </p>
          <p className="model-stat-pill">
            Active job: <strong>{jobId || "none"}</strong>
          </p>
          <p className="model-stat-pill">
            Batch summary: {completedAnalyses} complete, {failedAnalyses} failed
          </p>
        </div>
      </SectionCard>

      <SectionCard
        title="Hardware and warnings"
        subtitle="Diagram parity: infer assembly hardware and expose warnings"
      >
        {projectModel ? (
          <>
            <p>
              Hardware lines: <strong>{projectModel.hardware.length}</strong>
            </p>
            {projectModel.hardware.length > 0 ? (
              <ul>
                {projectModel.hardware.map((line) => (
                  <li key={`${line.code}-${line.qty}`}>
                    {line.code} x {line.qty}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No inferred hardware yet.</p>
            )}
            {projectModel.warnings.length > 0 ? (
              <>
                <StatusPill label="Warnings present" tone="warning" />
                <ul>
                  {projectModel.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </>
            ) : (
              <StatusPill label="No model warnings" tone="ok" />
            )}
          </>
        ) : (
          <p>Run the workflow to inspect hardware and topology warnings.</p>
        )}
      </SectionCard>

      <SectionCard title="Validation state" subtitle="Diagram parity: decision nodes and rule checks">
        {validation ? (
          <>
            <StatusPill label={validation.valid ? "Validation PASS" : "Validation FAIL"} tone={validation.valid ? "ok" : "error"} />
            {validation.errors.length > 0 ? (
              <ul>
                {validation.errors.map((error) => (
                  <li key={error}>{error}</li>
                ))}
              </ul>
            ) : null}
            {validation.warnings.length > 0 ? (
              <ul>
                {validation.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            ) : null}
          </>
        ) : (
          <p>No validation yet.</p>
        )}
      </SectionCard>

      <SectionCard title="Fabrication package preview" subtitle="Bridge to blueprint/BOM export route">
        {projectModel && projectId ? (() => {
          const fabrication = buildFabricationPackage(projectId, projectModel, validation);
          return (
            <div className="fabrication-preview-grid">
              <p>
                Product: <strong>{fabrication.blueprint.product_name}</strong>
              </p>
              <p>
                Dimensions: <strong>{fabrication.blueprint.dimensions_mm.width} x {fabrication.blueprint.dimensions_mm.height} x {fabrication.blueprint.dimensions_mm.depth} mm</strong>
              </p>
              <p>
                Shelves: <strong>{fabrication.blueprint.dimensions_mm.shelf_count}</strong>
              </p>
              <p>
                Nesting strategy: <strong>{fabrication.nesting.strategy}</strong>
              </p>
              <p>
                Estimated usage: <strong>{(fabrication.nesting.estimated_usage_ratio * 100).toFixed(1)}%</strong>
              </p>
              <p>
                BOM: <strong>{fabrication.bom.panel_count}</strong> panels, <strong>{fabrication.bom.hardware.length}</strong> hardware lines
              </p>
            </div>
          );
        })() : (
          <p>Create and update a project first.</p>
        )}
      </SectionCard>
    </div>
  );
}
