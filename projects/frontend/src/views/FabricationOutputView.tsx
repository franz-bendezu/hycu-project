import { useState } from "react";

import { downloadProjectArtifact } from "../api/backend";
import { buildFabricationPackage } from "../domain/fabrication";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";
import { SectionCard } from "../components/SectionCard";

export function FabricationOutputView(): React.JSX.Element {
  const { projectId, projectModel, validation, queuedImages } = useVisionWorkflow();
  const [downloadMessage, setDownloadMessage] = useState("");
  const [downloadBusy, setDownloadBusy] = useState(false);
  const completedAnalyses = queuedImages.filter((item) => item.status === "complete").length;
  const failedAnalyses = queuedImages.filter((item) => item.status === "failed").length;

  const output =
    projectId && projectModel ? buildFabricationPackage(projectId, projectModel, validation) : null;

  async function onDownload(kind: "blueprint" | "bom" | "nesting" | "package"): Promise<void> {
    if (!projectId) {
      setDownloadMessage("Create or open a project first.");
      return;
    }

    setDownloadBusy(true);
    setDownloadMessage("");
    try {
      const { blob, fileName } = await downloadProjectArtifact(projectId, kind);
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = fileName;
      anchor.click();
      URL.revokeObjectURL(objectUrl);
      setDownloadMessage(`${kind} export downloaded.`);
    } catch (error) {
      setDownloadMessage((error as Error).message);
    } finally {
      setDownloadBusy(false);
    }
  }

  return (
    <section className="content-grid" style={{ gridTemplateColumns: "1fr" }}>
      <SectionCard
        title="Fabrication output"
        subtitle="Blueprint, BOM, and nesting representation from current model"
      >
        {output ? (
          <>
            <div className="fabrication-kpi-grid" aria-label="Fabrication summary">
              <article className="fabrication-kpi-card">
                <h4>Analysis</h4>
                <p>
                  <strong>{completedAnalyses}</strong> complete / <strong>{failedAnalyses}</strong> failed
                </p>
              </article>
              <article className="fabrication-kpi-card">
                <h4>Sheet usage</h4>
                <p>
                  <strong>{(output.nesting.estimated_usage_ratio * 100).toFixed(1)}%</strong>
                </p>
              </article>
              <article className="fabrication-kpi-card">
                <h4>Panels</h4>
                <p>
                  <strong>{output.bom.panel_count}</strong>
                </p>
              </article>
              <article className="fabrication-kpi-card">
                <h4>Hardware lines</h4>
                <p>
                  <strong>{output.bom.hardware.length}</strong>
                </p>
              </article>
            </div>

            <div className="fabrication-block">
              <h4>Nesting plan</h4>
              <p style={{ marginBottom: "0.4rem" }}>
                Strategy: <strong>{output.nesting.strategy}</strong>
              </p>
              <p style={{ marginBottom: "0.4rem" }}>
                Sheet size: <strong>{output.nesting.sheet_size_mm.width} x {output.nesting.sheet_size_mm.height} mm</strong>
              </p>
              <div className="fabrication-scroll">
                <table className="fabrication-table">
                  <thead>
                    <tr>
                      <th>Panel</th>
                      <th>Orientation</th>
                      <th>Size (mm)</th>
                      <th>Position (x, y)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {output.nesting.panels.map((panel) => (
                      <tr key={panel.id}>
                        <td>{panel.id}</td>
                        <td>{panel.orientation}</td>
                        <td>{panel.width} x {panel.height}</td>
                        <td>{panel.x}, {panel.y}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="fabrication-block">
              <h4>Bill of materials</h4>
              {output.bom.hardware.length > 0 ? (
                <div className="fabrication-scroll">
                  <table className="fabrication-table">
                    <thead>
                      <tr>
                        <th>Hardware code</th>
                        <th>Quantity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {output.bom.hardware.map((line) => (
                        <tr key={`${line.code}-${line.qty}`}>
                          <td>{line.code}</td>
                          <td>{line.qty}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p>No hardware lines inferred yet.</p>
              )}
            </div>

            <div className="fabrication-actions">
              <button type="button" className="secondary" onClick={() => void onDownload("blueprint")} disabled={downloadBusy}>
                Download blueprint
              </button>
              <button type="button" className="secondary" onClick={() => void onDownload("bom")} disabled={downloadBusy}>
                Download BOM
              </button>
              <button type="button" className="secondary" onClick={() => void onDownload("nesting")} disabled={downloadBusy}>
                Download nesting
              </button>
              <button type="button" onClick={() => void onDownload("package")} disabled={downloadBusy}>
                Download package
              </button>
            </div>
            {downloadMessage ? <p style={{ marginBottom: 0 }}>{downloadMessage}</p> : null}
          </>
        ) : (
          <p>
            No project output available yet. Run analyze -&gt; create -&gt; update in Workspace route first.
          </p>
        )}
      </SectionCard>
    </section>
  );
}
