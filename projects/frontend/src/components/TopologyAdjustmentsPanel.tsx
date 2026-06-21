import { FormEvent } from "react";

import { ProductSpec } from "../api/backend";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";
import { SectionCard } from "./SectionCard";

type DraftSpec = Pick<
  ProductSpec,
  "target_width" | "target_height" | "target_depth" | "shelf_count" | "material_thickness"
>;

function parseNumberInput(raw: string): number {
  const value = Number(raw);
  return Number.isNaN(value) ? 0 : value;
}

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

export function TopologyAdjustmentsPanel(): React.JSX.Element {
  const {
    onUpdateProject,
    projectId,
    draftSpec,
    setDraftSpec,
    appearance,
    setAppearance,
  } = useVisionWorkflow();

  function handleNumberField(key: keyof DraftSpec, event: FormEvent<HTMLInputElement>): void {
    setDraftSpec(updateDraftValue(draftSpec, key, event.currentTarget.value));
  }

  return (
    <SectionCard
      title="Topology adjustments"
      subtitle="Update dimensions and structure for the current project model"
    >
      <form onSubmit={onUpdateProject}>
        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="width" style={{ display: "block" }}>Target width: {draftSpec?.target_width}mm</label>
          <input
            id="width"
            type="range"
            style={{ width: "100%" }}
            value={draftSpec?.target_width ?? 800}
            onChange={(e) => handleNumberField("target_width", e)}
            disabled={!projectId}
            min={300}
            max={2400}
            step={10}
          />
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="height" style={{ display: "block" }}>Target height: {draftSpec?.target_height}mm</label>
          <input
            id="height"
            type="range"
            style={{ width: "100%" }}
            value={draftSpec?.target_height ?? 1200}
            onChange={(e) => handleNumberField("target_height", e)}
            disabled={!projectId}
            min={300}
            max={2800}
            step={10}
          />
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="depth" style={{ display: "block" }}>Target depth: {draftSpec?.target_depth}mm</label>
          <input
            id="depth"
            type="range"
            style={{ width: "100%" }}
            value={draftSpec?.target_depth ?? 450}
            onChange={(e) => handleNumberField("target_depth", e)}
            disabled={!projectId}
            min={200}
            max={1200}
            step={10}
          />
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="shelf-count" style={{ display: "block" }}>Shelf count: {draftSpec?.shelf_count}</label>
          <input
            id="shelf-count"
            type="range"
            style={{ width: "100%" }}
            value={draftSpec?.shelf_count ?? 3}
            onChange={(e) => handleNumberField("shelf_count", e)}
            disabled={!projectId}
            min={0}
            max={12}
            step={1}
          />
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="material-thickness" style={{ display: "block" }}>Material thickness: {draftSpec?.material_thickness}mm</label>
          <input
            id="material-thickness"
            type="range"
            style={{ width: "100%" }}
            value={draftSpec?.material_thickness ?? 18}
            onChange={(e) => handleNumberField("material_thickness", e)}
            disabled={!projectId}
            min={8}
            max={50}
            step={1}
          />
        </div>

        <label htmlFor="finish">Finish</label>
        <select
          id="finish"
          value={appearance.finish}
          onChange={(event) =>
            setAppearance({ finish: event.currentTarget.value as "matte" | "satin" | "gloss" })
          }
          disabled={!projectId}
        >
          <option value="matte">Matte</option>
          <option value="satin">Satin</option>
          <option value="gloss">Gloss</option>
        </select>

        <p style={{ margin: 0, fontSize: "0.85rem", opacity: 0.8 }}>
          Finish selection is frontend-only until backend appearance fields are enabled.
        </p>

        <p style={{ margin: 0, fontSize: "0.85rem", opacity: 0.8 }}>
          The model preview updates in real time while you edit sliders. Changes are saved only when you click the button below.
        </p>

        <button type="submit" disabled={!projectId || !draftSpec}>
          Save topology updates
        </button>
      </form>
    </SectionCard>
  );
}
