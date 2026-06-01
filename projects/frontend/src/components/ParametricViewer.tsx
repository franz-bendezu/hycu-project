import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { ProjectModel, ValidateResponse } from "../api/backend";

type ParametricViewerProps = {
  model: ProjectModel | null;
  validation: ValidateResponse | null;
};

type PanelMeshModel = {
  id: string;
  label: string;
  size: [number, number, number];
  position: [number, number, number];
  tone: "panel" | "shelf" | "back" | "fallback";
};

const SCALE_TO_METERS = 0.001;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function dimensionOrFallback(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function buildPanelMeshes(model: ProjectModel): PanelMeshModel[] {
  const modelWidth = dimensionOrFallback(model.product.target_width, 900);
  const modelHeight = dimensionOrFallback(model.product.target_height, 1200);
  const modelDepth = dimensionOrFallback(model.product.target_depth, 450);
  const thickness = clamp(model.product.material_thickness || 18, 8, 50);

  const shelfComponents = model.components.filter((component) =>
    component.kind.toLowerCase().includes("shelf")
  );
  const shelfIndexById = new Map<string, number>();
  shelfComponents.forEach((component, index) => shelfIndexById.set(component.id, index));

  return model.components.map((component, index) => {
    const signature = `${component.kind} ${component.id}`.toLowerCase();
    const defaultWidth = clamp(modelWidth * 0.5, thickness, modelWidth);
    const defaultHeight = clamp(modelHeight * 0.5, thickness, modelHeight);
    const defaultDepth = clamp(modelDepth * 0.5, thickness, modelDepth);
    const compWidth = clamp(dimensionOrFallback(component.width, defaultWidth), thickness, modelWidth);
    const compHeight = clamp(dimensionOrFallback(component.height, defaultHeight), thickness, modelHeight);
    const compDepth = clamp(dimensionOrFallback(component.depth, defaultDepth), thickness, modelDepth);

    if (signature.includes("left") && signature.includes("leg")) {
      return {
        id: component.id,
        label: component.kind,
        size: [compWidth * SCALE_TO_METERS, compHeight * SCALE_TO_METERS, compDepth * SCALE_TO_METERS],
        position: [
          ((-modelWidth / 2) + compWidth / 2 + thickness) * SCALE_TO_METERS,
          ((-modelHeight / 2) + compHeight / 2) * SCALE_TO_METERS,
          (signature.includes("back")
            ? (-modelDepth / 2) + compDepth / 2 + thickness
            : (modelDepth / 2) - compDepth / 2 - thickness) * SCALE_TO_METERS,
        ],
        tone: "panel",
      };
    }

    if (signature.includes("right") && signature.includes("leg")) {
      return {
        id: component.id,
        label: component.kind,
        size: [compWidth * SCALE_TO_METERS, compHeight * SCALE_TO_METERS, compDepth * SCALE_TO_METERS],
        position: [
          ((modelWidth / 2) - compWidth / 2 - thickness) * SCALE_TO_METERS,
          ((-modelHeight / 2) + compHeight / 2) * SCALE_TO_METERS,
          (signature.includes("back")
            ? (-modelDepth / 2) + compDepth / 2 + thickness
            : (modelDepth / 2) - compDepth / 2 - thickness) * SCALE_TO_METERS,
        ],
        tone: "panel",
      };
    }

    if (signature.includes("left")) {
      return {
        id: component.id,
        label: component.kind,
        size: [thickness * SCALE_TO_METERS, modelHeight * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [
          ((-modelWidth / 2) + thickness / 2) * SCALE_TO_METERS,
          0,
          0,
        ],
        tone: "panel",
      };
    }

    if (signature.includes("right")) {
      return {
        id: component.id,
        label: component.kind,
        size: [thickness * SCALE_TO_METERS, modelHeight * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [
          ((modelWidth / 2) - thickness / 2) * SCALE_TO_METERS,
          0,
          0,
        ],
        tone: "panel",
      };
    }

    if (signature.includes("top")) {
      return {
        id: component.id,
        label: component.kind,
        size: [modelWidth * SCALE_TO_METERS, thickness * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [0, ((modelHeight / 2) - thickness / 2) * SCALE_TO_METERS, 0],
        tone: "panel",
      };
    }

    if (signature.includes("bottom") || signature.includes("base")) {
      return {
        id: component.id,
        label: component.kind,
        size: [modelWidth * SCALE_TO_METERS, thickness * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [0, ((-modelHeight / 2) + thickness / 2) * SCALE_TO_METERS, 0],
        tone: "panel",
      };
    }

    if (signature.includes("back")) {
      return {
        id: component.id,
        label: component.kind,
        size: [
          Math.max(thickness, modelWidth - thickness * 2) * SCALE_TO_METERS,
          Math.max(thickness, modelHeight - thickness * 2) * SCALE_TO_METERS,
          thickness * SCALE_TO_METERS,
        ],
        position: [0, 0, ((-modelDepth / 2) + thickness / 2) * SCALE_TO_METERS],
        tone: "back",
      };
    }

    if (signature.includes("front_panel") || signature.includes("front apron")) {
      return {
        id: component.id,
        label: component.kind,
        size: [
          Math.max(thickness, compWidth) * SCALE_TO_METERS,
          Math.max(thickness, compHeight) * SCALE_TO_METERS,
          Math.max(thickness, compDepth) * SCALE_TO_METERS,
        ],
        position: [
          0,
          ((-modelHeight / 2) + compHeight / 2 + thickness * 1.2) * SCALE_TO_METERS,
          ((modelDepth / 2) - compDepth / 2 - thickness) * SCALE_TO_METERS,
        ],
        tone: "panel",
      };
    }

    if (signature.includes("shelf")) {
      const shelfIndex = shelfIndexById.get(component.id) || 0;
      const shelfCount = Math.max(shelfComponents.length, 1);
      const shelfY = ((shelfIndex + 1) / (shelfCount + 1)) * (modelHeight - thickness) - modelHeight / 2;
      return {
        id: component.id,
        label: component.kind,
        size: [
          Math.max(thickness, modelWidth - thickness * 2) * SCALE_TO_METERS,
          Math.max(thickness, compHeight) * SCALE_TO_METERS,
          Math.max(thickness, modelDepth - thickness * 2) * SCALE_TO_METERS,
        ],
        position: [0, shelfY * SCALE_TO_METERS, 0],
        tone: "shelf",
      };
    }

    const jitterX = ((index % 3) - 1) * thickness * 1.3;
    const jitterY = (Math.floor(index / 3) % 3 - 1) * thickness * 1.3;
    return {
      id: component.id,
      label: component.kind,
      size: [compWidth * SCALE_TO_METERS, compHeight * SCALE_TO_METERS, compDepth * SCALE_TO_METERS],
      position: [jitterX * SCALE_TO_METERS, jitterY * SCALE_TO_METERS, 0],
      tone: "fallback",
    };
  });
}

function toneColor(tone: PanelMeshModel["tone"], selected: boolean): string {
  if (selected) {
    return "#f4cb8f";
  }
  if (tone === "back") {
    return "#cddce8";
  }
  if (tone === "shelf") {
    return "#e8b976";
  }
  if (tone === "fallback") {
    return "#d9b07f";
  }
  return "#eec98c";
}

export function ParametricViewer({ model, validation }: ParametricViewerProps): React.JSX.Element {
  const [selectedId, setSelectedId] = useState<string>("");

  const panels = useMemo(() => {
    if (!model) {
      return [];
    }
    return buildPanelMeshes(model);
  }, [model]);

  const selectedPanel = panels.find((panel) => panel.id === selectedId);

  if (!model) {
    return (
      <div className="parametric-viewer" role="img" aria-label="No model loaded">
        <p className="parametric-empty">No model yet. Create and update a project to render workspace geometry.</p>
      </div>
    );
  }

  return (
    <div className="parametric-viewer-shell">
      <div className="parametric-viewer" role="img" aria-label="Interactive 3D parametric furniture model">
        <Canvas camera={{ position: [1.5, 1.1, 1.6], fov: 52 }}>
          <color attach="background" args={["#f7fbff"]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[2.5, 3, 2]} intensity={0.9} />
          <directionalLight position={[-1.5, 1.2, -2]} intensity={0.35} />

          {panels.map((panel) => (
            <mesh
              key={panel.id}
              position={panel.position}
              onClick={(event) => {
                event.stopPropagation();
                setSelectedId(panel.id);
              }}
            >
              <boxGeometry args={panel.size} />
              <meshStandardMaterial color={toneColor(panel.tone, panel.id === selectedId)} />
            </mesh>
          ))}

          <gridHelper args={[3, 24, "#9fb9cc", "#dce7ef"]} position={[0, -0.65, 0]} />
          <OrbitControls makeDefault enablePan enableZoom enableRotate />
        </Canvas>
      </div>

      <div className="viewer-meta-row" aria-live="polite">
        <p>
          Components: <strong>{model.components.length}</strong>
        </p>
        <p>
          Selection: <strong>{selectedPanel?.label || "none"}</strong>
        </p>
        <p>
          Validation: <strong>{validation ? (validation.valid ? "pass" : "fail") : "not run"}</strong>
        </p>
      </div>
      {!validation?.valid && validation ? (
        <p className="viewer-validation-hint">Validation has failing rules. Run adjustments and validate again to clear highlighted issues.</p>
      ) : null}
    </div>
  );
}