import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { ProjectModel, ValidateResponse } from "../api/backend";
import { buildPanelMeshes, isFrontLikeComponent } from "../domain/viewer/layout";
import { buildHardwareMarkers } from "../domain/viewer/hardware";
import { MAX_RENDER_HARDWARE, HardwareMarker, PanelMeshModel } from "../domain/viewer/models";

type ParametricViewerProps = {
  model: ProjectModel | null;
  validation: ValidateResponse | null;
};


function toneColor(tone: PanelMeshModel["tone"], selected: boolean): string {
  if (selected) {
    return "#f4cb8f";
  }
  if (tone === "door") {
    return "#7f6740";
  }
  if (tone === "drawer") {
    return "#9e7f4f";
  }
  if (tone === "divider") {
    return "#8d6f42";
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
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [showHardware, setShowHardware] = useState<boolean>(true);
  const [explode, setExplode] = useState<number>(0);

  const panels = useMemo(() => {
    if (!model) {
      return [];
    }
    return buildPanelMeshes(model);
  }, [model]);

  const selectedPanel = panels.find((panel) => panel.id === selectedId);
  const hardwareMarkers = useMemo(() => {
    if (!model) {
      return [];
    }
    return buildHardwareMarkers(model, panels);
  }, [model, panels]);
  const hardwareLineCount = useMemo(() => {
    if (!model) {
      return 0;
    }
    return model.hardware.reduce((acc, line) => acc + Math.max(0, line.qty || 0), 0);
  }, [model]);
  const frontLikeComponentCount = useMemo(() => {
    if (!model) {
      return 0;
    }
    return model.components.filter((component) => isFrontLikeComponent(component)).length;
  }, [model]);
  const explicitMountSlotCount = useMemo(() => {
    if (!model) {
      return 0;
    }
    return model.hardware.reduce((acc, line) => {
      const qty = Math.max(0, Math.min(line.qty || 0, MAX_RENDER_HARDWARE));
      const targets = line.mount_targets?.length || 0;
      return acc + Math.min(qty, targets);
    }, 0);
  }, [model]);
  const missingMountSlotCount = Math.max(0, hardwareLineCount - explicitMountSlotCount);

  if (!model) {
    return (
      <div className="parametric-viewer" role="img" aria-label="No model loaded">
        <p className="parametric-empty">No model yet. Create and update a project to render workspace geometry.</p>
      </div>
    );
  }

  return (
    <div className="parametric-viewer-shell">
      <div className="viewer-toolbar" aria-label="3D viewer controls">
        <label className="viewer-toolbar-item" htmlFor="auto-rotate-toggle">
          <input
            id="auto-rotate-toggle"
            type="checkbox"
            checked={autoRotate}
            onChange={(event) => setAutoRotate(event.currentTarget.checked)}
          />
          Auto rotate
        </label>
        <label className="viewer-toolbar-item" htmlFor="show-grid-toggle">
          <input
            id="show-grid-toggle"
            type="checkbox"
            checked={showGrid}
            onChange={(event) => setShowGrid(event.currentTarget.checked)}
          />
          Show grid
        </label>
        <label className="viewer-toolbar-item" htmlFor="show-hardware-toggle">
          <input
            id="show-hardware-toggle"
            type="checkbox"
            checked={showHardware}
            onChange={(event) => setShowHardware(event.currentTarget.checked)}
          />
          Show hardware
        </label>
        <label className="viewer-toolbar-item slider" htmlFor="explode-slider">
          Exploded view
          <input
            id="explode-slider"
            type="range"
            min={0}
            max={100}
            step={1}
            value={explode}
            onChange={(event) => setExplode(Number(event.currentTarget.value))}
          />
        </label>
      </div>

      <div className="parametric-viewer" role="img" aria-label="Interactive 3D parametric furniture model">
        <Canvas camera={{ position: [1.5, 1.1, 1.6], fov: 52 }}>
          <color attach="background" args={["#f7fbff"]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[2.5, 3, 2]} intensity={0.9} />
          <directionalLight position={[-1.5, 1.2, -2]} intensity={0.35} />

          {panels.map((panel) => (
            <mesh
              key={panel.id}
              position={[
                panel.position[0] * (1 + explode / 140),
                panel.position[1] * (1 + explode / 140),
                panel.position[2] + (explode / 2500),
              ]}
              rotation={panel.rotation}
              onClick={(event) => {
                event.stopPropagation();
                setSelectedId(panel.id);
              }}
            >
              <boxGeometry args={panel.size} />
              <meshStandardMaterial color={toneColor(panel.tone, panel.id === selectedId)} />
            </mesh>
          ))}

          {showHardware
            ? hardwareMarkers.map((marker) => (
                <mesh key={marker.id} position={marker.position}>
                  {marker.kind === "screw" ? <cylinderGeometry args={[marker.size * 0.36, marker.size * 0.36, marker.size * 1.45, 16]} /> : null}
                  {marker.kind === "cam_lock" ? <sphereGeometry args={[marker.size * 0.52, 18, 18]} /> : null}
                  {marker.kind === "slide" ? <boxGeometry args={[marker.size * 2.2, marker.size * 0.55, marker.size * 1.05]} /> : null}
                  {marker.kind === "hinge" ? <torusGeometry args={[marker.size * 0.58, marker.size * 0.2, 12, 24]} /> : null}
                  {marker.kind === "generic" ? <octahedronGeometry args={[marker.size * 0.62, 0]} /> : null}
                  <meshStandardMaterial color={marker.color} emissive={marker.color} emissiveIntensity={0.24} />
                </mesh>
              ))
            : null}

          {showGrid ? <gridHelper args={[3, 24, "#9fb9cc", "#dce7ef"]} position={[0, -0.65, 0]} /> : null}
          <OrbitControls makeDefault enablePan enableZoom enableRotate autoRotate={autoRotate} autoRotateSpeed={0.9} />
        </Canvas>
      </div>

      <div className="viewer-meta-row" aria-live="polite">
        <p>
          Components: <strong>{model.components.length}</strong>
        </p>
        <p>
          Hardware lines: <strong>{model.hardware.length}</strong>
        </p>
        <p>
          Hardware qty: <strong>{hardwareLineCount}</strong>
        </p>
        <p>
          Explicit mounts: <strong>{explicitMountSlotCount}</strong>
        </p>
        <p>
          Missing mounts: <strong>{missingMountSlotCount}</strong>
        </p>
        <p>
          Hardware assets: <strong>{showHardware ? hardwareMarkers.length : 0}</strong>
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
      {showHardware && hardwareLineCount > 0 && hardwareMarkers.length === 0 && frontLikeComponentCount > 0 ? (
        <p className="viewer-validation-hint">Hardware rendering is strict: add mount_targets (component_id, face, local_x, local_y, local_z) to each hardware line.</p>
      ) : null}
      {showHardware && hardwareLineCount > 0 && hardwareMarkers.length === 0 && frontLikeComponentCount === 0 ? (
        <p className="viewer-validation-hint">No front components (door/drawer/front) exist in this model, so strict hardware rendering hides markers on structural parts.</p>
      ) : null}
      <div className="viewer-hardware-legend" aria-label="Hardware legend">
        <span className="viewer-hardware-pill viewer-hardware-pill-screw">Screw/Bolt</span>
        <span className="viewer-hardware-pill viewer-hardware-pill-cam">Cam lock</span>
        <span className="viewer-hardware-pill viewer-hardware-pill-slide">Slide/Rail</span>
        <span className="viewer-hardware-pill viewer-hardware-pill-hinge">Hinge</span>
        <span className="viewer-hardware-pill viewer-hardware-pill-generic">Generic</span>
      </div>
    </div>
  );
}